import time
import traceback
from queue import Empty
from threading import Thread

import torch
from torch.multiprocessing import Manager, Process, Queue

from serpentrain.reinforcement_learning.distributed.env_worker import EnvWorker
from serpentrain.reinforcement_learning.distributed.utils import create_model


class ModelInferenceServer(Process):
    """
    Model Inference Server
    Checks for latest model
    Pull from backlog
    Process batched states
    Return values via return dict
    """

    def __init__(self, model_shared_dict, buffer_queue, device, batch_size=16, max_workers=140, episode_length=1000):
        super(ModelInferenceServer, self).__init__()
        self.model_shared_dict = model_shared_dict
        self.workers = []
        self.pipes = []

        # Communication channels with workers
        self.backlog = Queue()
        self.return_dict = Manager().dict()
        self.buffer_queue = buffer_queue

        self.model = None
        self.device = device
        self.batch_size = batch_size
        self.max_workers = max_workers  # 140 seems to be the optimum
        self.episode_length = episode_length

        self.shutdown = False
        self.empty_count = 0

        self.avg_queue_size = 0
        self.avg_batch_size = 0
        self.avg_total_time = 0
        self.avg_batch_pull_time = 0
        self.avg_batch_infr_time = 0
        self.avg_batch_retu_time = 0

    def update_model(self):
        print("Updated model")
        if self.model is None:
            self.model = create_model(self.model_shared_dict, device=self.device)
        else:
            self.model.load_state_dict(self.model_shared_dict)
        self.model.eval()

    def get_batch(self):
        start_batch_pull_time = time.time()
        batch = []
        return_ids = []
        try:
            self.avg_queue_size = 0.9 * self.avg_queue_size + 0.1 * self.backlog.qsize()
            return_id, element = self.backlog.get(True, 60)
            return_ids.append(return_id)
            batch.append(element)
            while len(batch) < self.batch_size:
                return_id, element = self.backlog.get(True, 0.1)
                return_ids.append(return_id)
                batch.append(element)
                del return_id
                del element

        except Empty:
            self.empty_count += 1
            if len(self.workers) < self.max_workers:
                self.add_worker(1)
        except TimeoutError:
            print("60 seconds without anything being put in the backlog, cleaning up")
            self.cleanup()
        finally:
            self.avg_batch_size = 0.9 * self.avg_batch_size + 0.1 * len(batch)
            self.avg_batch_pull_time = 0.9 * self.avg_batch_pull_time + 0.1 * (time.time() - start_batch_pull_time)
            batch = torch.cat(batch, 0)

        return return_ids, batch

    def threaded_return_batch(self, return_ids, batch: torch.Tensor):
        x = Thread(target=self.return_batch, args=(return_ids, batch,))
        x.start()

    def return_batch(self, return_ids, batch: torch.Tensor):
        self.return_dict.update({return_id: element.cpu()
                                 for return_id, element in zip(return_ids, batch.unbind())})

    def add_worker(self, n):
        for i in range(n):
            worker = EnvWorker(self.backlog, self.return_dict, self.buffer_queue, len(self.workers),
                               self.episode_length)
            worker.start()
            self.workers.append(worker)

    def cleanup(self):
        for worker in self.workers:
            del worker

    def run(self):
        """
        Eternal loop that runs the inference
        """
        try:
            print("Model Run Server started!")
            self.add_worker(self.max_workers)
            while not self.shutdown:
                for _ in range(100):
                    # Update model every 100 forward passes
                    self.update_model()
                    for _ in range(1000):
                        total_start_time = time.time()
                        return_ids, batch = self.get_batch()
                        start_inf_time = time.time()
                        with torch.no_grad():
                            batch = batch.to(self.device)
                            out = self.model.forward(batch)
                        del batch
                        self.avg_batch_infr_time = 0.9 * self.avg_batch_infr_time + 0.1 * (time.time() - start_inf_time)
                        start_return_time = time.time()
                        self.threaded_return_batch(return_ids, out)
                        self.avg_batch_retu_time = 0.9 * self.avg_batch_retu_time + 0.1 * (
                                time.time() - start_return_time)
                        self.avg_total_time = 0.9 * self.avg_total_time + 0.1 * (time.time() - total_start_time)
                print(f"Averages with {len(self.workers)}\n"
                      f"queue size: {self.avg_queue_size}\n"
                      f"batch size: {self.avg_batch_size}\n"
                      f"pull  time: {self.avg_batch_pull_time}\n"
                      f"infer time: {self.avg_batch_infr_time}\n"
                      f"return time: {self.avg_batch_retu_time}\n"
                      f"total time: {self.avg_total_time}")
        except Exception as e:
            traceback.print_exc()
            print(f"{e} in model run server")
        finally:
            for worker in self.workers:
                worker.shutdown = True
