from queue import Empty

from torch.multiprocessing import Process

from serpentrain.reinforcement_learning.distributed.utils import create_replay_buffer
from serpentrain.reinforcement_learning.memory.trajectory import Trajectory


class BufferServer(Process):
    """
    Buffer servers take trajectories from the buffer queue
    Processes the trajectories
    Places samples in the sample queue
    """

    def __init__(self, buffer_queue, sample_queue, batch_size):
        super().__init__()
        self.min_sample_qsize = 100
        self.max_iterations = 50
        self.max_sample_qsize = 1_000

        self.batch_size = batch_size

        self.buffer_queue = buffer_queue
        self.sample_queue = sample_queue
        self.shutdown = False

        self.replay_buffer = create_replay_buffer()

    def run(self) -> None:
        print(f"Buffer server started")
        while not self.shutdown:
            print(f"Pulling from backlog, currently it is {self.buffer_queue.qsize()} long")
            for i in range(10):
                trajectory = self.pull_from_backlog()
                self.process(trajectory)
            self.push_to_sample()

    def pull_from_backlog(self):
        tries = 0
        while not self.shutdown:
            try:
                return self.buffer_queue.get(True, 60)
            except Empty:
                if self.shutdown:
                    print(f"Shutdown detected in Buffer Server")
                    break
                print(f"Buffer Queue Empty for more then 60 seconds,\n"
                      f"retrying (try: {tries})")
                tries += 1

    def process(self, trajectory: Trajectory):
        for state, action, reward, next_state, done in trajectory.as_rows():
            self.replay_buffer.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

    def push_to_sample(self):
        try:
            iterations = 0
            while self.sample_queue.qsize() < self.min_sample_qsize and iterations < self.max_iterations:
                self.sample_queue.put(self.replay_buffer.sample(self.batch_size))
                iterations += 1
                if self.sample_queue.qsize() > self.max_sample_qsize:
                    break
        except ValueError:
            print("Memory not big enough to push batch")
