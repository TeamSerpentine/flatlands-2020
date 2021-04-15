from argparse import ArgumentParser
from os import getcwd

from torch import device, load
from torch.multiprocessing import Manager, Queue, set_start_method

from serpentrain.reinforcement_learning.distributed.buffer_server import BufferServer
from serpentrain.reinforcement_learning.distributed.model_run_server import ModelInferenceServer
from serpentrain.reinforcement_learning.distributed.model_train_server import ModelTrainServer
from serpentrain.reinforcement_learning.distributed.utils import create_model

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--max_workers", type=int, default=140, required=False,
                        help="The amount of workers each inference server should spin up")
    parser.add_argument("--i_batch_size", type=int, default=16, required=False,
                        help="The batch size used during inference")
    parser.add_argument("--episode_length", type=int, default=1000, required=False,
                        help="The length of an episode")
    parser.add_argument("--t_batch_size", type=int, default=128, required=False,
                        help="The batch size used during training")
    parser.add_argument("--load", type=str, help="if specified, load a saved model", metavar="PATH")
    parser.add_argument("--save-dir", default=getcwd(), type=str,
                        help="the directory to periodically save the current model in (default: cwd)", metavar="PATH")
    args = parser.parse_args()

    set_start_method("spawn", force=True)

    device0 = device("cuda:0")
    device1 = device("cuda:1")
    device2 = device("cuda:2")

    # Setting up communication channels
    buffer_queue = Queue()
    sample_queue = Queue()
    model_shared_dict = Manager().dict()

    # Load checkpoint, if specified
    checkout, state_dict_model, state_dict_optimizer = None, None, None
    if args.load:
        checkpoint = load(args.load)
        state_dict_model = checkpoint.get("model")
        state_dict_optimizer = checkpoint.get("optimizer")

    model = create_model(state_dict_model)
    model.cpu()
    model_shared_dict.update(model.state_dict())

    # Creating Main processes and hand them their communication channels
    main_train_process = ModelTrainServer(sample_queue, model_shared_dict, state_dict_optimizer=state_dict_optimizer,
                                          save_dir=args.save_dir, device=device0, episode_length=args.episode_length)
    main_buffer_process = BufferServer(buffer_queue, sample_queue, batch_size=args.t_batch_size)
    main_inference_process = ModelInferenceServer(model_shared_dict, buffer_queue, device1,
                                                  batch_size=args.i_batch_size,
                                                  max_workers=args.max_workers,
                                                  episode_length=args.episode_length)

    # Starting processes
    main_train_process.start()
    main_buffer_process.start()
    main_inference_process.start()

    try:
        while True:
            assert main_train_process.is_alive()
            assert main_buffer_process.is_alive()
            assert main_inference_process.is_alive()
    except AssertionError as e:
        print("One of the sub processes died")
    finally:
        main_inference_process.shutdown = True
        main_buffer_process.shutdown = True
        main_train_process.shutdown = True
        main_inference_process.join()
        main_buffer_process.join()
        main_train_process.join()
