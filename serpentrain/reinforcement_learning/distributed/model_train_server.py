import time
from os import path
from random import SystemRandom, seed, randint

import cv2
import numpy as np
import wandb
from torch import optim, save
from torch.multiprocessing import Process

from serpentrain.reinforcement_learning.distributed.utils import create_model, create_controller, create_env


class ModelTrainServer(Process):
    """
    Pulls samples from sample queue
    Train Model
    Share model via shared model dict
    """

    def __init__(self, sample_queue, model_shared_dict, state_dict_optimizer, save_dir, device="cuda:0",
                 episode_length=1000, num_eval_episodes=10, save_interval=30):
        super().__init__()
        self.save_interval = save_interval
        self.sample_queue = sample_queue
        self.device = device
        self.model_shared_dict = model_shared_dict
        self.save_dir = save_dir
        self.model = create_model(state_dict=model_shared_dict, device=device)
        self.episode_length = episode_length
        self.num_eval_episodes = num_eval_episodes
        self.state_dict_optimizer = state_dict_optimizer

        self.gamma = 0.99
        self.steps = 10_000

        self.shutdown = False

    def _initialize(self):
        # Initialize wandb
        wandb.init(project="flatland")

        # Create the controller so we can call the specific train function
        # The controller will not be used for anything else, therefore the model can be anything
        self.controller = create_controller(self.model)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.5e-4)

        if self.state_dict_optimizer:
            self.optimizer.load_state_dict(self.state_dict_optimizer)

    def run(self) -> None:
        self._initialize()
        time_last_save = time.time()  # in seconds
        epoch = 0
        wandb.watch(self.model)
        while not self.shutdown:
            print(f"Starting Training on {self.device}")
            start_time = time.time()
            self.train(epoch)
            print(f"Training Episode finished in {time.time() - start_time}")
            self.update_shared_model()

            print(f"Evaluate")
            self.evaluate()
            print(f"Finished Evaluating")

            time_current = time.time()
            if time_current - time_last_save >= self.save_interval * 60:
                time_last_save = time_current
                name = f"snapshot-{time.strftime('%Y%m%d-%H%M')}-epoch-{epoch}.pt"
                save({
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict()
                }, path.join(self.save_dir, name))
                print("Saved Snapshot")
            epoch += 1

    def train(self, epoch):
        """
        Does self.steps training steps
        """
        losses = []
        for i in range(self.steps):
            losses.append(self.train_step(i))
        total_loss = sum(losses)
        wandb.log({"Epoch Loss": total_loss / len(losses), "Epoch": epoch}, commit=False)

    def train_step(self, i):
        """
        Does a single batch training step
        """
        loss = self.controller.train(self.model, self, self.gamma, optimizer=self.optimizer)
        wandb.log({"Training loss": loss, "Step": i}, commit=False)
        return loss

    def evaluate(self):
        # Set a random state
        rng = SystemRandom()
        seed_ = rng.randint(0, 2 ** 32 - 1)
        seed(seed_)
        np.random.seed(seed_)

        # Create and Start Environment
        env = create_env(seed=seed_)

        # renderer = RenderTool(env, gl="PIL")

        # model_ref = ModelWrapper.wrap_model(self.backlog_queue, self.return_dict, self.id)
        self.controller.model = self.model

        env.number_of_agents = randint(1, 100)
        env.width = randint(10, 100)
        env.height = randint(10, 100)

        sum_score = 0
        sum_score_per_agent = 0
        sum_steps = 0
        sum_agents = 0
        sum_time = 0
        sum_agents_done = 0
        sum_agents_done_percentage = 0
        total_action_dict = {0: 0,
                             1: 0,
                             2: 0,
                             3: 0,
                             4: 0}

        for game in range(self.num_eval_episodes):
            start_time = time.time()
            print(f"Evaluating for {game} out of {self.num_eval_episodes} games.")
            obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True, random_seed=True)
            # renderer.reset()
            # images = [renderer.get_image()]
            score = 0
            # trajectories = [Trajectory() for _ in env.get_agent_handles()] # Not needed

            # Create and Start Controller
            self.controller.start_of_round(obs=obs, env=env)

            step = 0
            done = {}

            for step in range(self.episode_length):
                action_dict, processed_obs = self.controller.act(observation=obs, info=info, epsilon=0)
                obs, all_rewards, done, info = env.step(action_dict)

                for action in action_dict.values():
                    total_action_dict[action] = total_action_dict.get(action, 0) + 1

                score += sum(all_rewards)

                if done['__all__'] or self.shutdown:
                    break

            n_agents_done = sum(done.values()) - 1
            n_agents = env.get_num_agents()
            score_per_agent = score / n_agents
            agents_done_percentage = n_agents_done / n_agents

            sum_score += score
            sum_score_per_agent += score_per_agent
            sum_steps += step
            sum_agents += n_agents
            sum_agents_done += n_agents_done
            sum_agents_done_percentage += agents_done_percentage
            sum_time = time.time() - start_time

            print(f"Finished evaluation game #{game}.\n"
                  f"Score: {score}. #Agents: {n_agents}.\n"
                  f"Score per agent: {score_per_agent}."
                  f"Steps: {step}. Time: {time.time() - start_time}"
                  f"#Agents done: {n_agents_done}. %Agents done: {agents_done_percentage}.")


        avg_score = sum_score / self.num_eval_episodes
        avg_score_per_agent = sum_score_per_agent / self.num_eval_episodes
        avg_steps = sum_steps / self.num_eval_episodes
        avg_agents = sum_agents / self.num_eval_episodes
        avg_agents_done = sum_agents_done / self.num_eval_episodes
        avg_agents_done_percentage = sum_agents_done_percentage / self.num_eval_episodes

        print("=" * 20)
        print(f"Finished all evaluation games. Averages:")
        print(f"Score: {avg_score}. #Agents: {avg_agents}. "
              f"Score per agent: {avg_score_per_agent}. Steps: {avg_steps}. "
              f"#Agents done: {avg_agents_done}. %Agents done: {avg_agents_done_percentage}.\n"
              f"Total Time: {sum_time} per episode")
        print("=" * 20)

        data = [[label, val] for (label, val) in total_action_dict.items()]
        table = wandb.Table(data=data, columns=["action", "count"])

        wandb.log({
            "Action Distribution": wandb.plot.bar(table, "action", "count", title="Action Distribution"),
            "Average Score": avg_score,
            "Average Agents": avg_agents,
            "Score per Agent": avg_score_per_agent,
            "Average Steps": avg_steps,
            "Average Agents Done": avg_agents_done,
            "Percentage Agents Done": avg_agents_done_percentage}, commit=True)

    def update_model(self):
        print("Updated model")
        if self.model is None:
            self.model = create_model(self.model_shared_dict, device=self.device)
        else:
            self.model.load_state_dict(self.model_shared_dict)
        self.model.eval()

    def update_shared_model(self):
        print("Updating global model")
        cpu_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        self.model_shared_dict.update(cpu_state_dict)

    def sample(self):
        """
        Wrapper function for the controller train function
        """
        return self.sample_queue.get()

    @staticmethod
    def log_video(_images, episode):
        height, width, depth = _images[0].shape
        out = cv2.VideoWriter(f'flatland_{episode}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
        [out.write(cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)) for image in _images]
        out.release()
        wandb.log({"Replay": wandb.Video(f"flatland_{episode}.mp4")}, commit=False)
