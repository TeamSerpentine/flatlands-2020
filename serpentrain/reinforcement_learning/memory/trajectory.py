class Trajectory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.done: int = -1

    def add_row(self, state, action, reward, done):
        if not (state is None or action is None or reward is None or done is None):
            if done:
                self.done = len(self.states)
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
        else:
            print("Illegal input detected:")
            print(f"state:  {state}\n"
                  f"action: {action}\n"
                  f"reward: {reward}\n"
                  f"done:   {done}")

    def as_rows(self):
        """
        Generator that returns each time step as a row of:
        state, action, reward, next_state, done
        """
        for i in range(len(self.states) - 1):
            # State, Action, Reward, Next State, Done Flag=0
            if i == self.done:
                return self.states[i], self.actions[i], self.rewards[i], self.states[i + 1], 1
            yield self.states[i], self.actions[i], self.rewards[i], self.states[i + 1], 0
