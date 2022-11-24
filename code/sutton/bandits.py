import numpy as np

rng = np.random.default_rng()


class BanditBase:
    def __init__(self, number_of_actions: int, reward_deviation: float, eps: float) -> None:
        self.number_of_actions = number_of_actions
        self.reward_deviation = reward_deviation
        self.eps = eps

    def initialize_state(self):
        """Initializes and resets object state"""
        self.total_reward = 0
        self.estimated_action_values = np.zeros((self.number_of_actions,))
        self.true_action_values = rng.normal(size=(self.number_of_actions,))
        self.action_frequencies = np.zeros((self.number_of_actions,))
        self.optimal_action = self.get_optimal_action()
        self.optimal_actions_selected = []
        self.rewards = []

    def take_action(self) -> int:
        """Selects action based on policy"""
        if rng.uniform() < 1 - self.eps:
            max_value_action_indexes = np.argwhere(self.estimated_action_values == np.amax(self.estimated_action_values)).flatten()
            action = rng.choice(max_value_action_indexes)
        else:
            action = rng.integers(self.number_of_actions)
        self.action_frequencies[action] += 1
        return action
    
    def change_true_action_values(self):
        raise NotImplementedError

    def get_reward(self, action: int) -> float:
        """For given action calculates reward"""
        return rng.normal(loc=self.true_action_values[action], scale=self.reward_deviation)

    def get_step_size(self, action: int) -> float:
        """For given action returns step size."""
        raise NotImplementedError

    def update_action_value_estimate(self, action: int, reward: float) -> None:
        """Updates action value estimate provided action's last reward value."""
        alpha = self.get_step_size(action)
        q_old = self.estimated_action_values[action]
        q_new = q_old + alpha * (reward - q_old)
        self.estimated_action_values[action] = q_new

    def get_optimal_action(self) -> int:
        """Returns action with highest true action value."""
        return np.argmax(self.true_action_values)

    def get_estimated_optimal_action(self) -> int:
        """Returns action with highest estimated action value."""
        return np.argmax(self.estimated_action_values)

    def is_optimal_action(self, action: int) -> bool:
        """Checks if the selected action is the action with the highest true action value"""
        return bool(self.optimal_action == action)

    def run(self, total_steps: int) -> None:
        """Runs action value estimation loop"""
        self.initialize_state()
        step = 0
        while step < total_steps:
            action = self.take_action()
            reward = self.get_reward(action)
            self.update_action_value_estimate(action, reward)
            self.rewards.append(reward)
            self.optimal_actions_selected.append(self.is_optimal_action(action))
            self.change_true_action_values()
            step += 1


class BanditStationarySampleAverageStep(BanditBase):
    def __init__(self, number_of_actions: int, reward_deviation: float, eps: float) -> None:
        super().__init__(number_of_actions, reward_deviation, eps)

    def change_true_action_values(self):
        """True action values do not change over time"""
        pass

    def get_step_size(self, action: int) -> float:
        """For given action returns step size."""
        return 1 / self.action_frequencies[action]


class BanditNonStationarySampleAverageStep(BanditBase):
    def __init__(self, number_of_actions: int, reward_deviation: float, eps: float) -> None:
        super().__init__(number_of_actions, reward_deviation, eps)

    def change_true_action_values(self):
        """True action values change by some random walk"""
        self.true_action_values = self.true_action_values + rng.normal(size=(self.number_of_actions,))

    def get_step_size(self, action: int) -> float:
        """For given action returns step size."""
        return 1 / self.action_frequencies[action]


class BanditNonStationaryConstantStep(BanditBase):
    def __init__(self, number_of_actions: int, reward_deviation: float, eps: float, step_size:float) -> None:
        self.step_size = step_size
        super().__init__(number_of_actions, reward_deviation, eps)

    def change_true_action_values(self):
        """True action values change by some random walk"""
        self.true_action_values = self.true_action_values + rng.normal(size=(self.number_of_actions,))

    def get_step_size(self, action: int) -> float:
        """For given action returns step size."""
        return self.step_size