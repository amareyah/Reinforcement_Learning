import numpy as np

rng = np.random.default_rng()

hyper_params = {}


class Bandit:
    def __init__(self, number_of_actions: int, reward_deviation: float, eps: float) -> None:
        self.number_of_actions = number_of_actions
        self.reward_deviation = reward_deviation
        self.eps = eps
        # self.initialize_state()
    
    def initialize_state(self):
        self.total_reward = 0
        self.estimated_action_values = np.zeros((self.number_of_actions,))
        self.true_action_values = rng.normal(size=(self.number_of_actions,))
        self.optimal_action = self.get_optimal_action()
        self.optimal_actions_selected = []
        self.rewards = []


    def get_reward(self, action: int) -> float:
        """For given action calculates reward"""
        return rng.normal(loc=self.true_action_values[action], scale=self.reward_deviation)

    def take_action(self) -> int:
        """Selects action based on policy"""
        if rng.uniform() < 1 - self.eps:
            max_value_action_indexes = np.argwhere(self.estimated_action_values == np.amax(self.estimated_action_values)).flatten()
            action_idx = rng.choice(max_value_action_indexes)
        else:
            action_idx = rng.integers(self.number_of_actions)
        return action_idx

    @staticmethod
    def get_step_size(step: int) -> float:
        return 1 / step

    def update_action_value_estimate(self, action: int, reward: float, step: int) -> None:
        alpha = self.get_step_size(step)
        q_old = self.estimated_action_values[action]
        q_new = q_old + alpha * (reward - q_old)
        self.estimated_action_values[action] = q_new

    def run(self, total_steps: int):
        self.initialize_state()
        for step in range(1, total_steps + 1):
            action = self.take_action()
            reward = self.get_reward(action)
            self.update_action_value_estimate(action, reward, step)
            self.rewards.append(reward)
            self.optimal_actions_selected.append(self.is_optimal_action(action))

    def get_optimal_action(self):
        return np.argmax(self.true_action_values)

    def get_estimated_optimal_action(self):
        return np.argmax(self.estimated_action_values)

    def is_optimal_action(self, action):
        return bool(self.optimal_action==action)