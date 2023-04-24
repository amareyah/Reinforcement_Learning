from typing import Tuple, List, Dict, Iterable, Union, Any
from pathlib import Path
import itertools
from multiprocessing import Pool

import numpy as np
import scipy as sp
import joblib

rng = np.random.default_rng()

data_type = None


class Environment:
    def __init__(self) -> None:
        self.discount_rate = 0.9

        self.max_storage = (20, 20)
        self.max_requests = (20, 20)
        self.max_returns = (20, 20)
        self.max_free_storage = 10
        self.max_car_moves = 5

        self.unit_rent_reward = 10
        self.unit_move_reward = -2
        self.storage_lot_reward = -4

        self.miu_request = (3, 4)
        self.miu_return = (3, 2)

        self.state_space = Environment._create_state_space(self.max_storage)
        self.action_space = Environment._create_action_space(self.max_car_moves)

        self.request_grids, self.return_grids = Environment._create_request_return_meshgrid(self.max_requests, self.max_returns)
        self.calculate_joined_probability_distribution()

    @staticmethod
    def _create_state_space(max_storage: Tuple) -> List[Tuple]:
        return list(itertools.product(*[np.arange(elem + 1) for elem in max_storage]))

    @staticmethod
    def _create_action_space(max_moves: int) -> np.ndarray:
        return np.arange(-max_moves, max_moves + 1)

    @staticmethod
    def _create_request_return_meshgrid(max_requests: Tuple, max_returns: Tuple) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        request = [np.arange(r + 1) for r in max_requests]
        returns = [np.arange(r + 1) for r in max_returns]
        mesh = np.meshgrid(*request, *returns, indexing="ij")
        return mesh[: len(max_requests)], mesh[len(max_requests) :]

    def calculate_joined_probability_distribution(self) -> None:
        q1_distribution = sp.stats.poisson.pmf(self.request_grids[0], self.miu_request[0])
        q2_distribution = sp.stats.poisson.pmf(self.request_grids[1], self.miu_request[1])
        u1_distribution = sp.stats.poisson.pmf(self.return_grids[0], self.miu_return[0])
        u2_distribution = sp.stats.poisson.pmf(self.return_grids[1], self.miu_return[1])
        q1_distribution[-1:, :, :, :] = sp.stats.poisson.sf(self.max_requests[0] - 1, self.miu_request[0])
        q2_distribution[:, -1:, :, :] = sp.stats.poisson.sf(self.max_requests[1] - 1, self.miu_request[1])
        u1_distribution[:, :, -1:, :] = sp.stats.poisson.sf(self.max_returns[0] - 1, self.miu_return[0])
        u2_distribution[:, :, :, -1:] = sp.stats.poisson.sf(self.max_returns[1] - 1, self.miu_return[1])
        # q2_distribution[:, -1:, :, :] === q2_distribution[slice(None, None),slice(-2,-1),slice(None),slice(None)]
        joined_probability_distribution = q1_distribution * q2_distribution * u1_distribution * u2_distribution
        self.joined_probability_distribution = joined_probability_distribution.astype(data_type)

    @staticmethod
    def _calculate_cars_available(current_state: Tuple[int, int], action: int, max_storage: Tuple) -> Tuple:
        cars_to_move = min(current_state[0], np.abs(action)) if action >= 0 else min(current_state[1], np.abs(action))
        ca = (
            np.clip(current_state[0] - np.sign(action) * cars_to_move, a_min=0, a_max=max_storage[0]),
            np.clip(current_state[1] + np.sign(action) * cars_to_move, a_min=0, a_max=max_storage[1]),
        )
        return ca

    @staticmethod
    def _calculate_expected_reward(
        cars_available: Tuple,
        action: int,
        unit_rent_reward: int,
        unit_move_reward: int,
        storage_lot_reward: int,
        max_free_storage: int,
        request_grids: List[np.ndarray],
        joined_probability_distribution: np.ndarray,
    ) -> float:
        # Comment this out to have initial version of the Jack's car rent problem.
        if action > 0:
            moving_reward = unit_move_reward * (action - 1)
        else:
            moving_reward = unit_move_reward * np.abs(action)
        storage_reward = np.sum([storage_lot_reward for ca_at_loc in cars_available if ca_at_loc > max_free_storage])
        # storage_reward = 0
        # moving_reward = unit_move_reward * np.abs(action)
        r = (
            (np.minimum(cars_available[0], request_grids[0]) + np.minimum(cars_available[1], request_grids[1])) * unit_rent_reward
            + moving_reward
            + storage_reward
        )
        expected_return = np.sum(np.multiply(r, joined_probability_distribution, dtype=data_type))
        return expected_return

    def calculate_expected_reward(self, state_action: Tuple[int, int]) -> float:
        current_state_idx, action_idx = state_action
        current_state = self.state_space[current_state_idx]
        action = self.action_space[action_idx]
        ca = Environment._calculate_cars_available(current_state, action, self.max_storage)
        expected_reward = Environment._calculate_expected_reward(
            ca,
            action,
            self.unit_rent_reward,
            self.unit_move_reward,
            self.storage_lot_reward,
            self.max_free_storage,
            self.request_grids,
            self.joined_probability_distribution,
        )
        return expected_reward

    @staticmethod
    def _calculate_next_state_grids(
        current_state: Tuple, action: int, request_grids: List[np.ndarray], return_grids: List[np.ndarray], max_storage: Tuple
    ) -> List[np.ndarray]:
        """Each grid from state grids corresponds to particular location (component of state).
        State component grid contains values of possible next states at the given location, given action, requests and returns for that location.
        Args:
            current_state (Tuple): state object;
            action (int): action object;
            request_grids (List[np.ndarray]): request mesh grid
            return_grids (List[np.ndarray]): return mesh grid
            max_storage (Tuple): max available storage by location

        Returns:
            List[np.ndarray]: The element of this list is the state grid for that location index.
        """
        ca = Environment._calculate_cars_available(current_state, action, max_storage)
        state_component_grids = [ca[i] - np.minimum(ca[i], request_grids[i]) + return_grids[i] for i in range(len(ca))]
        state_component_grids = [
            np.clip(state_component_grid, a_min=0, a_max=max_storage_per_location)
            for state_component_grid, max_storage_per_location in zip(state_component_grids, max_storage)
        ]
        return state_component_grids

    @staticmethod
    def _calculate_next_states_probabilities(
        state_component_grids: List[np.ndarray], joined_prob_mass: np.ndarray, state_space: List[Tuple]
    ) -> List[float]:
        next_states_probabilities = []
        for s in state_space:
            predicate = True
            for s_grid, s_component in zip(state_component_grids, s):
                predicate = predicate & (s_grid == s_component)
            next_states_probabilities.append(np.sum(joined_prob_mass[predicate], dtype=data_type))
        return next_states_probabilities

    def _calculate_state_action_next_states_probabilities(self, state_action: Tuple[int, int]) -> List[float]:
        current_state_idx, action_idx = state_action
        current_state = self.state_space[current_state_idx]
        action = self.action_space[action_idx]
        next_state_component_grids = Environment._calculate_next_state_grids(
            current_state, action, self.request_grids, self.return_grids, self.max_storage
        )
        next_state_probabilities = Environment._calculate_next_states_probabilities(
            next_state_component_grids, self.joined_probability_distribution, self.state_space
        )
        return next_state_probabilities

    def calculate_all_state_action_next_state_probability_distributions(self) -> np.ndarray:
        state_space_length = len(self.state_space)
        action_space_lenght = len(self.action_space)
        state_action_space = list(itertools.product(range(state_space_length), range(action_space_lenght)))
        with Pool() as pool:
            state_action_probability_distributions = pool.map(self._calculate_state_action_next_states_probabilities, state_action_space)
        state_action_probability_distributions_numpy = np.array(state_action_probability_distributions).reshape(
            state_space_length, action_space_lenght, -1
        )
        return state_action_probability_distributions_numpy

    def calculate_all_state_action_expected_rewards(self):
        state_space_length = len(self.state_space)
        action_space_lenght = len(self.action_space)
        state_action_space = list(itertools.product(range(state_space_length), range(action_space_lenght)))
        with Pool() as pool:
            state_action_expected_rewards = pool.map(self.calculate_expected_reward, state_action_space)
            state_action_expected_rewards_numpy = np.array(state_action_expected_rewards).reshape(state_space_length, action_space_lenght)
        return state_action_expected_rewards_numpy

    @staticmethod
    def save(obj: Any, save_path: str) -> None:
        with open(save_path, "wb") as out_file:
            joblib.dump(obj, out_file)

    @staticmethod
    def load(file_path: str) -> Any:
        with open(file_path, "rb") as f:
            obj = joblib.load(f)
        return obj

    def load_state_action_next_state_probability_distributions(self, file_path: str) -> None:
        self.state_action_probability_distributions = Environment.load(file_path)

    def load_state_action_expected_rewards(self, file_path: str) -> None:
        self.state_action_expected_rewards = Environment.load(file_path)


class Agent:
    def __init__(self, env: Environment) -> None:
        self.environment = env

    @staticmethod
    def _initialize_policy(state_space_size: int, action_space_size: int) -> np.ndarray:
        """Returns array where index corresponds to state index and value corresponds to action index."""
        return rng.integers(low=0, high=action_space_size, endpoint=False, size=(state_space_size,))

    @staticmethod
    def _initialize_state_values(state_space_size: int) -> np.ndarray:
        return rng.normal(
            size=(state_space_size),
        )

    def initialize_policy(self):
        self.policy = Agent._initialize_policy(len(self.environment.state_space), len(self.environment.action_space))

    def initialize_state_values(self):
        self.state_values = Agent._initialize_state_values(len(self.environment.state_space))

    def get_policy_action(self, state_idx: int) -> int:
        """For given state return policy action index"""
        return self.policy[state_idx]

    def policy_evaluation(self, gamma: float, theta: float, max_iter: int = 1000) -> None:
        """Calculates and updateds current state_values using current policy (Policy Evaluation).
        Args:
            gamma (float): discount rate
            theta (float): residue threshold between old and new state values.
        """
        state_space_len = self.environment.state_action_probability_distributions.shape[0]
        policy_action_indexes = [self.get_policy_action(state_idx) for state_idx in range(state_space_len)]
        current_iter = 0
        while True:
            state_value_probabilities = self.environment.state_action_probability_distributions[
                range(state_space_len), policy_action_indexes, :
            ]
            new_state_values = self.environment.state_action_expected_rewards[
                range(state_space_len), policy_action_indexes
            ] + gamma * np.dot(state_value_probabilities, self.state_values)
            delta = max(np.abs(new_state_values - self.state_values))
            if delta < theta:
                print(f"Policy_evaluation: Stoped by delta. Total iters: {current_iter}, Delta: {delta}")
                break
            if current_iter > max_iter:
                print(f"Policy_evaluation: Stoped by max_iter. Delta: {delta}")
                break
            self.state_values = new_state_values
            current_iter += 1

    def policy_improvement(self, gamma: float) -> np.ndarray:
        new_policy = np.argmax(
            self.environment.state_action_expected_rewards
            + gamma * np.dot(self.environment.state_action_probability_distributions, self.state_values),
            axis=1,
        )
        return new_policy

    def policy_iteration(self, gamma: float, theta: float, max_iter: int) -> Tuple[np.ndarray, np.ndarray]:
        self.initialize_policy()
        self.initialize_state_values()
        policy_is_stable = False
        current_iter = 0
        while not policy_is_stable:
            self.policy_evaluation(gamma, theta, max_iter)
            new_policy = self.policy_improvement(gamma)
            policy_is_stable = all(new_policy == self.policy)
            self.policy = new_policy
            current_iter += 1
            if current_iter > 1000:
                print(f"Policy Iteration: Stoped by max_iter.")
                break
        print(f"Policy Iteration: Stable policy reached. Total iters: {current_iter}")
        return self.policy, self.state_values
