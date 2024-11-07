from typing import Iterator
import jax
import numpy as np

from actsafe.common.double_buffer import double_buffer
from actsafe.rl.trajectory import TrajectoryData, Transition


class ReplayBuffer:
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        max_length: int,
        seed: int,
        capacity: int,
        batch_size: int,
        sequence_length: int,
        num_rewards: int,
    ):
        self.episode_id = 0
        self.dtype = np.float32
        self.obs_dtype = np.uint8
        self.observation = np.zeros(
            (
                capacity,
                max_length + 1,
            )
            + observation_shape,
            dtype=self.obs_dtype,
        )
        self.action = np.zeros(
            (
                capacity,
                max_length,
            )
            + action_shape,
            dtype=self.dtype,
        )
        self.reward = np.zeros(
            (capacity, max_length, num_rewards),
            dtype=self.dtype,
        )
        self.cost = np.zeros(
            (
                capacity,
                max_length,
            ),
            dtype=self.dtype,
        )
        self._valid_episodes = 0
        self.rs = np.random.RandomState(seed)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.id = 0

    def add(self, transition: Transition):
        capacity, max_length, _ = self.reward.shape
        batch_size = min(transition.observation.shape[0], capacity)
        # Discard data if batch size overflows capacity.
        end = min(self.episode_id + batch_size, capacity)
        episode_slice = slice(self.episode_id, end)
        if transition.reward.ndim == 1:
            transition = Transition(
                transition.observation,
                transition.next_observation,
                transition.action,
                transition.reward[..., None],
                transition.cost,
                transition.done,
                transition.terminal,
            )
        for data, val in zip(
            (self.action, self.reward, self.cost),
            (
                transition.action,
                transition.reward,
                transition.cost,
            ),
        ):
            data[episode_slice, self.id] = val[:batch_size].astype(self.dtype)
        self.observation[episode_slice, self.id] = transition.observation.astype(
            self.obs_dtype
        )
        if transition.terminal.any() or transition.done.any():
            assert transition.done.all()
            assert self.id + 1 == max_length
            observation = transition.next_observation[:batch_size, -1:]
            self.observation[episode_slice, self.id + 1] = observation.astype(
                self.obs_dtype
            )
            self.episode_id = (self.episode_id + batch_size) % capacity
            self._valid_episodes = min(self._valid_episodes + batch_size, capacity)
        self.id = (self.id + 1) % max_length

    def _sample_batch(
        self,
        batch_size: int,
        sequence_length: int,
        valid_episodes: int | None = None,
    ):
        if valid_episodes is not None:
            valid_episodes = valid_episodes
        else:
            valid_episodes = self._valid_episodes
        time_limit = self.observation.shape[1]
        assert time_limit > sequence_length
        while True:
            low = self.rs.choice(time_limit - sequence_length - 1, batch_size)
            timestep_ids = low[:, None] + np.tile(
                np.arange(sequence_length + 1),
                (batch_size, 1),
            )
            episode_ids = self.rs.choice(valid_episodes, size=batch_size)
            # Sample a sequence of length H for the actions, rewards and costs,
            # and a length of H + 1 for the observations (which is needed for
            # bootstrapping)
            a, r, c = [
                x[episode_ids[:, None], timestep_ids[:, :-1]]
                for x in (
                    self.action,
                    self.reward,
                    self.cost,
                )
            ]
            o = self.observation[episode_ids[:, None], timestep_ids]
            o, next_o = o[:, :-1], o[:, 1:]
            yield o, next_o, a, r, c, np.zeros_like(r), np.zeros_like(r)

    def sample(self, n_batches: int) -> Iterator[TrajectoryData]:
        if self.empty:
            return
        iterator = (
            TrajectoryData(
                *next(self._sample_batch(self.batch_size, self.sequence_length))
            )  # type: ignore
            for _ in range(n_batches)
        )
        if jax.default_backend() == "gpu":
            iterator = double_buffer(iterator)  # type: ignore
        yield from iterator

    @property
    def empty(self):
        return self._valid_episodes == 0
