from typing import Iterator, Dict
import jax
import numpy as np

from actsafe.common.double_buffer import double_buffer
from actsafe.rl.trajectory import TrajectoryData


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
        self.max_length = max_length
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.num_rewards = num_rewards

        # Main storage arrays
        self.observation = np.zeros(
            (
                capacity,
                max_length,
            )
            + observation_shape,
            dtype=self.obs_dtype,
        )
        self.action = np.zeros(
            (capacity, max_length) + action_shape,
            dtype=self.dtype,
        )
        self.reward = np.zeros(
            (capacity, max_length, num_rewards),
            dtype=self.dtype,
        )
        self.cost = np.zeros(
            (capacity, max_length),
            dtype=self.dtype,
        )
        self.terminated = np.ones(
            (capacity, max_length),
            dtype=bool,
        )
        self.episode_lengths = np.zeros(capacity, dtype=np.int32)

        # Tracking ongoing episodes
        self.ongoing_episodes: Dict[int, Dict] = {}

        self._valid_episodes = 0
        self.rs = np.random.RandomState(seed)
        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def add_batch(self, trajectory: TrajectoryData):
        capacity, *_ = self.reward.shape
        batch_size = min(trajectory.observation.shape[0], capacity)
        # Discard data if batch size overflows capacity.
        end = min(self.episode_id + batch_size, capacity)
        episode_slice = slice(self.episode_id, end)
        if trajectory.reward.ndim == 2:
            trajectory = TrajectoryData(
                trajectory.observation,
                trajectory.next_observation,
                trajectory.action,
                trajectory.reward[..., None],
                trajectory.cost,
                trajectory.done,
                trajectory.terminal,
            )
        for data, val in zip(
            (self.action, self.reward, self.cost),
            (trajectory.action, trajectory.reward, trajectory.cost),
        ):
            data[episode_slice] = val[:batch_size].astype(self.dtype)
        self.observation[episode_slice] = trajectory.observation[:batch_size].astype(
            self.obs_dtype
        )
        self.episode_id = (self.episode_id + batch_size) % capacity
        self._valid_episodes = min(self._valid_episodes + batch_size, capacity)

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

        while True:
            episode_ids = self.rs.choice(valid_episodes, size=batch_size)
            low = np.array(
                [
                    self.rs.randint(
                        0, max(1, self.episode_lengths[episode_id] - sequence_length)
                    )
                    for episode_id in episode_ids
                ]
            )
            timestep_ids = low[:, None] + np.tile(
                np.arange(sequence_length + 1),
                (batch_size, 1),
            )
            for i, (episode_id, time_steps) in enumerate(
                zip(episode_ids, timestep_ids)
            ):
                episode_length = self.episode_lengths[episode_id]
                if time_steps[-1] >= episode_length:
                    # Adjust timesteps to end at episode termination
                    offset = time_steps[-1] - episode_length + 1
                    timestep_ids[i] -= offset

            a, r, c = [
                x[episode_ids[:, None], timestep_ids[:, :-1]]
                for x in (self.action, self.reward, self.cost)
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
            )
            for _ in range(n_batches)
        )
        if jax.default_backend() == "gpu":
            iterator = double_buffer(iterator)  # type: ignore
        yield from iterator

    @property
    def empty(self):
        return self._valid_episodes == 0
