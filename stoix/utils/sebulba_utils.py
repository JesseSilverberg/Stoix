import queue
import threading
import time
from functools import partial
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from colorama import Fore, Style
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
from jumanji.types import TimeStep

from stoix.base_types import Parameters, StoixTransition
from stoix.utils.rate_limiters import RateLimiter


# Copied from https://github.com/instadeepai/sebulba/blob/main/sebulba/core.py
class ThreadLifetime:
    """Simple class for a mutable boolean that can be used to signal a thread to stop."""

    def __init__(self) -> None:
        self._stop = False

    def should_stop(self) -> bool:
        return self._stop

    def stop(self) -> None:
        self._stop = True


class OnPolicyPipeline(threading.Thread):
    """
    The `OnPolicyPipeline` shards trajectories into `learner_devices`,
    ensuring trajectories are consumed in the right order to avoid being off-policy
    and limit the max number of samples in device memory at one time to avoid OOM issues.
    """

    def __init__(self, max_size: int, learner_devices: List[jax.Device], lifetime: ThreadLifetime):
        """
        Initializes the pipeline with a maximum size and the devices to shard trajectories across.

        Args:
            max_size: The maximum number of trajectories to keep in the pipeline.
            learner_devices: The devices to shard trajectories across.
        """
        super().__init__(name="OnPolicyPipeline")
        self.learner_devices = learner_devices
        self.tickets_queue: queue.Queue = queue.Queue()
        self._queue: queue.Queue = queue.Queue(maxsize=max_size)
        self.lifetime = lifetime

    def run(self) -> None:
        """This function ensures that trajectories on the queue are consumed in the right order. The
        start_condition and end_condition are used to ensure that only 1 thread is processing an
        item from the queue at one time, ensuring predictable memory usage.
        """
        while not self.lifetime.should_stop():
            try:
                start_condition, end_condition = self.tickets_queue.get(timeout=1)
                with end_condition:
                    with start_condition:
                        start_condition.notify()
                    end_condition.wait()
            except queue.Empty:
                continue

    def put(self, traj: Sequence[StoixTransition], timestep: TimeStep, timings_dict: Dict) -> None:
        """Put a trajectory on the queue to be consumed by the learner."""
        start_condition, end_condition = (threading.Condition(), threading.Condition())
        with start_condition:
            self.tickets_queue.put((start_condition, end_condition))
            start_condition.wait()  # wait to be allowed to start

        # [Transition(num_envs)] * rollout_len --> Transition[(rollout_len, num_envs,)
        traj = self.stack_trajectory(traj)
        # Split trajectory on the num envs axis so each learner device gets a valid full rollout
        sharded_traj = jax.tree.map(lambda x: self.shard_split_playload(x, axis=1), traj)

        # Timestep[(num_envs, ...), ...] -->
        # [(num_envs / num_learner_devices, ...)] * num_learner_devices
        sharded_timestep = jax.tree.map(self.shard_split_playload, timestep)

        # We block on the put to ensure that actors wait for the learners to catch up. This does two
        # things:
        # 1. It ensures that the actors don't get too far ahead of the learners, which could lead to
        # off-policy data.
        # 2. It ensures that the actors don't in a sense "waste" samples and their time by
        # generating samples that the learners can't consume.
        # However, we put a timeout of 180 seconds to avoid deadlocks in case the learner
        # is not consuming the data. This is a safety measure and should not be hit in normal
        # operation. We use a try-finally since the lock has to be released even if an exception
        # is raised.
        try:
            self._queue.put((sharded_traj, sharded_timestep, timings_dict), block=True, timeout=180)
        except queue.Full:
            print(
                f"{Fore.RED}{Style.BRIGHT}Pipeline is full and actor has timed out, "
                f"this should not happen. A deadlock might be occurring{Style.RESET_ALL}"
            )
        finally:
            with end_condition:
                end_condition.notify()  # tell we have finish

    def qsize(self) -> int:
        """Returns the number of trajectories in the pipeline."""
        return self._queue.qsize()

    def get(
        self, block: bool = True, timeout: Union[float, None] = None
    ) -> Tuple[StoixTransition, TimeStep, Dict]:
        """Get a trajectory from the pipeline."""
        return self._queue.get(block, timeout)  # type: ignore

    @partial(jax.jit, static_argnums=(0,))
    def stack_trajectory(self, trajectory: List[StoixTransition]) -> StoixTransition:
        """Stack a list of parallel_env transitions into a single
        transition of shape [rollout_len, num_envs, ...]."""
        return jax.tree_map(lambda *x: jnp.stack(x, axis=0), *trajectory)  # type: ignore

    def shard_split_playload(self, payload: Any, axis: int = 0) -> Any:
        split_payload = jnp.split(payload, len(self.learner_devices), axis=axis)
        return jax.device_put_sharded(split_payload, devices=self.learner_devices)

    def clear(self) -> None:
        """Clear the pipeline."""
        while not self._queue.empty():
            self._queue.get()


class OffPolicyPipeline(threading.Thread):
    """
    The `OffPolicyPipeline` shards sampled batches from a replay buffer into `learner_devices`,
    ensuring batches are consumed in-line with the replay buffer's sampling rate.
    """

    def __init__(
        self,
        replay_buffer_add: Callable[
            [TrajectoryBufferState, StoixTransition], TrajectoryBufferState
        ],
        replay_buffer_sample: Callable[[TrajectoryBufferState, chex.PRNGKey], StoixTransition],
        buffer_state: TrajectoryBufferState,
        rate_limiter: RateLimiter,
        rng_key: chex.PRNGKey,
        learner_devices: List[jax.Device],
        lifetime: ThreadLifetime,
    ):
        super().__init__(name="OffPolicyPipeline")
        self.replay_buffer_add = jax.jit(replay_buffer_add)
        self.replay_buffer_sample = jax.jit(replay_buffer_sample)
        self.split_key_fn = jax.jit(jax.random.split)
        self.buffer_state = buffer_state
        self.rate_limiter = rate_limiter
        self.rng_key = rng_key
        self.learner_devices = learner_devices
        self.tickets_queue: queue.Queue = queue.Queue()
        self.lifetime = lifetime

    def run(self) -> None:
        while not self.lifetime.should_stop():
            try:
                start_condition, end_condition = self.tickets_queue.get(timeout=1)
                with end_condition:
                    with start_condition:
                        start_condition.notify()
                    end_condition.wait()
            except queue.Empty:
                continue

    def put(self, traj: Sequence[StoixTransition], timings_dict: Dict) -> None:
        start_condition, end_condition = (threading.Condition(), threading.Condition())
        with start_condition:
            self.tickets_queue.put((start_condition, end_condition))
            start_condition.wait()  # wait to be allowed to start

        # [Transition(num_envs)] * rollout_len --> Transition[(rollout_len, num_envs,)
        traj = self.stack_trajectory(traj, 1)

        # wait until we can insert the data
        try:
            self.rate_limiter.await_can_insert(timeout=180)
        except TimeoutError:
            print(
                f"{Fore.RED}{Style.BRIGHT}Actor has timed out on insertion, "
                f"this should not happen. A deadlock might be occurring{Style.RESET_ALL}"
            )

        if self.buffer_state.is_full:
            self.rate_limiter.delete()

        # insert the data
        self.buffer_state = self.replay_buffer_add(self.buffer_state, traj)

        # signal that we have inserted the data
        self.rate_limiter.insert()

        with end_condition:
            end_condition.notify()  # tell we have finish

    def get(self, timeout: Union[float, None] = None) -> Tuple[StoixTransition, Dict]:
        """Get a trajectory from the buffer."""

        self.rng_key, key = self.split_key_fn(self.rng_key)

        # wait until we can sample the data
        try:
            self.rate_limiter.await_can_sample(timeout=timeout)
        except TimeoutError:
            print(
                f"{Fore.RED}{Style.BRIGHT}Learner has timed out on sampling, "
                f"this should not happen. A deadlock might be occurring{Style.RESET_ALL}"
            )

        # sample the data
        sampled_batch = self.replay_buffer_sample(self.buffer_state, key)

        # signal that we have sampled the data
        self.rate_limiter.sample()

        # split the trajectory over the learner devices
        sharded_sampled_batch = jax.tree.map(lambda x: self.shard_split_playload(x), sampled_batch)

        # TODO(edan): fix issue with timings_dict
        return sharded_sampled_batch, {}  # type: ignore

    @partial(jax.jit, static_argnums=(0, 2))
    def stack_trajectory(self, trajectory: List[StoixTransition], axis: int = 0) -> StoixTransition:
        """Stack a list of parallel_env transitions into a single
        transition of shape [rollout_len, num_envs, ...]."""
        return jax.tree.map(lambda *x: jnp.stack(x, axis=axis), *trajectory)  # type: ignore

    def shard_split_playload(self, payload: Any, axis: int = 0) -> Any:
        """Split the payload over the learner devices."""
        split_payload = jnp.split(payload, len(self.learner_devices), axis=axis)
        return jax.device_put_sharded(split_payload, devices=self.learner_devices)

    def clear(self) -> None:
        """Clear the buffer."""
        raise NotImplementedError("Clearing the buffer is not yet implemented.")


class ParamsSource(threading.Thread):
    """A `ParamSource` is a component that allows networks params to be passed from a
    `Learner` component to `Actor` components.
    """

    def __init__(self, init_value: Parameters, device: jax.Device, lifetime: ThreadLifetime):
        super().__init__(name=f"ParamsSource-{device.id}")
        self.value: Parameters = jax.device_put(init_value, device)
        self.device = device
        self.new_value: queue.Queue = queue.Queue()
        self.lifetime = lifetime

    def run(self) -> None:
        """This function is responsible for updating the value of the `ParamSource` when a new value
        is available.
        """
        while not self.lifetime.should_stop():
            try:
                waiting = self.new_value.get(block=True, timeout=1)
                self.value = jax.device_put(jax.block_until_ready(waiting), self.device)
            except queue.Empty:
                continue

    def update(self, new_params: Parameters) -> None:
        """Update the value of the `ParamSource` with a new value.

        Args:
            new_params: The new value to update the `ParamSource` with.
        """
        self.new_value.put(new_params)

    def get(self) -> Parameters:
        """Get the current value of the `ParamSource`."""
        return self.value


class RecordTimeTo:
    def __init__(self, to: Any):
        self.to = to

    def __enter__(self) -> None:
        self.start = time.monotonic()

    def __exit__(self, *args: Any) -> None:
        end = time.monotonic()
        self.to.append(end - self.start)


if __name__ == "__main__":
    pass
    # import flashbax as fbx

    # # Test off-policy pipeline
    # replay_buffer = fbx.make_trajectory_buffer(128, 2, 1, 1, 1, max_length_time_axis=10000)
    # fake_transition = {"obs": jnp.ones((4)), "act": jnp.ones((1,)), "rew": jnp.ones((1,))}
    # buffer_state = replay_buffer.init(fake_transition)

    # min_replay_size = 1
    # samples_per_insert = 2.0
    # samples_per_insert_tolerance_rate = 1.0
    # samples_per_insert_tolerance = samples_per_insert_tolerance_rate * samples_per_insert
    # error_buffer = min_replay_size * samples_per_insert_tolerance

    # # rate_limiter = MinSize(min_replay_size)
    # rate_limiter = SampleToInsertRatio(
    #     samples_per_insert=samples_per_insert,
    #     min_size_to_sample=min_replay_size,
    #     error_buffer=error_buffer,
    # )

    # lifetime = ThreadLifetime()
    # pipeline = OffPolicyPipeline(
    #     replay_buffer.add,
    #     replay_buffer.sample,
    #     buffer_state,
    #     rate_limiter,
    #     jax.random.PRNGKey(0),
    #     [jax.devices()[0]],
    #     lifetime,
    # )
    # pipeline.start()

    # # Make a thread that inserts data into the pipeline
    # def insert_data():
    #     for i in range(1000):
    #         batched_fake_transition = jax.tree.map(
    #             lambda *x: jnp.stack(x) + i, *([fake_transition] * 128)
    #         )
    #         rollout_of_batched_fake_transition = [batched_fake_transition] * 8
    #         pipeline.put(rollout_of_batched_fake_transition, {})
    #         print(f"Inserted batch {i}")

    # # Make a thread that samples data from the pipeline
    # def sample_data(thread_life: ThreadLifetime):
    #     i = 0
    #     while not thread_life.should_stop():
    #         batch = pipeline.get()
    #         print(f"Sampled batch {i}")
    #         i += 1

    # sample_lifetime = ThreadLifetime()
    # # Start the threads
    # insert_thread = threading.Thread(target=insert_data)
    # sample_thread = threading.Thread(target=sample_data, args=(sample_lifetime,))
    # insert_thread.start()
    # sample_thread.start()

    # # Wait for the threads to finish
    # insert_thread.join()

    # sample_lifetime.stop()

    # sample_thread.join()

    # lifetime.stop()
    # pipeline.join()
    # print("Off-policy pipeline test passed")
