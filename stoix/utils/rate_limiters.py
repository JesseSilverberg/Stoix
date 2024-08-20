import logging
import sys
import threading
from typing import Optional, Tuple, Union

# Taken and modified
# from https://github.com/google-deepmind/reverb/blob/master/reverb/rate_limiters.py


class RateLimiter:
    def __init__(
        self, samples_per_insert: float, min_size_to_sample: int, min_diff: float, max_diff: float
    ):
        assert min_size_to_sample > 0, "min_size_to_sample must be greater than 0"
        assert samples_per_insert > 0, "samples_per_insert must be greater than 0"

        self.samples_per_insert = samples_per_insert
        self.min_diff = min_diff
        self.max_diff = max_diff
        self.min_size_to_sample = min_size_to_sample

        self.inserts = 0
        self.samples = 0
        self.deletes = 0

        self.mutex = threading.Lock()
        self.condition = threading.Condition(self.mutex)

    def num_inserts(self) -> int:
        """Returns the number of inserts."""
        with self.mutex:
            return self.inserts

    def num_samples(self) -> int:
        """Returns the number of samples."""
        with self.mutex:
            return self.samples

    def num_deletes(self) -> int:
        """Returns the number of deletes."""
        with self.mutex:
            return self.deletes

    def insert(self) -> None:
        """Increment the number of inserts and notify all waiting threads."""
        with self.mutex:
            self.inserts += 1
            self.condition.notify_all()  # Notify all waiting threads

    def delete(self) -> None:
        """Increment the number of deletes and notify all waiting threads."""
        with self.mutex:
            self.deletes += 1
            self.condition.notify_all()  # Notify all waiting threads

    def sample(self) -> None:
        """Increment the number of samples and notify all waiting threads."""
        with self.mutex:
            self.samples += 1
            self.condition.notify_all()  # Notify all waiting threads

    def can_insert(self, num_inserts: int) -> bool:
        """Check if the caller can insert `num_inserts` items."""
        # Assume lock is already held by the caller
        if num_inserts <= 0:
            return False
        if self.inserts + num_inserts - self.deletes <= self.min_size_to_sample:
            return True
        diff = (num_inserts + self.inserts) * self.samples_per_insert - self.samples
        return diff <= self.max_diff

    def can_sample(self, num_samples: int) -> bool:
        """Check if the caller can sample `num_samples` items."""
        # Assume lock is already held by the caller
        if num_samples <= 0:
            return False
        if self.inserts - self.deletes < self.min_size_to_sample:
            return False
        diff = self.inserts * self.samples_per_insert - self.samples - num_samples
        return diff >= self.min_diff

    def await_can_insert(self, num_inserts: int = 1, timeout: Optional[float] = None) -> bool:
        """Wait until the caller can insert `num_inserts` items."""
        with self.condition:
            result = self.condition.wait_for(lambda: self.can_insert(num_inserts), timeout)
            if not result:
                raise TimeoutError(f"Timeout occurred while waiting to insert {num_inserts} items.")
            return result

    def await_can_sample(self, num_samples: int = 1, timeout: Optional[float] = None) -> bool:
        """Wait until the caller can sample `num_samples` items."""
        with self.condition:
            result = self.condition.wait_for(lambda: self.can_sample(num_samples), timeout)
            if not result:
                raise TimeoutError(f"Timeout occurred while waiting to sample {num_samples} items.")
            return result

    def __repr__(self) -> str:
        return (
            f"RateLimiter(samples_per_insert={self.samples_per_insert}, "
            f"min_size_to_sample={self.min_size_to_sample}, "
            f"min_diff={self.min_diff}, max_diff={self.max_diff})"
        )


class MinSize(RateLimiter):
    """Block sample calls unless replay contains `min_size_to_sample`.

    This limiter blocks all sample calls when the replay contains less than
    `min_size_to_sample` items, and accepts all sample calls otherwise.
    """

    def __init__(self, min_size_to_sample: int):
        if min_size_to_sample < 1:
            raise ValueError(
                f"min_size_to_sample ({min_size_to_sample}) must be a positive integer"
            )

        super().__init__(
            samples_per_insert=1.0,
            min_size_to_sample=min_size_to_sample,
            min_diff=-sys.float_info.max,
            max_diff=sys.float_info.max,
        )


class SampleToInsertRatio(RateLimiter):
    """Maintains a specified ratio between samples and inserts.

    The limiter works in two stages:

      Stage 1. Size of table is lt `min_size_to_sample`.
      Stage 2. Size of table is ge `min_size_to_sample`.

    During stage 1 the limiter works exactly like MinSize, i.e. it allows
    all insert calls and blocks all sample calls. Note that it is possible to
    transition into stage 1 from stage 2 when items are removed from the table.

    During stage 2 the limiter attempts to maintain the `samples_per_insert`
    ratio between the samples and inserts. This is done by
    measuring the `error`, calculated as:

      error = number_of_inserts * samples_per_insert - number_of_samples

    and making sure that `error` stays within `allowed_range`. Any operation
    which would move `error` outside of the `allowed_range` is blocked.
    Such approach allows for small deviation from a target `samples_per_insert`,
    which eliminates excessive blocking of insert/sample operations and improves
    performance.

    If `error_buffer` is a tuple of two numbers then `allowed_range` is defined as

      (error_buffer[0], error_buffer[1])

    When `error_buffer` is a single number then the range is defined as

      (
        min_size_to_sample * samples_per_insert - error_buffer,
        min_size_to_sample * samples_per_insert + error_buffer
      )
    """

    def __init__(
        self,
        samples_per_insert: float,
        min_size_to_sample: int,
        error_buffer: Union[float, Tuple[float, float]],
    ):
        """Constructor of SampleToInsertRatio.

        Args:
          samples_per_insert: The average number of times the learner should sample
            each item in the replay buffer during the item's entire lifetime.
          min_size_to_sample: The minimum number of items that the table must
            contain  before transitioning into stage 2.
          error_buffer: Maximum size of the "error" before calls should be blocked.
            When a single value is provided then inferred range is
              (
                min_size_to_sample * samples_per_insert - error_buffer,
                min_size_to_sample * samples_per_insert + error_buffer
              )
            The offset is added so that the error tracked is for the insert/sample
            ratio only takes into account operations occurring AFTER stage 1. If a
            range (two float tuple) then the values are used without any offset.

        Raises:
          ValueError: If error_buffer is smaller than max(1.0, samples_per_inserts).
        """
        if isinstance(error_buffer, float) or isinstance(error_buffer, int):
            offset = samples_per_insert * min_size_to_sample
            min_diff = offset - error_buffer
            max_diff = offset + error_buffer
        else:
            min_diff, max_diff = error_buffer

        if samples_per_insert <= 0:
            raise ValueError(f"samples_per_insert ({samples_per_insert}) must be > 0")

        if max_diff - min_diff < 2 * max(1.0, samples_per_insert):
            raise ValueError(
                "The size of error_buffer must be >= max(1.0, samples_per_insert) as "
                "smaller values could completely block samples and/or insert calls."
            )

        if max_diff < samples_per_insert * min_size_to_sample:
            logging.warning(
                "The range covered by error_buffer is below "
                "samples_per_insert * min_size_to_sample. If the sampler cannot "
                "sample concurrently, this will result in a deadlock as soon as "
                "min_size_to_sample items have been inserted."
            )
        if min_diff > samples_per_insert * min_size_to_sample:
            raise ValueError(
                "The range covered by error_buffer is above "
                "samples_per_insert * min_size_to_sample. This will result in a "
                "deadlock as soon as min_size_to_sample items have been inserted."
            )

        if min_size_to_sample < 1:
            raise ValueError(
                f"min_size_to_sample ({min_size_to_sample}) must be a positive integer"
            )

        super().__init__(
            samples_per_insert=samples_per_insert,
            min_size_to_sample=min_size_to_sample,
            min_diff=min_diff,
            max_diff=max_diff,
        )


if __name__ == "__main__":
    # Example usage:
    min_replay_size = 32
    samples_per_insert = 32
    samples_per_insert_tolerance_rate = 0.1
    samples_per_insert_tolerance = samples_per_insert_tolerance_rate * samples_per_insert
    error_buffer = min_replay_size * samples_per_insert_tolerance
    rate_lim = SampleToInsertRatio(
        samples_per_insert=samples_per_insert,
        min_size_to_sample=min_replay_size,
        error_buffer=error_buffer,
    )
    # rate_lim = MinSize(
    #     min_size_to_sample=min_replay_size
    # )

    num_inserts = 1000
    num_samples = num_inserts * samples_per_insert + samples_per_insert

    def insert_thread_fn(rate_limiter: RateLimiter) -> None:
        for i in range(num_inserts):
            rate_limiter.await_can_insert(1)
            rate_limiter.insert()
            print(f"Inserted {i+1}")

    def sample_thread_fn(rate_limiter: RateLimiter) -> None:
        for i in range(num_samples):
            rate_limiter.await_can_sample(1)
            rate_limiter.sample()
            print(f"Sampled {i+1}")

    insert_thread = threading.Thread(target=insert_thread_fn, args=(rate_lim,))
    sample_thread = threading.Thread(target=sample_thread_fn, args=(rate_lim,))

    insert_thread.start()
    sample_thread.start()

    sample_thread.join()
    insert_thread.join()
