"""Online affine mapping from device time to host monotonic/system time."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ClockMapping:
    mapped_host_monotonic_ns: int
    mapped_host_unix_ns: int
    timing_valid: bool
    sample_count: int
    slope: float
    residual_rms_ns: float


class OnlineClockMapper:
    """Robust small-window affine clock estimator.

    Device timestamps may be epoch-based or boot-relative. Only their rate and
    monotonicity matter. The estimator resets on controller/daemon reconnect.
    """

    def __init__(
        self,
        *,
        window_size: int = 128,
        min_samples: int = 20,
        min_span_ns: int = 50_000_000,
        max_rate_error: float = 2.0e-3,
        max_residual_rms_ns: float = 5_000_000.0,
    ) -> None:
        self._pairs: deque[tuple[int, int, int]] = deque(maxlen=window_size)
        self._min_samples = min_samples
        self._min_span_ns = min_span_ns
        self._max_rate_error = max_rate_error
        self._max_residual_rms_ns = max_residual_rms_ns
        self._last_device_ns: int | None = None
        self._latest_host_receive_ns = 0

    def reset(self) -> None:
        self._pairs.clear()
        self._last_device_ns = None
        self._latest_host_receive_ns = 0

    def update(
        self,
        device_ns: int,
        host_monotonic_ns: int,
        host_unix_ns: int,
    ) -> ClockMapping:
        device_ns = int(device_ns)
        host_monotonic_ns = int(host_monotonic_ns)
        host_unix_ns = int(host_unix_ns)
        if min(device_ns, host_monotonic_ns, host_unix_ns) <= 0:
            self.reset()
            return ClockMapping(0, 0, False, 0, math.nan, math.inf)
        if self._last_device_ns is not None and device_ns < self._last_device_ns:
            self.reset()
        elif self._last_device_ns is not None and device_ns == self._last_device_ns:
            # RDK may repeat a controller timestamp across adjacent polls.
            # Do not duplicate-weight that sample or destroy a converged window.
            self._latest_host_receive_ns = max(
                self._latest_host_receive_ns, host_monotonic_ns
            )
            return self.map(device_ns)
        self._last_device_ns = device_ns
        self._latest_host_receive_ns = host_monotonic_ns
        self._pairs.append((device_ns, host_monotonic_ns, host_unix_ns))
        return self.map(device_ns)

    def map(self, device_ns: int) -> ClockMapping:
        count = len(self._pairs)
        if count < 2:
            return ClockMapping(0, 0, False, count, math.nan, math.inf)
        origin_device = self._pairs[0][0]
        origin_host = self._pairs[0][1]
        x = np.asarray(
            [device - origin_device for device, _, _ in self._pairs],
            dtype=np.float64,
        )
        y = np.asarray(
            [host - origin_host for _, host, _ in self._pairs],
            dtype=np.float64,
        )
        denominator = float(np.dot(x, x))
        if denominator <= 0.0:
            return ClockMapping(0, 0, False, count, math.nan, math.inf)
        slope = float(np.dot(x, y) / denominator)
        intercept = origin_host
        prediction = intercept + slope * x
        residual_rms = float(np.sqrt(np.mean(np.square(y - slope * x))))
        span = int(self._pairs[-1][0] - origin_device)
        valid = (
            count >= self._min_samples
            and span >= self._min_span_ns
            and abs(slope - 1.0) <= self._max_rate_error
            and residual_rms <= self._max_residual_rms_ns
        )
        mapped_mono = int(round(intercept + slope * (int(device_ns) - origin_device)))
        # UNIX-minus-monotonic offset is sampled together at the daemon. Median
        # rejects individual scheduling outliers without mixing clock domains.
        unix_minus_mono = int(np.median([
            unix - monotonic for _, monotonic, unix in self._pairs
        ]))
        mapped_unix = mapped_mono + unix_minus_mono
        # A mapped acquisition time in the future relative to the receive
        # envelope is invalid; callers must not hide this with age=max(0,...).
        valid = valid and mapped_mono <= self._latest_host_receive_ns
        return ClockMapping(
            mapped_host_monotonic_ns=mapped_mono if valid else 0,
            mapped_host_unix_ns=mapped_unix if valid else 0,
            timing_valid=valid,
            sample_count=count,
            slope=slope,
            residual_rms_ns=residual_rms,
        )
