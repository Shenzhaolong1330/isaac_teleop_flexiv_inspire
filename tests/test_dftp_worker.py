import struct

from flexiv_inspire_isaac.dftp.models import (
    Acquisition,
    HandCommand,
    HandState,
    TactileSurface,
)
from flexiv_inspire_isaac.dftp.protocol import TACTILE_LAYOUT
from flexiv_inspire_isaac.dftp.worker import DftpHandWorker, LatestOnlyMailbox


def command(sequence, deadline=1_000_000):
    return HandCommand(
        sequence=sequence,
        angles=(sequence,) * 6,
        force_limits=(500,) * 6,
        deadline_ns=deadline,
        source="test",
    )


def state(angle=321):
    return HandState(
        side="left",
        actuator_position=(0,) * 6,
        actuator_angle=(angle,) * 6,
        actual_force_g=(0,) * 6,
        current_ma=(0,) * 6,
        temperature_c=(20,) * 6,
        error_code=(0,) * 6,
        status_code=(0,) * 6,
        acquisition=Acquisition(1, 2, 1),
        field_times_ns={},
    )


class Sink:
    def __init__(self, events):
        self.events = events
        self.commands = []

    def apply(self, value):
        self.events.append(("command", value.sequence, value.source))
        self.commands.append(value)


class Reader:
    def __init__(self, events):
        self.events = events
        self.worker = None
        self.index = 0

    def connect(self):
        pass

    def close(self):
        pass

    def read_state(self):
        return state()

    def read_surface(self, spec):
        self.events.append(("surface", self.index))
        if self.index == 2:
            assert self.worker.submit(command(7))
        self.index += 1
        return TactileSurface(
            name=spec.name,
            rows=spec.rows,
            cols=spec.cols,
            values=(0,) * spec.taxels,
            acquisition_start_ns=self.index,
            acquisition_end_ns=self.index + 1,
        )


def test_latest_only_mailbox_drops_old_sequences():
    mailbox = LatestOnlyMailbox()
    assert mailbox.put(command(1))
    assert mailbox.put(command(3))
    assert not mailbox.put(command(2))
    assert mailbox.take().sequence == 3
    assert not mailbox.put(command(3))


def test_action_preempts_between_tactile_surfaces():
    events = []
    reader = Reader(events)
    sink = Sink(events)
    worker = DftpHandWorker(
        "left",
        reader,
        command_sink=sink,
        clock_ns=lambda: 10,
        on_tactile=lambda _frame: events.append(("frame",)),
    )
    reader.worker = worker
    worker._latest_state = state()
    worker._read_tactile_frame_safely()
    command_index = events.index(("command", 7, "test"))
    fourth_surface_index = events.index(("surface", 3))
    assert command_index < fourth_surface_index
    assert events[-1] == ("frame",)


def test_timeout_holds_measured_angle_not_open_hand():
    events = []
    sink = Sink(events)
    reader = Reader(events)
    worker = DftpHandWorker(
        "left", reader, command_sink=sink, clock_ns=lambda: 200
    )
    worker._latest_state = state(angle=437)
    worker._last_applied = command(10, deadline=100)
    worker._service_command(200)
    hold = sink.commands[-1]
    assert hold.angles == (437,) * 6
    assert hold.angles != (1000,) * 6
    assert hold.force_limits == worker.safe_force_limits
    assert hold.source.startswith("watchdog-hold")


def test_read_only_worker_rejects_command_submission():
    worker = DftpHandWorker("left", Reader([]), command_sink=None)
    assert worker.read_only
    assert not worker.submit(command(1))


def test_tactile_frame_and_surfaces_share_observation_clock():
    class ClockedReader(Reader):
        def __init__(self):
            super().__init__([])
            self._time = 1_700_000_000_000_000_000

        def clock_ns(self):
            self._time += 1
            return self._time

        def read_surface(self, spec):
            start = self.clock_ns()
            end = self.clock_ns()
            return TactileSurface(
                name=spec.name,
                rows=spec.rows,
                cols=spec.cols,
                values=(0,) * spec.taxels,
                acquisition_start_ns=start,
                acquisition_end_ns=end,
            )

    reader = ClockedReader()
    frames = []
    worker = DftpHandWorker(
        "left",
        reader,
        clock_ns=lambda: 123,
        on_tactile=frames.append,
    )
    worker._read_tactile_frame_safely()
    frame = frames[0]
    assert frame.acquisition_start_ns > 1_000_000_000_000_000_000
    assert frame.acquisition_end_ns > frame.acquisition_start_ns
    assert all(
        frame.acquisition_start_ns <= surface.acquisition_start_ns
        <= surface.acquisition_end_ns <= frame.acquisition_end_ns
        for surface in frame.surfaces
    )
