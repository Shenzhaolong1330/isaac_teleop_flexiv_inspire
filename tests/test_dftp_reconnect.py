from flexiv_inspire_isaac.dftp.models import HandCommand
from flexiv_inspire_isaac.dftp.worker import DftpHandWorker


class ReconnectingReader:
    def __init__(self):
        self.connect_calls = 0
        self.close_calls = 0
        self.worker = None

    def connect(self):
        self.connect_calls += 1
        if self.connect_calls == 1:
            raise TimeoutError("transient connect timeout")
        # The second connection is enough to prove bounded retry without
        # entering an unbounded synthetic sampling loop.
        self.worker._stop.set()

    def close(self):
        self.close_calls += 1

    def read_state(self):
        raise AssertionError("sampling should not run in this focused test")

    def read_surface(self, _spec):
        raise AssertionError("sampling should not run in this focused test")


class RejectWrites:
    def __init__(self):
        self.commands = []

    def apply(self, command):
        self.commands.append(command)


def test_transient_connect_timeout_retries_and_discards_queued_action():
    reader = ReconnectingReader()
    sink = RejectWrites()
    worker = DftpHandWorker(
        "left",
        reader,
        command_sink=sink,
        sleeper=lambda _seconds: None,
    )
    reader.worker = worker
    assert worker.submit(
        HandCommand(
            sequence=9,
            angles=(500,) * 6,
            force_limits=(500,) * 6,
            deadline_ns=10**18,
            source="stale-during-disconnect",
        )
    )
    worker.run()
    assert reader.connect_calls == 2
    assert reader.close_calls == 2
    assert worker.mailbox.pending_sequence is None
    assert sink.commands == []
