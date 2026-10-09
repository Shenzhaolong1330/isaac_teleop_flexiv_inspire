"""Single-use incremental point admission, shared by production and clock-controlled tests."""

STEP_NS = round(1e9 / 60)


class PointSchedule:
    def __init__(self, command, receive_ns):
        self.command = command
        self.origin_ns = receive_ns
        self.consumed = set()
        self.last_send_ns = None

    def admit(self, index, now_ns):
        if index in self.consumed:
            return "already-consumed"
        due = self.origin_ns + round(self.command.points[index].execute_after_s * 1e9)
        if now_ns < due:
            return "future"
        self.consumed.add(index)
        if now_ns >= self.command.expires_monotonic_ns or now_ns - due >= STEP_NS:
            return "expired"
        if self.last_send_ns is not None and now_ns - self.last_send_ns < STEP_NS // 2:
            return "expired-no-burst"
        self.last_send_ns = now_ns
        return "execute"
