"""Small dependency-free progress display for terminals and redirected logs."""
import sys
import time


class Progress:
    def __init__(self, total, description, every=10, enabled=True, stream=None):
        if total < 1 or every < 1:
            raise ValueError('progress total/every must be positive')
        self.total, self.description, self.every = total, description, every
        self.enabled, self.stream = enabled, stream or sys.stderr
        self.tty = self.stream.isatty()
        self.start = time.monotonic()
        self.count, self.last_time, self.last_printed = 0, 0., 0
        self.fields = {}

    def __enter__(self):
        return self

    def update(self, **fields):
        self.count += 1
        self.fields = fields
        now = time.monotonic()
        if (self.count in (1,self.total) or
                (self.tty and now-self.last_time >= .25) or
                (not self.tty and self.count % self.every == 0)):
            self.render(now)

    def render(self, now):
        if not self.enabled:
            return
        elapsed = max(now-self.start, 1e-6)
        speed = self.count/elapsed
        eta = max(self.total-self.count,0)/speed if speed else 0
        filled = min(20, int(20*self.count/self.total))
        bar = '#' * filled + '-' * (20-filled)
        values = ' '.join(f'{k}={v}' for k,v in self.fields.items())
        text = (f'{self.description} [{bar}] {self.count}/{self.total} '
                f'{100*self.count/self.total:5.1f}% {speed:.2f} batch/s '
                f'elapsed={elapsed:.0f}s ETA={eta:.0f}s | {values}')
        end = '\n' if not self.tty or self.count == self.total else ''
        self.stream.write(('\r\033[2K' if self.tty else '') + text + end)
        self.stream.flush()
        self.last_time, self.last_printed = now, self.count

    def __exit__(self, *exception):
        if self.enabled and self.count != self.last_printed:
            self.render(time.monotonic())
        if self.enabled and self.tty and self.count != self.total:
            self.stream.write('\n')
