import time


class FrameMeter:
    def __init__(self):
        self._frames = 0
        self._fps = 1
        self._start = None

    def start(self):
        if self._start is not None:
            raise Exception('Frame Counter already started')
        self._start = time.monotonic()

    def meter(self):
        current = time.monotonic()
        delta = current - self._start
        if delta >= 1:
            self._fps = self._frames / delta
            self._frames = 0
            self._start = current
        self._frames = self._frames + 1
        return self._fps

    def fps(self):
        return self._fps
