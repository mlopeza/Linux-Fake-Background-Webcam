from threading import Thread


class ThreadWorker:
    def __init__(self, queue, worker):
        self._queue = queue
        self._thread = Thread(target=worker, args=(queue,))

    def start(self):
        self._thread.start()

    def join(self):
        self._thread.join()
