import cv2
from pyfakewebcam import FakeWebcam


class VideoOutput:

    def __init__(self, frame_counter):
        self._frame_counter = frame_counter

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def _write(self, frame):
        raise RuntimeError("Write hasn't been implemented")

    def write(self, frame):
        self._write(frame)
        # Only add data to frame if the frame counter was received, otherwise we ignore it
        # This is only useful for the composite frame counter as we don't want to count the framerate twice
        # when writing
        if self._frame_counter:
            self._frame_counter.meter()


class V4L2LoopbackOutput(VideoOutput):

    def __init__(self, path, width, height, frame_counter=None):
        super().__init__(frame_counter)
        self._camera = FakeWebcam(path, width, height)

    def _write(self, frame):
        self._camera.schedule_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


class CV2FrameOutput(VideoOutput):

    def __init__(self, name='CV2Window - Output', frame_counter=None):
        super().__init__(frame_counter)
        self._window_name = name

    def _write(self, frame):
        cv2.imshow(self._window_name, frame)
        # This makes the imshow refresh and show the frame
        cv2.waitKey(1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        cv2.destroyWindow(self._window_name)


class MultiOutput(VideoOutput):
    def __init__(self, frame_counter, *outputs):
        super().__init__(frame_counter)
        self._outputs = outputs

    def _write(self, frame):
        for out in self._outputs:
            out.write(frame)
