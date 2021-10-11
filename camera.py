import signal
import time
from queue import Queue, Empty, Full
from threading import Thread, Event

from camera.input.background import read_background
from camera.input.camera import Camera
from camera.output.camera import V4L2LoopbackOutput, CV2FrameOutput, MultiOutput
from camera.statistics.meters import FrameMeter
from pipeline.transformers import *


# Todo: create class that will manage the interruption of the data that will be executed or something on the same line
# Todo: create an actual pipeline that starts everything in lockstep

class CameraPipeline:

    def __init__(self, video_in, video_out, background_replacer, mask_pipeline, target_fps,
                 queue_size=5):
        self._thread_kill = Event()
        self._input = Queue(queue_size)
        self._output = Queue(queue_size)
        self._framer_counter = frame_counter
        self._video_in = video_in
        self._video_out = video_out
        self._input_thread = Thread(target=CameraPipeline.reader, args=(video_in, self._input, self._thread_kill))
        self._output_thread = Thread(target=CameraPipeline.writer, args=(video_out, self._output, self._thread_kill))
        self._background_replacer = background_replacer
        self._mask_pipeline = mask_pipeline
        self._input_timeout = 1 / target_fps
        self._latest_frame = None

    def start(self):
        self._input_thread.start()
        self._output_thread.start()

    def step(self):
        if self._thread_kill.is_set():
            return self._thread_kill.is_set()
        try:
            self._latest_frame = self._input.get(timeout=self._input_timeout)
        except Empty:
            # If we can't get a frame we try to use the latest one, otherwise we just skip this execution step
            # it will be a slow startup but after the first frame everything will be smoother
            if self._latest_frame is None:
                return self._thread_kill.is_set()

        mask = self._mask_pipeline.process(self._latest_frame)
        result = self._background_replacer.process(self._latest_frame, mask)
        try:
            self._output.put(result)
        except Full:
            print('Dropped write frame')
            pass
        return self._thread_kill.is_set()

    def stop(self):
        print('Sending kill event to threads')
        self._thread_kill.set()
        self._output_thread.join()
        self._input_thread.join()
        print('Done with stop')

    @staticmethod
    def reader(v_in, queue, pill):
        while not pill.is_set():
            try:
                # try to put frames as fast as possible, doesn't matter if we are dropping them
                # as that would mean that the consumer is slow compared to the input
                queue.put_nowait(v_in.next())
            except Full:
                print(f"Dropping camera frames as queue is full")
                pass

    @staticmethod
    def writer(v_out, queue, pill):
        while not pill.is_set():
            try:
                # try to get processed images and output them, the timeout
                # is used mainly for the exit of the program, as at one point
                # the queue might be empty and we can't be blocked indefinitely or the
                # program will never end
                v_out.write(queue.get(timeout=0.1))
            except Empty:
                pass


def frame_printer(frame_counter, writer, reader):
    while True:
        time.sleep(1)
        print("FPS: {:6.2f} writer({}) reader({})".format(frame_counter.fps(), writer.qsize(), reader.qsize()))


def main(width, height, frame_counter, video_in, video_out, target_fps):
    video_background = '/home/mlopez/Downloads/Seemed - 3639.mp4'
    image_background = '/home/mlopez/Pictures/Backgrounds/fursona.jpg'
    red_background = '/home/mlopez/Desktop/red-background.jpg'
    mask_pipeline = build_mask_pipeline(0.75, 1, 0.5)

    background = read_background(video_background, width, height)

    # The background provider is a lambda of the current process
    background_provider = lambda: background.next(frame_counter.fps())
    background_apply = BackgroundBlurReplace(background_provider)

    pipeline = CameraPipeline(video_in, video_out, background_apply, mask_pipeline, target_fps)

    # Find a better way to handle the pipeline stop
    def handle(signum, frame):
        print('Handling termination, stopping pipeline')
        pipeline.stop()

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)

    frame_counter.start()
    pipeline.start()

    stop = False
    while not stop:
        stop = pipeline.step()
    print('Stopping pipeline')
    pipeline.stop()


def build_mask_pipeline(threshold, dilate_iterations, mask_update_speed):
    # Create the mask pipeline, in this case the pipeline would be build at the beginning and then we would apply
    # Everything at runtime
    return MaskPipeline() \
        .attach(SelfieSegmentation(model=0)) \
        .attach(BilateralFilter()) \
        .attach(Threshold(threshold=0.05)) \
        .attach(Sigmoid()) \
        .attach(GaussianBlur()) \
        .attach(AccumulatedWeighted(0.5))


if __name__ == "__main__":
    width = 640
    height = 360
    target_fps = 30
    webcam_codec = 'YUYV'
    frame_counter = FrameMeter()
    with Camera('/dev/video0', width, height, target_fps, webcam_codec) as video_in:
        with V4L2LoopbackOutput('/dev/video2', width, height) as v4l:
            with CV2FrameOutput() as cv2out:
                video_out = MultiOutput(frame_counter, cv2out, v4l)
                main(width, height, frame_counter, video_in, video_out, target_fps)
