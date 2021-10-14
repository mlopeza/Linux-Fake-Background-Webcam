import signal
import time
from queue import Queue, Empty, Full
from threading import Thread, Event

import configargparse

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


def main(v_in, v_out, background, fps):
    mask_pipeline = build_mask_pipeline(selfie_model=0, threshold=0.05, erode_iterations=2, mask_update_speed=0.5)
    background_apply = BackgroundBlurReplace(background)
    pipeline = CameraPipeline(v_in, v_out, background_apply, mask_pipeline, 60)

    # Find a better way to handle the pipeline stop
    def handle(signum, frame):
        print('Handling termination, stopping pipeline')
        pipeline.stop()

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)

    fps.start()
    pipeline.start()

    stop = False
    while not stop:
        stop = pipeline.step()
    print('Stopping pipeline')
    pipeline.stop()


def build_mask_pipeline(selfie_model, threshold, erode_iterations, mask_update_speed):
    # Create the mask pipeline, in this case the pipeline would be build at the beginning and then we would apply
    # Everything at runtime
    return MaskPipeline() \
        .attach(SelfieSegmentation(model=selfie_model)) \
        .attach(BilateralFilter()) \
        .attach(Threshold(threshold=threshold)) \
        .attach(Sigmoid()) \
        .attach(GaussianBlur()) \
        .attach(Erode(erode_iterations)) \
        .attach(AccumulatedWeighted(mask_update_speed))


def _parse_arguments():
    parser = configargparse.ArgParser(description="Faking your webcam background under GNU/Linux.",
                                      formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-c", "--config", is_config_file=True,
                        help="Config file")

    # Webcam Options
    parser.add_argument("-w", "--webcam-path", default="/dev/video0",
                        help="Set real webcam path")
    parser.add_argument("-W", "--width", default=1280, type=int,
                        help="Set real webcam width")
    parser.add_argument("-H", "--height", default=720, type=int,
                        help="Set real webcam height")
    parser.add_argument("-F", "--fps", default=30, type=int,
                        help="Set real webcam FPS")
    parser.add_argument("-C", "--codec", default='YUYV', type=str,
                        help="Set real webcam codec")

    # V42Loopback options
    parser.add_argument("-v", "--v4l2loopback-path", default="/dev/video2",
                        help="V4l2loopback device path")

    # Output options
    parser.add_argument("--preview-output", action="store_true",
                        help="Show in a window the preview of the camera processing")
    # Background options
    parser.add_argument("-b", "--background", default=None,
                        help="Background image or video that should be used")

    parser.add_argument("--background-aspect", choices=['tile', 'keep-aspect'],
                        default='keep-aspect',
                        help="Select the aspect of the background")

    # Background mask processing skip, by default we apply all of them
    parser.add_argument("--skip-bilateral-filter", action="store_true",
                        help="Force the mask to follow a sigmoid distribution")
    parser.add_argument("--skip-bilateral-threshold", action="store_true",
                        help="Force the mask to follow a sigmoid distribution")
    parser.add_argument("--skip-sigmoid", action="store_true",
                        help="Force the mask to follow a sigmoid distribution")
    parser.add_argument("--skip-gaussian-blur", action="store_true",
                        help="Force the mask to follow a sigmoid distribution")
    parser.add_argument("--skip-bilateral-erosion", action="store_true",
                        help="Force the mask to follow a sigmoid distribution")
    parser.add_argument("--skip-background-average-mask", action="store_true",
                        help="Force the mask to follow a sigmoid distribution")

    # Options for background processors
    parser.add_argument("--select-model", default="landscape", choices=['general-purpose', 'landscape'],
                        help="Mediapipe model for background extractions")

    parser.add_argument("--background-mask-update-speed", default="50", type=int, choices=range(1, 100),
                        help="The running average percentage for background mask updates")

    parser.add_argument("--threshold", default="75", type=int, choices=range(1, 100),
                        help="The minimum percentage threshold for accepting a pixel as foreground")
    return parser.parse_args()


def _get_camera(path, width, height, codec, fps):
    return Camera(path, width, height, fps, codec)


def _get_output(fps, camera, preview):
    output = [V4L2LoopbackOutput('/dev/video2', camera.get_frame_width(), camera.get_frame_height())]
    if preview:
        output.append(CV2FrameOutput())
    return MultiOutput(fps, *output)


def _get_background(path, fps, width, height):
    # Todo: figure out a way to synchronize a possible video frame rate with the frame rate of the camera
    # possibly the fastest one will be the one that dictates the fps
    background = read_background(path, width, height)
    background_provider = lambda: background.next(fps.fps())
    return background_provider


if __name__ == "__main__":
    args = _parse_arguments()
    frame_counter = FrameMeter()

    # Todo: parametrize background
    video_background = '/home/mlopez/Downloads/Seemed - 3639.mp4'
    image_background = '/home/mlopez/Pictures/Backgrounds/fursona.jpg'
    red_background = '/home/mlopez/Desktop/red-background.jpg'
    nice_background = '/home/mlopez/Desktop/background.jpg'

    with _get_camera(args.webcam_path, args.width, args.height, args.codec, args.fps) as video_in:
        with _get_output(frame_counter, video_in, args.preview_output) as vide_out:
            background_provider = _get_background(video_background, frame_counter, video_in.get_frame_width(),
                                                  video_in.get_frame_height())
            main(video_in, vide_out, background_provider, frame_counter)