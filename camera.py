import time
from queue import Queue, Empty, Full
from threading import Thread

import cv2
import numpy as np
from pyfakewebcam import FakeWebcam

from camera.producers.background import read_background
from camera.producers.camera import Camera
from pipeline.transformers import *


# Todo: make sure that the fps match between the writer and video source (webcam or video background)
# Todo: create class that will manage the interruption of the data that will be executed or something on the same line
class FrameCounter:
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


def cam_reader(camera, frames, frame_counter):
    while True:
        try:
            frames.put_nowait(camera.next())
        except Full:
            print(f"Dropping camera frames as queue is full")
            pass


def cam_writer(fake_cam, frames, frame_counter):
    while True:
        fake_cam.schedule_frame(frames.get())
        frame_counter.meter()


def frame_printer(frame_counter, writer, reader):
    while True:
        time.sleep(1)
        print("FPS: {:6.2f} writer({}) reader({})".format(frame_counter.fps(), writer.qsize(), reader.qsize()))


def extract_background_v2(algo, frame, old_mask, threshold, mask_update_speed):
    BG_COLOR = (192, 192, 192)  # gray
    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = algo.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw selfie segmentation on the background image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack(
        (results.segmentation_mask,) * 3, axis=-1) > 0.1
    # The background can be customized.
    #   a) Load an image (with the same width and height of the input image) to
    #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
    #   b) Blur the input image by applying image filtering, e.g.,
    #      bg_image = cv2.GaussianBlur(image,(55,55),0)
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    output_image = np.where(condition, image, bg_image)
    return output_image


# todo: make sure to match fps between video file and camera as it could be inconsistent after initialization

def main():
    video_background = '/home/mlopez/Downloads/Seemed - 3639.mp4'
    image_background = '/home/mlopez/Desktop/background.jpg'
    target_fps = 30
    width = 640
    height = 360
    background = read_background(image_background, width, height)
    print(f'Target fps {target_fps}')
    camera = Camera('/dev/video0', width, height, target_fps, 'YUYV')
    fake_cam = FakeWebcam('/dev/video2', width, height)
    # Queue should honestly drop oldest elements if full
    source_pipe = Queue(maxsize=5)
    destination_pipe = Queue(maxsize=5)
    frame_counter = FrameCounter()
    frame_counter.start()
    writer = Thread(target=cam_reader, args=(camera, source_pipe, frame_counter))
    reader = Thread(target=cam_writer, args=(fake_cam, destination_pipe, frame_counter))
    printer = Thread(target=frame_printer, args=(frame_counter, destination_pipe, source_pipe))
    reader.start()
    writer.start()
    printer.start()
    old_mask = None
    frame = source_pipe.get(block=True)
    frame_timeout = 1 / target_fps
    print(f'{target_fps} {frame_timeout}')
    mask_pipeline = build_mask_pipeline(0.75, 1, 0.5)

    # The background provider is a lambda of the current process
    provider = lambda: background.next(frame_counter.fps())
    linear_background_apply = LinearBlendBackgroundReplace(provider)
    background_apply = BackgroundReplace(provider)
    while True:
        try:
            frame = source_pipe.get(timeout=frame_timeout)
        except Empty:
            # Use the already existing frame to write it
            pass
        """
        image = extract_background_v2(algo, frame, old_mask, 0.5, 1)
        destination_pipe.put(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        """
        mask = mask_pipeline.process(frame)
        result = cv2.cvtColor(background_apply.process(frame, mask), cv2.COLOR_BGR2RGB)
        try:
            destination_pipe.put(result)
        except Full:
            print('Dropped write frame')
            pass


def build_mask_pipeline(threshold, dilate_iterations, mask_update_speed):
    # Create the mask pipeline, in this case the pipeline would be build at the beginning and then we would apply
    # Everything at runtime
    return MaskPipeline() \
        .attach(SelfieSegmentation()) \
        .attach(BilateralFilter()) \
        .attach(Threshold(threshold=0.05)) \
        .attach(Dilate(iterations=1)) \
        .attach(Sigmoid()) \
        .attach(AccumulatedWeighted(0.5))


if __name__ == "__main__":
    main()
