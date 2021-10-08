import time
from queue import Queue, Empty
from threading import Thread

import cv2
import mediapipe as mp
import numpy as np
from pyfakewebcam import FakeWebcam

from camera.producers.background import read_background
from camera.producers.camera import Camera


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
        frames.put(camera.next())


def cam_writer(fake_cam, frames, frame_counter, target_fps):
    # target fps for the frames
    frame_timeout = 1 / target_fps
    # get the very first frame as we will use it as the baseline in case we start slow
    frame = frames.get()
    while True:
        fake_cam.schedule_frame(frame)
        frame_counter.meter()
        try:
            frame = frames.get(timeout=frame_timeout)
        except Empty:
            # We will just put the last frame instead of waiting for a new one from the queue
            pass


def frame_printer(frame_counter, writer, reader):
    while True:
        time.sleep(1)
        print("FPS: {:6.2f} writer({}) reader({})".format(frame_counter.fps(), writer.qsize(), reader.qsize()))


old_mask = None

BG_COLOR = (192, 192, 192)  # gray


def extract_background(algo, frame, old_mask, threshold, mask_update_speed):
    mask = algo.process(frame).segmentation_mask
    cv2.threshold(mask, threshold, 1, cv2.THRESH_BINARY, dst=mask)
    cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1, dst=mask)
    cv2.blur(mask, (10, 10), dst=mask)
    if old_mask is None:
        old_mask = mask
    mask = cv2.accumulateWeighted(mask, old_mask, mask_update_speed)
    return mask


# todo: make sure to match fps between video file and camera as it could be inconsistent after initialization

def main():
    video_background = '/home/mlopez/Downloads/Seemed - 3639.mp4'
    image_background = '/home/mlopez/Pictures/Backgrounds/fursona.jpg'
    target_fps = 30
    width = 640
    height = 360
    background = read_background(image_background, width, height)
    print(f'Target fps {target_fps}')
    camera = Camera('/dev/video0', width, height, target_fps, 'YUYV')
    fake_cam = FakeWebcam('/dev/video2', width, height)
    source_pipe = Queue()
    destination_pipe = Queue()
    frame_counter = FrameCounter()
    frame_counter.start()
    writer = Thread(target=cam_reader, args=(camera, source_pipe, frame_counter))
    reader = Thread(target=cam_writer, args=(fake_cam, destination_pipe, frame_counter, target_fps))
    printer = Thread(target=frame_printer, args=(frame_counter, destination_pipe, source_pipe))
    reader.start()
    writer.start()
    printer.start()
    old_mask = None
    # Selfie segmentation could cause a memory leak if we create it multiple times
    algo = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    while True:
        frame = source_pipe.get(block=True)
        mask = extract_background(algo, frame, old_mask, 0.5, 0.5)
        old_mask = mask
        cv2.blendLinear(frame, background.next(frame_counter.fps()), mask, 1 - mask, dst=frame)
        transformation = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        destination_pipe.put(transformation)


if __name__ == "__main__":
    main()
