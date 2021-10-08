import time
from queue import Queue, Empty, Full
from threading import Thread

from camera.input.background import read_background
from camera.input.camera import Camera
from camera.output.camera import V4L2LoopbackOutput, CV2FrameOutput, MultiOutput
from camera.statistics.meters import FrameMeter
from pipeline.transformers import *


# Todo: create class that will manage the interruption of the data that will be executed or something on the same line

def cam_reader(camera, frames):
    while True:
        try:
            frames.put_nowait(camera.next())
        except Full:
            print(f"Dropping camera frames as queue is full")
            pass


def cam_writer(output, frames):
    while True:
        output.write(frames.get())


def frame_printer(frame_counter, writer, reader):
    while True:
        time.sleep(1)
        print("FPS: {:6.2f} writer({}) reader({})".format(frame_counter.fps(), writer.qsize(), reader.qsize()))


def main(width, height, frame_counter, video_in, video_out):
    video_background = '/home/mlopez/Downloads/Seemed - 3639.mp4'
    image_background = '/home/mlopez/Desktop/background.jpg'
    target_fps = 30
    background = read_background(video_background, width, height)
    # Queue should honestly drop oldest elements if full
    source_pipe = Queue(maxsize=5)
    destination_pipe = Queue(maxsize=5)
    frame_counter.start()
    reader = Thread(target=cam_reader, args=(video_in, source_pipe))
    writer = Thread(target=cam_writer, args=(video_out, destination_pipe))
    printer = Thread(target=frame_printer, args=(frame_counter, destination_pipe, source_pipe))
    reader.start()
    writer.start()
    printer.start()

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
        result = background_apply.process(frame, mask)
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
    width = 640
    height = 360
    target_fps = 30
    webcam_codec = 'YUYV'
    frame_counter = FrameMeter()
    with Camera('/dev/video0', width, height, target_fps, webcam_codec) as video_in:
        with V4L2LoopbackOutput('/dev/video2', width, height) as v4l:
            with CV2FrameOutput() as cv2out:
                video_out = MultiOutput(frame_counter, cv2out, v4l)
                main(width, height, frame_counter, video_in, video_out)
