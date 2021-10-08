from itertools import cycle

import cv2
import numpy as np


class Background:

    def __init__(self, resource, width, height, fps):
        transformation = Background._resize(resource, width, height)
        self.fps = fps
        self._resource = iter(cycle(transformation))
        self._current_frame = transformation[0]

    def next(self, fps):
        # No FPS just means that the image is static and we can return the same one at any time
        if self.fps is None:
            return next(self._resource)
        # If by any chance the video is stopped we just return the current frame
        if fps == 0:
            return self._current_frame
        # if the image is not static we need to calculate which frame we should return
        # based on the current fps for the image we will need to see how many frames we should skip
        # or if we shouldn't skip any we just return at least 1
        rate = self.fps / fps
        # we randomly jump to the next frame based on an uniform distribution
        if rate < 1:
            if np.random.uniform() < rate:
                rate = 1
            else:
                rate = 0
        for _ in range(round(rate)):
            self._current_frame = next(self._resource)
        return self._current_frame

    @staticmethod
    def _resize(resource, width, height):
        transformation = []
        for frame in resource:
            if frame is None:
                continue
            transformation.append(Background._resize_frame(frame, width, height))
        return transformation

    @staticmethod
    def _resize_frame(img, width, height):
        # Keep aspect ratio by default
        imgheight, imgwidth, = img.shape[:2]
        scale = max(width / imgwidth, height / imgheight)
        newimgwidth, newimgheight = int(np.floor(width / scale)), int(
            np.floor(height / scale))
        ix0 = int(np.floor(0.5 * imgwidth - 0.5 * newimgwidth))
        iy0 = int(np.floor(0.5 * imgheight - 0.5 * newimgheight))
        return cv2.resize(img[iy0:iy0 + newimgheight, ix0:ix0 + newimgwidth, :], (width, height))


class ImageBackground(Background):

    def __init__(self, background, width, height):
        super().__init__([background], width, height, None)


class VideoBackground(Background):
    def __init__(self, video, width, height):
        print(f'Video fps {video.get(cv2.CAP_PROP_FPS)}')
        super().__init__(VideoBackground._all_frames(video), width, height, video.get(cv2.CAP_PROP_FPS))

    @staticmethod
    def _all_frames(resource):
        if not resource.isOpened():
            raise RuntimeError("Video received wasn't open")
        # Read all the frames of a video to memory to avoid wasting resources on re-reading the video
        frames = []
        ret, frame = resource.read()
        while ret:
            frames.append(frame)
            ret, frame = resource.read()
        return frames


class NoBackGround(Background):
    # No background, always
    def next(self, fps):
        return None


def read_background(path, width, height):
    resource = cv2.imread(path)
    if resource is not None:
        return ImageBackground(resource, width, height)
    resource = cv2.VideoCapture(path)
    if resource is not None:
        if not resource.isOpened():
            raise RuntimeError(f"Couldn't open video {path}")
        return VideoBackground(resource, width, height)
    raise RuntimeError(f"Failed to read resource {path}")
