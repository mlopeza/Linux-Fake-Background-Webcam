import cv2
import mediapipe as mp
import numpy as np


class MaskTransformer:

    def __init__(self, overwrite_source=True):
        self._next = None
        self._overwrite_source = overwrite_source

    def attach(self, transformer):
        self._next = transformer

    def apply(self, source):
        raise RuntimeError("Apply not implemented")

    def process(self, frame):
        current = frame if self._overwrite_source else frame.copy()
        transformation = self.apply(current)
        if self._next:
            return self._next.process(transformation)
        return transformation


class MaskPipeline(MaskTransformer):
    def __init__(self):
        self._head = None
        self._tail = None

    def attach(self, transformer):
        if self._head is None:
            self._head = transformer
            self._tail = transformer
        else:
            self._tail.attach(transformer)
            self._tail = transformer
        return self

    def process(self, frame):
        return self._head.process(frame)


class SelfieSegmentation(MaskTransformer):
    def __init__(self, model=1):
        super().__init__()
        self._algo = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=model)

    def apply(self, source):
        source.flags.writeable = False
        result = self._algo.process(cv2.cvtColor(source, cv2.COLOR_BGR2RGB)).segmentation_mask
        source.flags.writeable = True
        return result


class Threshold(MaskTransformer):

    def __init__(self, threshold=1):
        super().__init__()
        self._threshold = threshold
        self._type = cv2.THRESH_BINARY
        self._max_value = 1

    def apply(self, source):
        cv2.threshold(source, self._threshold, self._max_value, self._type, dst=source)
        return source


class Dilate(MaskTransformer):
    def __init__(self, iterations=1, overwrite_source=True):
        super().__init__(overwrite_source=overwrite_source)
        self._iterations = iterations
        self._kernel = np.ones((5, 5), np.uint8)

    def apply(self, source):
        cv2.dilate(source, self._kernel, iterations=self._iterations, dst=source)
        return source


class Blur(MaskTransformer):
    def __init__(self):
        super().__init__()
        self._kernel = (10, 10)

    def apply(self, source):
        cv2.blur(source, self._kernel, source)
        return source


class GaussianBlur(MaskTransformer):
    def __init__(self):
        super().__init__()
        self._kernel = (101, 101)

    def apply(self, source):
        cv2.GaussianBlur(source, self._kernel, cv2.BORDER_TRANSPARENT)
        return source


class BilateralFilter(MaskTransformer):
    def __init__(self):
        super().__init__()

    def apply(self, source):
        return cv2.bilateralFilter(source, 5, 200, 200)


class AccumulatedWeighted(MaskTransformer):
    def __init__(self, update_speed):
        super().__init__()
        self._update_speed = update_speed
        self._last_source = None

    def apply(self, source):
        if self._last_source is None:
            self._last_source = source
        return cv2.accumulateWeighted(source, self._last_source, self._update_speed)


class Sigmoid(MaskTransformer):
    def __init__(self, a=5., b=-10.):
        super().__init__()
        self._a = a
        self._b = b

    def apply(self, source):
        z = np.exp(self._a + self._b * source)
        sig = 1 / (1 + z)
        return sig


class LinearBlendBackgroundReplace:
    def __init__(self, background_provider, overwrite_source=True):
        super().__init__()
        self._provider = background_provider
        self._overwrite_source = overwrite_source

    def process(self, frame, mask):
        current = frame if self._overwrite_source else frame.copy()
        cv2.blendLinear(current, self._provider(), mask, 1 - mask, current)
        return current


class BackgroundReplace:
    def __init__(self, background_provider, threshold=0.1):
        super().__init__()
        self._provider = background_provider
        self._threshold = threshold

    def process(self, frame, mask):
        condition = np.stack((mask,) * 3, axis=-1) > self._threshold
        return np.where(condition, frame, self._provider())


class BackgroundBlurReplace:
    def __init__(self, background_provider, threshold=0.1):
        super().__init__()
        self._provider = background_provider
        self._threshold = threshold
        self._gaussian_blur = GaussianBlur()
        self._dilate = Dilate(2, overwrite_source=False)

    def process2(self, lena, mask):
        # mask = cv2.GaussianBlur(mask, (301, 301), 0)
        blur_mask = self._dilate.process(mask)
        blur_mask = cv2.blur(blur_mask, (500, 500))
        blur_mask = cv2.cvtColor(blur_mask, cv2.COLOR_GRAY2BGR)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        foreground = cv2.multiply(lena, blur_mask, dtype=cv2.CV_8U)
        background = cv2.multiply(self._provider(), (1 - mask), dtype=cv2.CV_8U)
        output = cv2.add(foreground, background)
        return output

    def process(self, image, pmask):
        pmask = cv2.erode(pmask, np.ones((5, 5), np.uint8), iterations=2)
        blur_mask = self._dilate.process(pmask)
        blur_mask = cv2.cvtColor(blur_mask, cv2.COLOR_GRAY2BGR)

        mask = cv2.cvtColor(pmask, cv2.COLOR_GRAY2BGR)

        foreground = cv2.multiply(image, blur_mask, dtype=cv2.CV_8U)
        background = cv2.multiply(self._provider(), (1 - mask), dtype=cv2.CV_8U)
        output = cv2.add(foreground, background)
        return output

    def _scale_uint(self, data):
        return cv2.normalize(src=data, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    def alphaBlend(self, img1, img2, mask):
        """ alphaBlend img1 and img 2 (of CV_8UC3) with mask (CV_8UC1 or CV_8UC3)
        """
        if mask.ndim == 3 and mask.shape[-1] == 3:
            alpha = mask / 255.0
        else:
            alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        blended = cv2.convertScaleAbs(img1 * (1 - alpha) + img2 * alpha)
        return blended
