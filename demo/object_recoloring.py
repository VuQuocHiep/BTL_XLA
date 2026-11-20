"""Pixel-exact masked recoloring and session-local editing state (RGB uint8)."""
from dataclasses import dataclass, field
import re

import cv2
import numpy as np


def rgb_image(image):
    if image is None:
        raise ValueError('Vui lòng tải ảnh hoặc dùng kết quả DDColor.')
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Ảnh phải là RGB uint8.')
    return np.ascontiguousarray(image).copy()


def image_point(index, image_shape, preview_shape=None):
    """Gradio SelectData is already in natural image pixels, not CSS pixels.

    preview_shape is only needed by clients sending coordinates on a resized raster.
    SAM itself performs its own encoder-coordinate transform.
    """
    x, y = map(float, index)
    h, w = image_shape[:2]
    ph, pw = (preview_shape or image_shape)[:2]
    if not (np.isfinite(x) and np.isfinite(y) and 0 <= x < pw and 0 <= y < ph):
        raise ValueError('Điểm chọn nằm ngoài ảnh.')
    return min(w - 1, int(x * w / pw)), min(h - 1, int(y * h / ph))


def feather_mask(mask, radius):
    mask = np.asarray(mask, dtype=bool)
    # Feather inward only: Gaussian blur must never spill into unselected pixels.
    alpha = mask.astype(np.float32)
    if radius > 0:
        distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        alpha = np.minimum(distance / float(radius), 1.0) * mask
    return alpha


def recolor(image, mask, color, strength=100, feather=3):
    image = rgb_image(image)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != image.shape[:2] or not mask.any():
        raise ValueError('Vùng chọn trống hoặc sai kích thước.')
    if not isinstance(color, str) or not re.fullmatch(r'#[0-9a-fA-F]{6}', color):
        raise ValueError('Chọn màu dạng #RRGGBB.')
    if not 0 <= float(strength) <= 100 or not 0 <= float(feather) <= 30:
        raise ValueError('Cường độ hoặc độ mềm cạnh không hợp lệ.')
    if float(strength) == 0:
        return image
    target = np.array([[[int(color[i:i+2], 16) for i in (1, 3, 5)]]], np.float32) / 255
    lab = cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB)[0, 0]
    changed = lab.copy()
    # Keep L exactly; keep a little local chroma variation for material texture.
    chroma = lab[:, :, 1:]
    changed[:, :, 1:] = target_lab[1:] + .2 * (chroma - chroma[mask].mean(axis=0))
    colored = np.clip(cv2.cvtColor(changed, cv2.COLOR_LAB2RGB) * 255, 0, 255)
    alpha = feather_mask(mask, feather)[:, :, None] * (float(strength) / 100)
    blended = np.rint(image * (1 - alpha) + colored * alpha).astype(np.uint8)
    image[mask] = blended[mask]  # Outside pixels are never assigned or round-tripped through LAB.
    return image


@dataclass
class RecolorSession:
    original: np.ndarray
    current: np.ndarray
    points: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    mask: object = None
    confirmed: bool = False
    applied: bool = False
    selection_base: object = None
    history: list = field(default_factory=list)

    @classmethod
    def load(cls, image):
        image = rgb_image(image)
        return cls(image, image.copy())

    def clear(self):
        self.points, self.labels = [], []
        self.mask = self.selection_base = None
        self.confirmed = self.applied = False

    def select(self, index, positive, segmenter):
        point = image_point(index, self.current.shape)
        previous_points = [] if self.applied else self.points
        previous_labels = [] if self.applied else self.labels
        if not previous_points and not positive:
            raise ValueError('Hãy chọn một điểm + trên vật thể trước.')
        points, labels = previous_points + [point], previous_labels + [int(positive)]
        base = self.current if self.applied or self.selection_base is None else self.selection_base
        mask = segmenter.predict(base, points, labels)
        if mask.shape != base.shape[:2] or not mask.any():
            raise ValueError('Không tìm thấy vùng phù hợp; hãy chọn lại điểm +.')
        self.points, self.labels, self.mask = points, labels, mask.astype(bool)
        self.selection_base = base.copy()
        self.confirmed = False
        self.applied = False

    def confirm(self):
        if self.mask is None or not self.mask.any():
            raise ValueError('Chọn vật thể trước khi xác nhận.')
        self.confirmed = True

    def apply(self, color, strength, feather):
        if not self.confirmed:
            raise ValueError('Hãy xác nhận vùng chọn trước khi đổi màu.')
        result = recolor(self.selection_base, self.mask, color, strength, feather)
        if not np.array_equal(result, self.current):
            self.history.append(self.current.copy())
            while len(self.history) > 1 and sum(a.nbytes for a in self.history) > 128 * 1024**2:
                self.history.pop(0)
        self.current = result
        self.applied = True

    def undo(self):
        if self.history:
            self.current = self.history.pop()
        self.clear()

    def reset(self):
        self.current = self.original.copy()
        self.history = []
        self.clear()

    def overlay(self):
        image = self.current.copy()
        if self.mask is not None:
            image[self.mask] = np.rint(.65 * image[self.mask] + .35 * np.array([0, 210, 255])).astype(np.uint8)
        for (x, y), positive in zip(self.points, self.labels):
            cv2.circle(image, (x, y), 5, (30, 240, 60) if positive else (255, 40, 40), -1)
            cv2.circle(image, (x, y), 6, (255, 255, 255), 1)
        return image
