"""Object removal using existing OpenCV inpainting / optional LaMa CPU backend."""
from dataclasses import dataclass

import cv2
import numpy as np

from demo.object_recoloring import RecolorSession, rgb_image
from demo.restoration_backends import run_model


def removal_mask(mask, margin=3):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError('Chọn vật thể cần xóa trước.')
    if not np.isfinite(margin) or not 0 <= margin <= 15 or int(margin) != margin:
        raise ValueError('Mở rộng vùng phải từ 0 đến 15 px.')
    margin = int(margin)
    if margin:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin*2+1, margin*2+1))
        mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    return mask


def remove_object(image, mask, backend='telea', radius=3, margin=3, runner=None):
    image = rgb_image(image)
    mask = removal_mask(mask, margin)
    if mask.shape != image.shape[:2]:
        raise ValueError('Mask phải cùng kích thước ảnh.')
    if mask.all():
        raise ValueError('Vùng xóa phủ toàn ảnh; cần giữ lại nền để vá.')
    if not np.isfinite(radius) or not 1 <= radius <= 9:
        raise ValueError('Bán kính vá phải từ 1 đến 9 px.')
    binary = mask.astype(np.uint8)*255
    if backend in ('telea', 'ns'):
        flag = cv2.INPAINT_TELEA if backend == 'telea' else cv2.INPAINT_NS
        prediction = cv2.inpaint(image, binary, float(radius), flag)
    elif backend == 'lama':
        prediction = (runner or run_model)('lama', image, mask=binary, max_side=512)
    else:
        raise ValueError('Phương pháp xóa không hợp lệ.')
    if prediction.shape != image.shape or prediction.dtype != np.uint8:
        raise ValueError('Backend trả về ảnh không hợp lệ.')
    result = image.copy()
    result[mask] = prediction[mask]
    return result


@dataclass
class RemovalSession(RecolorSession):
    margin: int = 3

    def set_margin(self, margin):
        if not np.isfinite(margin) or int(margin) != margin or not 0 <= margin <= 15:
            raise ValueError('Mở rộng vùng phải từ 0 đến 15 px.')
        if self.margin != int(margin):
            self.margin = int(margin)
            self.confirmed = False

    def overlay(self):
        # Preview exactly the expanded region that the removal backend will modify.
        from copy import copy
        preview = copy(self)
        if self.mask is not None:
            preview.mask = removal_mask(self.mask, self.margin)
        return RecolorSession.overlay(preview)

    def remove(self, backend='telea', radius=3, margin=3):
        if not self.confirmed or int(margin) != self.margin:
            raise ValueError('Hãy xác nhận vùng xóa sau khi chỉnh vùng hoặc mức mở rộng.')
        result = remove_object(self.current, self.mask, backend, radius, margin)
        if not np.array_equal(result, self.current):
            self.history.append(self.current.copy())
            while len(self.history) > 1 and sum(a.nbytes for a in self.history) > 128*1024**2:
                self.history.pop(0)
        self.current = result
        self.clear()
