"""Repair detected scratches before denoising and enhancing a photograph."""

import cv2
import numpy as np


def detect_damage(gray):
    """Find narrow defects, including broken scratches at arbitrary angles.

    Contrast alone is insufficient: line support or a small isolated dust
    component is required. No mask is inferred for large missing scene regions.
    """
    height, width = gray.shape
    local = cv2.medianBlur(gray, 7)
    residual = cv2.absdiff(gray, local)
    noise = float(np.median(residual)) / 0.6745
    threshold = min(65, max(25, 4 * noise))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bright = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    mask = np.zeros_like(gray)
    min_length = max(24, round(min(height, width) * 0.045))

    # Treat white and black scratches separately so unrelated opposite edges
    # cannot jointly support the same line.
    for response, cutoff, length in ((bright, threshold, min_length),
                                     (bright, max(16, threshold * 0.65), min_length * 2),
                                     (dark, max(35, threshold), min_length * 2)):
        candidates = np.uint8((response >= cutoff) & (residual >= cutoff * 0.6)) * 255
        lines = cv2.HoughLinesP(candidates, 1, np.pi / 360, threshold=max(15, length // 2),
                                minLineLength=length, maxLineGap=max(6, length // 4))
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4)[:2000]:
                count = max(abs(int(x2 - x1)), abs(int(y2 - y1))) + 1
                xs = np.rint(np.linspace(x1, x2, count)).astype(int)
                ys = np.rint(np.linspace(y1, y2, count)).astype(int)
                if np.mean(candidates[ys, xs] > 0) < 0.65:
                    continue
                path = np.zeros_like(gray)
                cv2.line(path, (int(x1), int(y1)), (int(x2), int(y2)), 255, 5)
                mask |= cv2.bitwise_and(path, candidates)

    # Dust must stand well above ordinary grain and occupy only a small area.
    dust = np.uint8(residual >= max(45, threshold * 1.5)) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dust)
    accepted = np.zeros(count, dtype=np.uint8)
    for label in range(1, count):
        _, _, w, h, area = stats[label]
        if 2 <= area <= 36 and max(w, h) <= 9:
            accepted[label] = 255
    mask |= accepted[labels]
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    if np.count_nonzero(mask) > gray.size * 0.12:
        raise ValueError('Vùng nghi hỏng quá rộng để vá tự động an toàn. Vui lòng chọn ảnh rõ hơn.')
    return mask


def restore_photo(image, strength=1.0):
    """Return restored RGB and the actual inpainting mask, at original size."""
    if image is None:
        raise ValueError('Vui lòng tải ảnh lên trước khi phục hồi.')
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Vui lòng chọn ảnh RGB 8 bit hợp lệ.')
    if min(image.shape[:2]) < 32:
        raise ValueError('Ảnh cần có chiều rộng và chiều cao ít nhất 32 pixel.')
    if not np.isfinite(strength) or not 0.5 <= strength <= 1.5:
        raise ValueError('Mức phục hồi phải nằm trong khoảng 0,5 đến 1,5.')

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = detect_damage(gray)
    repaired = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA) if np.any(mask) else image.copy()
    lab = cv2.cvtColor(repaired, cv2.COLOR_RGB2LAB)
    light, a, b = cv2.split(lab)
    # Denoise before sharpening, otherwise grain is amplified as detail.
    light = cv2.fastNlMeansDenoising(light, None, h=4 * strength,
                                    templateWindowSize=7, searchWindowSize=21)
    low, high = np.percentile(light, (1, 99))
    if high - low >= 16:
        stretched = np.clip((light.astype(np.float32) - low) * 255 / (high - low), 0, 255).astype(np.uint8)
        light = cv2.addWeighted(light, 1 - 0.4 * strength, stretched, 0.4 * strength, 0)
    blur = cv2.GaussianBlur(light, (0, 0), 1.0)
    light = cv2.addWeighted(light, 1 + 0.3 * strength, blur, -0.3 * strength, 0)
    restored = cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2RGB)
    return restored, mask
