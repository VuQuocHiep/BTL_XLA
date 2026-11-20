"""Repair detected scratches before denoising and enhancing a photograph."""

from dataclasses import dataclass, asdict
from time import perf_counter

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
    return mask


@dataclass(frozen=True)
class RestorationOptions:
    detector: str = 'auto'
    repair_backend: str = 'auto'
    detect: bool = True
    repair: bool = True
    denoise: bool = True
    contrast: bool = True
    sharpen: bool = False
    detection_threshold: float = 0.4
    mask_grow: int = 1
    denoise_h: float = 0.0  # zero selects noise-adaptive strength
    contrast_amount: float = 0.2
    sharpen_amount: float = 0.2
    max_model_side: int = 512

    def validate(self):
        if self.detector not in ('auto', 'microsoft', 'opencv') or self.repair_backend not in ('auto', 'lama', 'telea'):
            raise ValueError('Backend phục hồi không hợp lệ.')
        values = (self.detection_threshold, self.denoise_h, self.contrast_amount, self.sharpen_amount)
        if not all(np.isfinite(value) for value in values):
            raise ValueError('Thông số phục hồi phải là số hữu hạn.')
        if not (0.1 <= self.detection_threshold <= 0.9 and 0 <= self.mask_grow <= 5
                and 0 <= self.denoise_h <= 12 and 0 <= self.contrast_amount <= 0.5
                and 0 <= self.sharpen_amount <= 0.5 and 256 <= self.max_model_side <= 768):
            raise ValueError('Thông số phục hồi nằm ngoài phạm vi hỗ trợ.')
        if int(self.mask_grow) != self.mask_grow or int(self.max_model_side) != self.max_model_side:
            raise ValueError('Kích thước mask và giới hạn ảnh phải là số nguyên.')
        return self


@dataclass
class RestorationResult:
    image: np.ndarray
    mask: np.ndarray
    stages: dict
    timings: dict
    notes: list
    options: dict


def noise_sigma(gray):
    # Robust high-frequency estimator, excluding strong scene edges.
    image = gray.astype(np.float32)
    kernel = np.array([[1,-2,1],[-2,4,-2],[1,-2,1]], np.float32) / 6
    high = cv2.filter2D(image, -1, kernel)[2:-2,2:-2]
    gradient = cv2.magnitude(cv2.Sobel(image, -1, 1, 0), cv2.Sobel(image, -1, 0, 1))[2:-2,2:-2]
    flat = high[gradient <= np.percentile(gradient, 40)]
    return float(np.median(np.abs(flat - np.median(flat))) / 0.6745) if flat.size else 0.0


def restore_pipeline(image, options=None, manual_mask=None, runner=None):
    from demo.restoration_backends import available_models, run_model

    if image is None:
        raise ValueError('Vui lòng tải ảnh lên trước khi phục hồi.')
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 32:
        raise ValueError('Cần ảnh RGB 8 bit có mỗi chiều ít nhất 32 pixel.')
    options = (options or RestorationOptions()).validate()
    runner = runner or run_model
    availability = available_models()
    stages, timings, notes = {'Source': image.copy()}, {}, []
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    monochrome = np.max(np.ptp(image.astype(np.int16), axis=2)) <= 1
    started = perf_counter()
    step = perf_counter()
    if manual_mask is not None:
        manual_mask = np.asarray(manual_mask)
        if manual_mask.ndim == 3:
            manual_mask = cv2.cvtColor(manual_mask[:,:,:3], cv2.COLOR_RGB2GRAY)
        if manual_mask.shape != gray.shape or not np.all(np.isfinite(manual_mask)):
            raise ValueError('Mask thủ công phải cùng kích thước ảnh; màu trắng là vùng cần vá.')
        cutoff = 0.5 if manual_mask.dtype == bool else 128
        mask = np.uint8(manual_mask >= cutoff) * 255
        notes.append('Dùng mask thủ công thay cho dò tự động.')
    elif options.detect:
        use_ms = options.detector == 'microsoft' or (options.detector == 'auto' and availability['scratch'])
        if use_ms:
            probability = runner('scratch', image, max_side=options.max_model_side)
            mask = np.uint8(probability >= options.detection_threshold) * 255
            notes.append('Microsoft Scratch Detection (CPU).')
        else:
            mask = detect_damage(gray)
            notes.append('OpenCV detector; có thể bỏ sót vết rộng/mờ hoặc nhận nhầm đường nét.')
    else:
        mask = np.zeros_like(gray)
    if options.mask_grow and np.any(mask):
        size = options.mask_grow * 2 + 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size,size)))
    timings['Detection'] = perf_counter() - step
    overlay = image.copy()
    selected = mask > 0
    overlay[selected] = (image[selected].astype(np.float32) * 0.5 + np.array([255,40,40]) * 0.5).astype(np.uint8)
    stages['Damage overlay'] = overlay
    coverage = float(np.mean(selected))
    notes.append(f'Mask: {coverage * 100:.2f}% ảnh.')
    current = image.copy()
    step = perf_counter()
    if options.repair and np.any(selected):
        limit = 0.3 if manual_mask is not None else 0.12
        if coverage > limit:
            notes.append(f'Bỏ qua vá: mask vượt {limit*100:.0f}%; cần kiểm tra hoặc sửa mask.')
        else:
            use_lama = options.repair_backend == 'lama' or (options.repair_backend == 'auto' and availability['lama'])
            if use_lama:
                prediction = runner('lama', image, mask=mask, max_side=options.max_model_side)
                if monochrome:
                    prediction = cv2.cvtColor(cv2.cvtColor(prediction, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
                current[selected] = prediction[selected]
                notes.append('LaMa CPU; chỉ thay pixel nằm trong mask.')
            else:
                current = cv2.inpaint(current, mask, 3, cv2.INPAINT_TELEA)
                notes.append('Telea; chỉ phù hợp hư hỏng hẹp.')
    elif options.repair and options.detect and not np.any(selected):
        notes.append('Không phát hiện vùng hỏng; chưa vá pixel nào.')
    timings['Repair'] = perf_counter() - step
    stages['Repaired'] = current.copy()

    def luminance(rgb):
        if monochrome:
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        return lab[:,:,0].copy(), lab

    def merge(light, lab):
        if lab is None:
            return cv2.cvtColor(light, cv2.COLOR_GRAY2RGB)
        lab[:,:,0] = light
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    step = perf_counter()
    if options.denoise:
        light, lab = luminance(current)
        estimated = noise_sigma(light)
        h = options.denoise_h if options.denoise_h else min(8.0, estimated * 0.8)
        if h >= 1.0:
            light = cv2.fastNlMeansDenoising(light, None, h=float(h), templateWindowSize=7, searchWindowSize=21)
            current = merge(light, lab)
        notes.append(f'Noise sigma={estimated:.2f}; NLM h={h:.2f}.')
    timings['Denoise'] = perf_counter() - step
    stages['Denoised'] = current.copy()
    step = perf_counter()
    if options.contrast and options.contrast_amount:
        light, lab = luminance(current)
        lo, hi = np.percentile(light, (1,99))
        # Do not stretch already high-contrast photos, or near-constant patches.
        if 16 <= hi - lo < 170:
            stretched = np.clip((light.astype(np.float32)-lo)*255/(hi-lo),0,255).astype(np.uint8)
            amount = options.contrast_amount
            current = merge(cv2.addWeighted(light,1-amount,stretched,amount,0),lab)
        else:
            notes.append('Giữ tương phản: dải sáng hiện tại không cần kéo giãn.')
    timings['Contrast'] = perf_counter() - step
    stages['Contrast'] = current.copy()
    step = perf_counter()
    if options.sharpen and options.sharpen_amount:
        light, lab = luminance(current)
        detail = light.astype(np.float32) - cv2.GaussianBlur(light,(0,0),1.0).astype(np.float32)
        floor = max(2.0, 2 * noise_sigma(light))
        detail[np.abs(detail) <= floor] = 0
        sharpened = np.clip(light.astype(np.float32)+options.sharpen_amount*detail,0,255).round().astype(np.uint8)
        current = merge(sharpened,lab)
    timings['Sharpen'] = perf_counter() - step
    stages['Sharpened'] = current.copy()
    timings['Total'] = perf_counter() - started
    return RestorationResult(current,mask,stages,timings,notes,asdict(options))


def restore_photo(image, strength=1.0):
    # Compatibility for scripts; explicit backends make this wrapper deterministic.
    if not np.isfinite(strength) or not 0.5 <= strength <= 1.5:
        raise ValueError('Mức phục hồi phải nằm trong khoảng 0,5 đến 1,5.')
    options = RestorationOptions(detector='opencv', repair_backend='telea', denoise_h=4*strength,
                                 contrast_amount=0.2*strength, sharpen_amount=0.2*strength)
    result = restore_pipeline(image, options)
    return result.image, result.mask
