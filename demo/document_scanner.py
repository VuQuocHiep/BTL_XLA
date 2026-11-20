"""Detect a sheet of paper and rectify it without changing its colors."""

import cv2
import numpy as np


def straighten_document(image):
    """Return a perspective-corrected RGB image, or raise ValueError.

    Detection runs on a reduced image; the crop uses the original resolution.
    A document must have four visible corners and cover at least 10% of the photo.
    """
    if image is None:
        raise ValueError('Vui lòng tải ảnh lên trước khi làm thẳng ảnh.')
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Vui lòng chọn ảnh màu RGB hợp lệ.')
    height, width = image.shape[:2]
    if min(height, width) < 32:
        raise ValueError('Ảnh quá nhỏ. Vui lòng chọn ảnh có độ phân giải cao hơn.')

    scale = min(1.0, 1200.0 / max(height, width))
    small = cv2.resize(image, (round(width * scale), round(height * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    _, light = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    candidates = []
    sh, sw = gray.shape
    for mask in (edges, light, cv2.bitwise_not(light)):
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < sh * sw * 0.10:
                continue
            perimeter = cv2.arcLength(contour, True)
            for epsilon in (0.015, 0.025, 0.04):
                polygon = cv2.approxPolyDP(contour, epsilon * perimeter, True)
                if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                    continue
                points = polygon.reshape(4, 2).astype(np.float32)
                # Image-boundary contours are backgrounds, not visible paper edges.
                if (np.any(points[:, 0] <= 1) or np.any(points[:, 0] >= sw - 2)
                        or np.any(points[:, 1] <= 1) or np.any(points[:, 1] >= sh - 2)):
                    continue
                area = cv2.contourArea(points)
                if area >= sh * sw * 0.10:
                    candidates.append((area, points))
                break

    if not candidates:
        raise ValueError('Không tìm thấy đủ bốn góc giấy. Hãy chụp rõ toàn bộ tờ giấy trên nền tương phản.')

    points = max(candidates, key=lambda candidate: candidate[0])[1]
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    points = np.roll(points, -np.argmin(points.sum(axis=1)), axis=0)
    points *= np.array([width / sw, height / sh], dtype=np.float32)
    tl, tr, br, bl = points
    out_width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    out_height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    if min(out_width, out_height) < 32:
        raise ValueError('Không xác định được khổ giấy. Vui lòng chụp lại ảnh rõ hơn.')
    destination = np.array([[0, 0], [out_width - 1, 0],
                            [out_width - 1, out_height - 1], [0, out_height - 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(points, destination)
    return cv2.warpPerspective(image, transform, (out_width, out_height), flags=cv2.INTER_CUBIC)
