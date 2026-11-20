import unittest

import cv2
import numpy as np

from demo.document_scanner import straighten_document


class DocumentScannerTests(unittest.TestCase):
    def photo(self, width=800, height=700, dark=False):
        paper = np.full((500, 360, 3), 35 if dark else 240, dtype=np.uint8)
        colors = [(220, 40, 40), (40, 220, 40), (40, 40, 220), (220, 180, 40)]
        for center, color in zip([(45, 45), (315, 45), (315, 455), (45, 455)], colors):
            cv2.circle(paper, center, 18, color, -1)
        source = np.float32([[0, 0], [359, 0], [359, 499], [0, 499]])
        corners = np.float32([[180, 80], [620, 145], [560, 625], [100, 560]])
        corners *= np.float32([width / 800, height / 700])
        matrix = cv2.getPerspectiveTransform(source, corners)
        photo = cv2.warpPerspective(paper, matrix, (width, height), borderValue=(180, 180, 180)
                                    if dark else (25, 25, 25))
        return photo, colors

    def test_perspective_crop_preserves_corner_order_and_rgb(self):
        photo, colors = self.photo()
        result = straighten_document(photo)
        self.assertTrue(440 < result.shape[1] < 480, result.shape)
        self.assertTrue(470 < result.shape[0] < 510, result.shape)
        for (x, y), expected in zip([(0.125, 0.09), (0.875, 0.09),
                                     (0.875, 0.91), (0.125, 0.91)], colors):
            actual = result[round(y * (result.shape[0] - 1)), round(x * (result.shape[1] - 1))]
            np.testing.assert_allclose(actual, expected, atol=25)

    def test_large_image_uses_original_resolution(self):
        photo, _ = self.photo(2400, 2100)
        result = straighten_document(photo)
        self.assertGreater(result.shape[0], 1400)
        self.assertGreater(result.shape[1], 1300)

    def test_dark_paper_on_light_background(self):
        photo, _ = self.photo(dark=True)
        result = straighten_document(photo)
        self.assertTrue(440 < result.shape[1] < 480, result.shape)

    def test_no_document_does_not_crop_background(self):
        for value in (0, 127, 255):
            with self.assertRaisesRegex(ValueError, 'bốn góc'):
                straighten_document(np.full((500, 500, 3), value, dtype=np.uint8))

    def test_missing_and_invalid_images(self):
        for image in (None, np.zeros((100, 100), dtype=np.uint8),
                      np.zeros((10, 10, 3), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                straighten_document(image)


if __name__ == '__main__':
    unittest.main()
