from pathlib import Path
import unittest

import cv2
import numpy as np

from demo.photo_restoration import detect_damage, restore_photo


class RestorationTests(unittest.TestCase):
    def test_broken_and_slanted_scratches_are_detected_and_repaired(self):
        image = np.full((320, 320, 3), 125, dtype=np.uint8)
        for y in range(25, 280, 28):
            cv2.line(image, (70, y), (70, y + 23), (240, 240, 240), 2)
        cv2.line(image, (140, 40), (270, 240), (235, 235, 235), 2)
        restored, mask = restore_photo(image)
        self.assertGreater(np.mean(mask[30:270, 69:72] > 0), 0.75)
        self.assertGreater(np.mean(mask > 0), 0)
        self.assertLess(abs(float(restored[35,70,0]) - float(restored[35,85,0])), 20)
        self.assertEqual(mask[200, 30], 0)

    def test_long_dark_scratch_is_detected(self):
        image = np.full((320, 320), 150, dtype=np.uint8)
        cv2.line(image, (140, 20), (140, 290), 15, 2)
        self.assertEqual(detect_damage(image)[100, 140], 255)

    def test_denoising_reduces_flat_region_noise(self):
        gray = np.clip(120 + np.random.default_rng(4).normal(0, 3, (192, 192)), 0, 255).astype(np.uint8)
        image = np.repeat(gray[:, :, None], 3, axis=2)
        restored, _ = restore_photo(image)
        self.assertLess(restored[32:-32,32:-32,0].std(), gray[32:-32,32:-32].std())

    def test_preserves_dimensions_dtype_and_input(self):
        image = np.full((80, 100, 3), [70, 100, 150], np.uint8)
        original = image.copy()
        restored, mask = restore_photo(image)
        self.assertEqual(restored.shape, image.shape)
        self.assertEqual(restored.dtype, np.uint8)
        self.assertEqual(mask.shape, image.shape[:2])
        np.testing.assert_array_equal(image, original)

    def test_smooth_photographic_edges_are_not_scratches(self):
        gray = np.full((320,320), 120, dtype=np.uint8)
        cv2.ellipse(gray, (160,160), (65,95), 0, 0, 360, 170, -1)
        gray = cv2.GaussianBlur(gray, (11,11), 2)
        self.assertLess(np.mean(detect_damage(gray)>0), 0.001)

    def test_invalid_inputs_and_strength(self):
        for image in (None, np.zeros((40,40),np.uint8), np.zeros((12,12,3),np.uint8)):
            with self.assertRaises(ValueError):
                restore_photo(image)
        for strength in (0, 3, float('nan')):
            with self.assertRaises(ValueError):
                restore_photo(np.zeros((64,64,3),np.uint8), strength)

    def test_sample_photo_scratches_are_reduced_without_masking_face(self):
        path = Path(__file__).resolve().parents[1] / 'assets' / 'old-photo-restoration-test.png'
        if not path.exists():
            self.skipTest('Local generated photo fixture not available')
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        restored, mask = restore_photo(image)
        # The long vertical scratch on the right background in the sample.
        region = np.s_[650:900, 1025:1050]
        def scratch_energy(photo):
            gray = cv2.cvtColor(photo,cv2.COLOR_RGB2GRAY)
            return np.maximum(gray.astype(float) - cv2.medianBlur(gray,9), 0)[region].mean()
        self.assertLess(scratch_energy(restored), scratch_energy(image) * 0.5)
        self.assertLess(np.mean(mask[250:850,350:800]>0), 0.015)


if __name__ == '__main__':
    unittest.main()
