import os
import unittest
from unittest.mock import patch

import numpy as np

from demo.quality_metrics import calculate_lpips, evaluate_quality, validate_pair


class QualityMetricTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(42).integers(0, 256, (96, 96, 3), dtype=np.uint8)

    def test_identical_pair_has_perfect_pixel_metrics(self):
        with patch('demo.quality_metrics.calculate_lpips', return_value=0.0):
            scores, _ = evaluate_quality(self.image, self.image.copy(), True)
        self.assertTrue(np.isinf(scores['PSNR']))
        self.assertAlmostEqual(scores['SSIM'], 1.0)
        self.assertAlmostEqual(scores['DeltaE00'], 0.0)

    def test_known_pixel_error_and_perceptual_result(self):
        reference = np.full((64, 64, 3), 100, dtype=np.uint8)
        processed = reference + 10
        with patch('demo.quality_metrics.calculate_lpips', return_value=0.125) as perceptual:
            scores, _ = evaluate_quality(reference, processed, True)
        self.assertAlmostEqual(scores['PSNR'], 28.130803608679106)
        self.assertTrue(0 < scores['SSIM'] < 1)
        self.assertEqual(scores['LPIPS'], 0.125)
        self.assertGreater(scores['DeltaE00'], 0)
        self.assertIs(perceptual.call_args.args[0], reference)
        self.assertIs(perceptual.call_args.args[1], processed)

    def test_missing_lpips_is_not_reported_as_zero(self):
        with patch('demo.quality_metrics.calculate_lpips', side_effect=ImportError('lpips')):
            scores, status = evaluate_quality(self.image, self.image, True)
        self.assertIsNone(scores['LPIPS'])
        self.assertIn('LPIPS chưa khả dụng', status)

    def test_mismatched_images_are_not_silently_resized(self):
        with self.assertRaisesRegex(ValueError, 'cùng kích thước'):
            validate_pair(self.image, self.image[:64], True)

    def test_alignment_confirmation_is_required(self):
        with self.assertRaisesRegex(ValueError, 'xác nhận'):
            validate_pair(self.image, self.image, False)

    def test_invalid_inputs(self):
        for image in (None, self.image[:, :, 0], self.image.astype(float), self.image[:16]):
            with self.subTest(shape=getattr(image, 'shape', None)):
                with self.assertRaises(ValueError):
                    validate_pair(image, self.image, True)

    @unittest.skipUnless(os.environ.get('RUN_LPIPS_INTEGRATION') == '1',
                         'Opt in to downloading pretrained AlexNet weights')
    def test_real_lpips_identical_and_distorted_pair(self):
        import torch

        torch.set_num_threads(2)
        identical = calculate_lpips(self.image, self.image.copy())
        blurred = self.image.copy()
        blurred[16:80, 16:80] = 128
        distorted = calculate_lpips(self.image, blurred)
        self.assertAlmostEqual(identical, 0.0, places=6)
        self.assertGreater(distorted, identical + 0.001)


if __name__ == '__main__':
    unittest.main()
