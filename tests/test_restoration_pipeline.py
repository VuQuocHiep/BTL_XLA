from dataclasses import replace
import unittest
from unittest.mock import Mock

import numpy as np

from demo.photo_restoration import RestorationOptions, restore_pipeline


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.image=np.full((96,96,3),[80,100,120],np.uint8)
        self.options=RestorationOptions(detector='opencv',repair_backend='telea',detect=False,
                                        repair=False,denoise=False,contrast=False,sharpen=False,mask_grow=0)

    def test_every_stage_can_be_disabled_without_changing_pixels(self):
        result=restore_pipeline(self.image,self.options)
        np.testing.assert_array_equal(result.image,self.image)
        self.assertFalse(np.any(result.mask))
        self.assertEqual(set(result.timings),{'Detection','Repair','Denoise','Contrast','Sharpen','Total'})
        self.assertEqual(set(result.stages),{'Source','Damage overlay','Repaired','Denoised','Contrast','Sharpened'})

    def test_lama_only_replaces_masked_pixels(self):
        mask=np.zeros((96,96),np.uint8);mask[30:45,30:45]=255
        prediction=np.full_like(self.image,180)
        runner=Mock(return_value=prediction)
        result=restore_pipeline(self.image,replace(self.options,repair=True,repair_backend='lama'),mask,runner)
        np.testing.assert_array_equal(result.image[mask==0],self.image[mask==0])
        np.testing.assert_array_equal(result.image[mask>0],prediction[mask>0])
        self.assertEqual(runner.call_args.args[0],'lama')

    def test_microsoft_probability_threshold_and_geometry(self):
        probability=np.zeros((96,96),np.float32);probability[20:60,40:42]=.7
        runner=Mock(return_value=probability)
        result=restore_pipeline(self.image,replace(self.options,detect=True,detector='microsoft'),runner=runner)
        self.assertEqual(runner.call_args.args[0],'scratch')
        np.testing.assert_array_equal(result.mask,np.uint8(probability>=.4)*255)

    def test_manual_mask_overrides_detection(self):
        mask=np.zeros((96,96),np.uint8);mask[10:20,10:20]=255
        runner=Mock()
        result=restore_pipeline(self.image,replace(self.options,detect=True,detector='microsoft'),mask,runner)
        runner.assert_not_called()
        np.testing.assert_array_equal(result.mask,mask)

    def test_mask_size_mismatch_is_not_silently_resized(self):
        with self.assertRaises(ValueError):restore_pipeline(self.image,self.options,np.zeros((32,32),np.uint8))

    def test_explicit_backend_failure_is_not_silently_replaced(self):
        runner=Mock(side_effect=ValueError('checkpoint unavailable'))
        with self.assertRaisesRegex(ValueError,'checkpoint unavailable'):
            restore_pipeline(self.image,replace(self.options,detect=True,detector='microsoft'),runner=runner)

    def test_excessive_mask_skips_filling_and_reports_it(self):
        result=restore_pipeline(self.image,replace(self.options,repair=True),np.full((96,96),255,np.uint8))
        np.testing.assert_array_equal(result.image,self.image)
        self.assertTrue(any('Bỏ qua vá' in note for note in result.notes))

    def test_suspicious_automatic_mask_remains_visible_but_is_not_filled(self):
        probability=np.zeros((96,96),np.float32);probability[:20]=.9
        result=restore_pipeline(self.image,replace(self.options,detect=True,detector='microsoft',repair=True),
                                 runner=Mock(return_value=probability))
        self.assertTrue(np.any(result.mask))
        np.testing.assert_array_equal(result.image,self.image)
        self.assertTrue(any('Bỏ qua vá' in note for note in result.notes))
