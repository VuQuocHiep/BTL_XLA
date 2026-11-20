import os
from dataclasses import replace
import unittest

import numpy as np

from demo.photo_restoration import RestorationOptions, restore_pipeline
from demo.restoration_backends import run_model


@unittest.skipUnless(os.environ.get('RUN_RESTORATION_INTEGRATION')=='1',
                     'Opt in to actual downloaded pretrained CPU workers')
class PretrainedTests(unittest.TestCase):
    def test_microsoft_returns_probabilities_in_original_geometry(self):
        image=np.full((96,128,3),120,np.uint8);image[15:80,40:42]=245
        probability=run_model('scratch',image,max_side=256)
        self.assertEqual(probability.shape,image.shape[:2])
        self.assertTrue(np.all(np.isfinite(probability)))
        self.assertGreaterEqual(float(probability.min()),0)
        self.assertLessEqual(float(probability.max()),1)

    def test_lama_repairs_a_hole_and_preserves_unmasked_pixels(self):
        clean=np.full((128,128,3),120,np.uint8)
        image=clean.copy();image[45:70,45:70]=250
        mask=np.zeros((128,128),np.uint8);mask[45:70,45:70]=255
        options=RestorationOptions(detect=False,repair=True,repair_backend='lama',denoise=False,
                                    contrast=False,sharpen=False,mask_grow=0,max_model_side=256)
        result=restore_pipeline(image,options,manual_mask=mask)
        np.testing.assert_array_equal(result.image[mask==0],image[mask==0])
        before=np.mean(np.abs(image[mask>0].astype(float)-clean[mask>0]))
        after=np.mean(np.abs(result.image[mask>0].astype(float)-clean[mask>0]))
        self.assertLess(after,before)
