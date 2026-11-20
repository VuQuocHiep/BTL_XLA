import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from demo.object_removal import remove_object, removal_mask, RemovalSession
from demo.object_segmentation import shared_segmenter


class RemovalTests(unittest.TestCase):
    def setUp(self):
        self.image=np.full((100,140,3),[90,140,180],np.uint8)
        self.image[35:60,55:75]=[240,30,20]
        self.mask=np.zeros((100,140),bool)
        self.mask[35:60,55:75]=True

    def test_both_opencv_algorithms_remove_object_and_preserve_outside(self):
        for backend in ['telea','ns']:
            result=remove_object(self.image,self.mask,backend,3,3)
            effective=removal_mask(self.mask,3)
            np.testing.assert_array_equal(result[~effective],self.image[~effective])
            self.assertLess(np.abs(result[self.mask].astype(float)-[90,140,180]).mean(),8)
            self.assertEqual(result.dtype,np.uint8)

    def test_lama_reuses_cpu_worker_and_only_copies_selected_pixels(self):
        runner=Mock(return_value=np.full_like(self.image,50))
        result=remove_object(self.image,self.mask,'lama',3,2,runner)
        runner.assert_called_once()
        self.assertEqual(runner.call_args.args[0],'lama')
        self.assertEqual(runner.call_args.kwargs['max_side'],512)
        effective=removal_mask(self.mask,2)
        np.testing.assert_array_equal(result[~effective],self.image[~effective])

    def test_confirmation_margin_undo_and_sequential_removal(self):
        state=RemovalSession.load(self.image)
        predictor=Mock();predictor.predict.return_value=self.mask
        state.select((60,40),True,predictor)
        with self.assertRaises(ValueError):state.remove()
        state.confirm();state.set_margin(5)
        with self.assertRaises(ValueError):state.remove(margin=5)
        state.confirm();state.remove(margin=5)
        self.assertIsNone(state.mask)
        predictor.predict.assert_called_once()
        state.undo();np.testing.assert_array_equal(state.current,self.image)
        state.select((60,40),True,predictor);state.confirm();state.remove(margin=5)
        state.reset();np.testing.assert_array_equal(state.current,self.image)

    def test_empty_full_wrong_size_and_invalid_parameters_rejected(self):
        for mask in [np.zeros_like(self.mask),np.ones_like(self.mask),self.mask[:50]]:
            with self.assertRaises(ValueError):remove_object(self.image,mask)
        for kwargs in [dict(backend='bad'),dict(radius=0),dict(margin=-1),dict(margin=float('nan'))]:
            with self.assertRaises(ValueError):remove_object(self.image,self.mask,**kwargs)

    def test_failed_backend_keeps_image_and_history(self):
        state=RemovalSession.load(self.image)
        state.mask=self.mask;state.confirm()
        with self.assertRaises(ValueError):state.remove('bad')
        np.testing.assert_array_equal(state.current,self.image)
        self.assertEqual(state.history,[])
        self.assertTrue(state.confirmed)

    def test_tabs_share_one_segmenter(self):
        self.assertIs(shared_segmenter(),shared_segmenter())


if __name__=='__main__':unittest.main()
