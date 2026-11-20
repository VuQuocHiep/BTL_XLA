import copy
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from demo.object_recoloring import RecolorSession, feather_mask, image_point, recolor
from demo.object_segmentation import MobileSegmenter


class RecolorTests(unittest.TestCase):
    def setUp(self):
        yy, xx = np.indices((60, 100))
        self.image = np.stack([60 + xx, 40 + yy, 50 + yy], axis=-1).astype(np.uint8)
        self.mask = (xx > 10) & (xx < 65) & (yy > 8) & (yy < 55)

    def test_outside_is_bit_exact_for_every_feather_and_strength(self):
        for radius in (0, 3, 30):
            for strength in (0, 25, 100):
                result = recolor(self.image, self.mask, '#14bb80', strength, radius)
                np.testing.assert_array_equal(result[~self.mask], self.image[~self.mask])
                if strength == 0:
                    np.testing.assert_array_equal(result, self.image)

    def test_lab_preserves_shading_and_texture(self):
        # Neutral target avoids gamut clipping so luminance preservation is measurable.
        result = recolor(self.image, self.mask, '#808080', 100, 0)
        before = cv2.cvtColor(self.image.astype(np.float32)/255, cv2.COLOR_RGB2LAB)[:,:,0]
        after = cv2.cvtColor(result.astype(np.float32)/255, cv2.COLOR_RGB2LAB)[:,:,0]
        self.assertLess(np.mean(np.abs(after[self.mask]-before[self.mask])), .3)
        self.assertGreater(np.corrcoef(before[self.mask], after[self.mask])[0,1], .99)

    def test_coordinate_mapping_and_reject_letterbox(self):
        self.assertEqual(image_point((99, 59), self.image.shape), (99, 59))
        self.assertEqual(image_point((25, 15), self.image.shape, (30, 50)), (50, 30))
        for index in ((-1,0), (100,0), (0,60), (float('nan'),1)):
            with self.assertRaises(ValueError): image_point(index, self.image.shape)

    def test_feather_is_inward_only(self):
        alpha = feather_mask(self.mask, 5)
        self.assertTrue(np.all(alpha[~self.mask] == 0))
        self.assertGreater(alpha[20, 20], alpha[9, 11])

    def test_selection_confirmation_and_color_changes_do_not_infer(self):
        segmenter = Mock()
        segmenter.predict.return_value = self.mask
        state = RecolorSession.load(self.image)
        state.select((20,20), True, segmenter)
        np.testing.assert_array_equal(state.current, self.image)
        with self.assertRaises(ValueError): state.apply('#ff0000',100,3)
        state.confirm()
        state.apply('#a03040',80,3)
        state.apply('#40a030',50,3)
        segmenter.predict.assert_called_once()
        np.testing.assert_array_equal(state.current, recolor(self.image,self.mask,'#40a030',50,3))
        state.select((30,30), True, segmenter)
        self.assertEqual(len(state.points), 1)
        self.assertFalse(state.confirmed)

    def test_many_points_negative_confirmation_invalidation_and_undo(self):
        segmenter = Mock()
        segmenter.predict.return_value = self.mask
        state = RecolorSession.load(self.image)
        with self.assertRaises(ValueError): state.select((20,20),False,segmenter)
        for i in range(20): state.select((20+i,20), i % 2 == 0,segmenter)
        self.assertEqual(len(state.points),20)
        state.confirm()
        state.select((40,40),False,segmenter)
        self.assertFalse(state.confirmed)
        state.confirm(); state.apply('#bb4422',100,3)
        state.undo()
        np.testing.assert_array_equal(state.current,self.image)
        self.assertIsNone(state.mask)
        state.select((20,20),True,segmenter);state.confirm();state.apply('#22bb44',100,3)
        state.reset()
        np.testing.assert_array_equal(state.current,self.image)
        self.assertEqual(len(state.history),0)

    def test_session_isolation(self):
        first = RecolorSession.load(self.image)
        second = copy.deepcopy(first)
        first.current[:] = 0
        np.testing.assert_array_equal(second.current,self.image)
        np.testing.assert_array_equal(first.original,self.image)

    def test_embedding_cache_switches_images_and_retries_oom_on_cpu(self):
        import torch
        segmenter=MobileSegmenter()
        predictor=Mock()
        predictor.predict.return_value=(self.mask[None],np.array([1.]),None)
        segmenter.predictor=predictor
        for _ in range(3): segmenter.predict(self.image,[(20,20)],[1])
        self.assertEqual(predictor.set_image.call_count,1)
        segmenter.predict(self.image+1,[(20,20)],[1])
        self.assertEqual(predictor.set_image.call_count,2)
        predictor.predict.side_effect=[torch.cuda.OutOfMemoryError(),(self.mask[None],np.array([1.]),None)]
        segmenter.predict(self.image,[(20,20)],[1])
        predictor.model.cpu.assert_called_once()
        self.assertTrue(segmenter.fell_back)

    def test_candidates_prefer_prompt_agreement_over_quality(self):
        segmenter=MobileSegmenter()
        predictor=Mock()
        wrong=np.ones_like(self.mask)
        predictor.predict.return_value=(np.stack([wrong,self.mask]),np.array([.99,.7]),None)
        segmenter.predictor=predictor
        result=segmenter.predict(self.image,[(20,20),(0,0)],[1,0])
        np.testing.assert_array_equal(result,self.mask)

    def test_detached_islands_require_a_positive_point(self):
        segmenter=MobileSegmenter()
        mask=self.mask.copy()
        mask[0:3,0:3]=True
        segmenter.predictor=Mock()
        segmenter.predictor.predict.return_value=(mask[None],np.array([1.]),None)
        result=segmenter.predict(self.image,[(20,20)],[1])
        np.testing.assert_array_equal(result,self.mask)
        result=segmenter.predict(self.image,[(20,20),(1,1)],[1,1])
        np.testing.assert_array_equal(result,mask)

    def test_failed_new_selection_keeps_previous_edit(self):
        segmenter=Mock()
        segmenter.predict.return_value=self.mask
        state=RecolorSession.load(self.image)
        state.select((20,20),True,segmenter);state.confirm();state.apply('#cc3030',100,3)
        previous=state.current.copy()
        with self.assertRaises(ValueError):state.select((20,20),False,segmenter)
        self.assertTrue(state.confirmed)
        np.testing.assert_array_equal(state.current,previous)

    def test_gradio_tab_has_readonly_click_canvas_and_upload_button(self):
        import tempfile
        import gradio as gr
        from demo.recoloring_ui import build_recoloring_tab
        with tempfile.TemporaryDirectory() as folder:
            with gr.Blocks() as blocks:
                state=gr.State(None)
                with gr.Tabs():
                    build_recoloring_tab(state,folder,dict(concurrency_id='image-pipeline',concurrency_limit=1))
            config=blocks.config
            select=next(d for d in config['dependencies'] if d['api_name']=='recolor_select')
            canvas=next(c for c in config['components'] if c['id']==select['targets'][0][0])
            self.assertFalse(canvas['props']['interactive'])
            self.assertTrue(canvas['props']['_selectable'])
            self.assertTrue(select['collects_event_data'])
            upload=next(d for d in config['dependencies'] if d['api_name']=='recolor_load')
            self.assertEqual(upload['targets'][0][1],'upload')


if __name__ == '__main__':
    unittest.main()
