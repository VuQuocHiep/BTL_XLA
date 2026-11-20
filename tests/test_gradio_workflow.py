import tempfile
from dataclasses import asdict, replace
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from demo.photo_restoration import RestorationOptions, RestorationResult
from demo.restoration_workflow import RestorationWorkflow, read_rgb, fingerprint
from demo.quality_metrics import comparison_metrics


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory();self.addCleanup(self.folder.cleanup)
        self.image=np.full((80,80,3),[30,60,90],np.uint8)
        self.restored=np.full((80,80,3),[80,110,140],np.uint8)
        self.options=RestorationOptions(detector='opencv',repair_backend='telea')
        result=RestorationResult(self.restored,np.zeros((80,80),np.uint8),
                                {'Source':self.image,'Repaired':self.restored},
                                {'Total':1.25},[],asdict(self.options))
        self.pipeline=Mock(return_value=result)
        self.colorizer=Mock()
        self.colorizer.process.side_effect=lambda bgr:bgr.copy()
        self.flow=RestorationWorkflow(self.folder.name,self.colorizer,self.pipeline)

    def test_default_colorization_uses_exact_restored_file(self):
        run=self.flow.colorize(self.image,self.options)
        np.testing.assert_array_equal(self.colorizer.process.call_args.args[0],self.restored[:,:,::-1])
        np.testing.assert_array_equal(read_rgb(run['source']),self.restored)
        self.assertGreaterEqual(run['total_seconds'],1.25)

    def test_baseline_bypasses_restoration(self):
        run=self.flow.colorize(self.image,use_restoration=False)
        self.pipeline.assert_not_called()
        np.testing.assert_array_equal(self.colorizer.process.call_args.args[0],self.image[:,:,::-1])
        self.assertEqual(run['restoration_seconds'],0)

    def test_valid_cached_restoration_is_reused(self):
        state=self.flow.prepare(self.image,self.options)
        self.flow.colorize(self.image,self.options,restored=state)
        self.pipeline.assert_called_once()
        self.assertIsInstance(state['restored'],str)
        self.assertTrue(all(isinstance(path,str) for path in state['paths'].values()))

    def test_changed_source_options_or_mask_invalidate_cache(self):
        state=self.flow.prepare(self.image,self.options)
        self.flow.prepare(self.image+1,self.options,previous=state)
        self.flow.prepare(self.image,replace(self.options,denoise=False),previous=state)
        self.flow.prepare(self.image,self.options,manual_mask=np.zeros((80,80),np.uint8),previous=state)
        self.assertEqual(self.pipeline.call_count,4)

    def test_comparison_uses_same_source_and_sequential_model_calls(self):
        comparison=self.flow.compare(self.image,self.options)
        calls=self.colorizer.process.call_args_list
        np.testing.assert_array_equal(calls[0].args[0],self.image[:,:,::-1])
        np.testing.assert_array_equal(calls[1].args[0],self.restored[:,:,::-1])
        self.assertEqual(comparison['enhanced']['restoration_seconds'],1.25)

    def test_no_reference_does_not_fabricate_quality_scores(self):
        comparison=self.flow.compare(self.image,self.options)
        evaluator=Mock()
        table,plot,note=comparison_metrics(None,comparison,evaluator)
        evaluator.assert_not_called()
        self.assertTrue(all(row[1:]==['N/A','N/A'] for row in table[:4]))
        self.assertEqual(len(plot.axes),1)
        self.assertIn('chưa kết luận',note)

    def test_both_branches_use_same_clean_reference(self):
        comparison=self.flow.compare(self.image,self.options)
        evaluator=Mock(return_value=({'PSNR':30.,'SSIM':.9,'LPIPS':.1,'DeltaE00':2.},''))
        table,plot,note=comparison_metrics(self.image,comparison,evaluator)
        self.assertEqual(evaluator.call_count,2)
        for call in evaluator.call_args_list:
            self.assertIs(call.args[0],self.image)
            self.assertTrue(call.kwargs['aligned'])
        self.assertEqual(len(plot.axes),4)

    def test_missing_source_is_rejected(self):
        for action in (self.flow.prepare,self.flow.colorize):
            with self.assertRaises(ValueError):action(None)

    def test_lpips_two_branches_share_one_worker_batch(self):
        comparison=self.flow.compare(self.image,self.options)
        with patch('demo.metrics_backend.available',return_value=True), \
                patch('demo.metrics_backend.lpips_reference_files',return_value=[0.1,0.2]) as batch, \
                patch('demo.quality_metrics.calculate_lpips') as individual:
            table,plot,note=comparison_metrics(self.image,comparison)
        batch.assert_called_once()
        individual.assert_not_called()
        self.assertEqual(table[2][1:],['0.100','0.200'])

    def test_png_roundtrip_preserves_rgb_and_source(self):
        original=self.image.copy()
        path=self.flow.save(self.image,'source')
        np.testing.assert_array_equal(read_rgb(path),self.image)
        np.testing.assert_array_equal(original,self.image)
