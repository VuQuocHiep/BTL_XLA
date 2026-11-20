"""Lazy, shared MobileSAM Tiny-ViT; one bounded embedding cache, serialized access."""
import hashlib
import os
from pathlib import Path
import sys
import threading
from functools import lru_cache

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'third_party' / 'MobileSAM'
RUNTIME = ROOT / 'third_party' / 'mobilesam_runtime'


@lru_cache(maxsize=1)
def shared_segmenter():
    return MobileSegmenter()


class MobileSegmenter:
    def __init__(self, checkpoint=None, device=None):
        self.checkpoint = Path(checkpoint or ROOT / 'pretrain/mobile_sam.pt')
        self.device = device or os.environ.get('DDCOLOR_SAM_DEVICE', 'cpu')
        if self.device not in ('cpu', 'cuda', 'auto'):
            raise ValueError('DDCOLOR_SAM_DEVICE phải là cpu, cuda hoặc auto.')
        self.predictor = None
        self.image_key = None
        self.lock = threading.RLock()
        self.fell_back = False

    def _load(self):
        if self.predictor is not None:
            return
        if not self.checkpoint.is_file():
            raise ValueError('Thiếu pretrain/mobile_sam.pt. Xem docs/object_recoloring.md.')
        for folder in (RUNTIME, VENDOR):
            if folder.is_dir() and str(folder) not in sys.path:
                sys.path.insert(0, str(folder))
        try:
            import torch
            from mobile_sam import sam_model_registry, SamPredictor
        except ImportError as error:
            raise ValueError('Thiếu MobileSAM/timm. Xem docs/object_recoloring.md.') from error
        model = sam_model_registry['vit_t'](checkpoint=str(self.checkpoint))
        device = self.device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 2 * 1024**3 else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
            self.fell_back = True
        try:
            model.to(device).eval()
        except torch.cuda.OutOfMemoryError:
            model.cpu().eval()
            torch.cuda.empty_cache()
            self.fell_back = True
        self.predictor = SamPredictor(model)

    def predict(self, image, points, labels):
        import torch
        key = (image.shape, hashlib.sha256(image.tobytes()).digest())
        with self.lock, torch.inference_mode():
            self._load()
            def infer():
                if key != self.image_key:
                    self.predictor.set_image(image, image_format='RGB')
                    self.image_key = key
                masks, scores, _ = self.predictor.predict(
                    point_coords=np.asarray(points, np.float32),
                    point_labels=np.asarray(labels, np.int32), multimask_output=True)
                # Ambiguous clothing/body panels can have several valid scales.
                # Prefer masks obeying every + / - prompt before using SAM's quality score.
                xy = np.asarray(points, dtype=int)
                matches = (masks[:, xy[:, 1], xy[:, 0]] == np.asarray(labels, dtype=bool)).sum(axis=1)
                best = max(range(len(scores)), key=lambda i: (int(matches[i]), float(scores[i])))
                mask = masks[best].astype(bool)
                # Drop detached islands (e.g. a hair/background speck on a shirt mask).
                # Disconnected intended parts can be explicitly included with another + point.
                _, components = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
                positive = xy[np.asarray(labels, dtype=bool)]
                anchors = np.unique(components[positive[:, 1], positive[:, 0]])
                anchors = anchors[anchors != 0]
                if not len(anchors):
                    raise ValueError('Mask chưa chứa điểm +. Hãy chọn điểm khác trên vật thể.')
                return np.isin(components, anchors)
            try:
                return infer()
            except torch.cuda.OutOfMemoryError:
                self.predictor.reset_image()
                self.image_key = None
                self.predictor.model.cpu()
                torch.cuda.empty_cache()
                self.fell_back = True
                return infer()
