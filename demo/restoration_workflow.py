"""Session-file workflow; model/pipeline injection keeps source routing testable."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import cv2
import numpy as np

from demo.photo_restoration import RestorationOptions, restore_pipeline


def fingerprint(image):
    if image is None:
        return '-'
    image = np.ascontiguousarray(image)
    digest = hashlib.sha256(str((image.shape, image.dtype)).encode())
    digest.update(image.data)
    return digest.hexdigest()


def read_rgb(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError('Ảnh tạm không còn khả dụng. Vui lòng xử lý lại.')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class RestorationWorkflow:
    def __init__(self, folder, colorizer, pipeline=restore_pipeline):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.colorizer = colorizer
        self.pipeline = pipeline

    def save(self, image, label):
        path = self.folder / f'{uuid4().hex}-{label}.png'
        output = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
        if not cv2.imwrite(str(path), output, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise ValueError('Không lưu được kết quả xử lý.')
        return str(path)

    def key(self, image, options, manual_mask):
        return (fingerprint(image), json.dumps(asdict(options), sort_keys=True), fingerprint(manual_mask))

    def prepare(self, image, options=None, manual_mask=None, previous=None):
        if image is None:
            raise ValueError('Vui lòng tải ảnh lên trước khi phục hồi.')
        options = options or RestorationOptions()
        key = self.key(image, options, manual_mask)
        if previous and tuple(previous['key']) == key and Path(previous['restored']).is_file():
            return previous
        result = self.pipeline(image, options, manual_mask=manual_mask)
        paths = {name: self.save(stage, name.replace(' ', '-')) for name, stage in result.stages.items()}
        state = {'key': key, 'source': paths['Source'], 'restored': self.save(result.image, 'restored'),
                 'mask': self.save(result.mask, 'mask'), 'paths': paths,
                 'timings': result.timings, 'notes': result.notes, 'options': result.options,
                 'coverage': float(np.mean(result.mask > 0))}
        metadata = self.folder / f'{uuid4().hex}-restoration.json'
        metadata.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf-8')
        return state

    def colorize(self, image, options=None, manual_mask=None, restored=None, use_restoration=True):
        if image is None:
            raise ValueError('Vui lòng tải ảnh lên trước khi tô màu.')
        if use_restoration:
            restored = self.prepare(image, options, manual_mask, restored)
            model_input = read_rgb(restored['restored'])
            source_path = restored['restored']
        else:
            model_input = image
            source_path = self.save(image, 'source')
        started = perf_counter()
        bgr = self.colorizer.process(cv2.cvtColor(model_input, cv2.COLOR_RGB2BGR))
        color_seconds = perf_counter() - started
        color_path = self.save(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 'colorized')
        restore_seconds = restored['timings']['Total'] if use_restoration else 0.0
        return {'source': source_path, 'output': color_path, 'color_seconds': color_seconds,
                'restoration_seconds': restore_seconds, 'total_seconds': color_seconds + restore_seconds,
                'restored': restored, 'method': 'Restoration + DDColor' if use_restoration else 'DDColor'}

    def compare(self, image, options=None, manual_mask=None, restored=None):
        # Sequential inference: one DDColor instance, no competing CUDA models.
        baseline = self.colorize(image, use_restoration=False)
        enhanced = self.colorize(image, options, manual_mask, restored, use_restoration=True)
        return {'baseline': baseline, 'enhanced': enhanced, 'restored': enhanced['restored']}


def stage_gallery(state):
    if not state:
        return []
    return [(path, name) for name, path in state['paths'].items()]
