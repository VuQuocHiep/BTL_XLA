"""Optional official pretrained models, without importing their repositories."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'pretrain' / 'restoration'


def available_models():
    return {'scratch': (MODEL_DIR / 'microsoft-scratch.pt').is_file()
            and (ROOT / 'third_party/microsoft_scratch/networks.py').is_file(),
            'lama': (MODEL_DIR / 'big-lama.pt').is_file()}


def run_model(mode, image, mask=None, max_side=512):
    filename = 'microsoft-scratch.pt' if mode == 'scratch' else 'big-lama.pt'
    if mode not in ('scratch', 'lama') or not available_models()[mode]:
        raise ValueError(f'Model {mode} chưa sẵn sàng. Chạy scripts/fetch_restoration_models.py để tải riêng model.')
    with tempfile.TemporaryDirectory(prefix='ddcolor-worker-') as folder:
        folder = Path(folder)
        image_path = folder / 'input.png'
        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        output = folder / ('probability.npy' if mode == 'scratch' else 'result.png')
        request = {'mode': mode, 'image': str(image_path), 'model': str(MODEL_DIR / filename),
                   'output': str(output), 'max_side': int(max_side)}
        if mask is not None:
            mask_path = folder / 'mask.png'
            cv2.imwrite(str(mask_path), mask)
            request['mask'] = str(mask_path)
        request_path = folder / 'request.json'
        request_path.write_text(json.dumps(request), encoding='utf-8')
        environment = os.environ.copy()
        environment['CUDA_VISIBLE_DEVICES'] = ''
        environment['OMP_NUM_THREADS'] = '2'
        # Optional separate compatible environment; never modify current Conda.
        python = os.environ.get('DDCOLOR_RESTORATION_PYTHON', sys.executable)
        try:
            result = subprocess.run([python, str(ROOT / 'demo/restoration_worker.py'), str(request_path)],
                                    cwd=str(ROOT), env=environment, capture_output=True,
                                    text=True, encoding='utf-8', errors='replace', timeout=180,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except subprocess.TimeoutExpired as error:
            raise ValueError(f'{mode} vượt thời gian chờ 180 giây; không dùng kết quả chưa hoàn tất.') from error
        if result.returncode:
            raise ValueError(f'{mode} không chạy được trong môi trường hiện tại: {result.stderr[-1200:]}')
        if mode == 'scratch':
            probability = np.load(str(output), allow_pickle=False)
            if probability.shape != image.shape[:2] or not np.all(np.isfinite(probability)):
                raise ValueError('Scratch detector trả về dữ liệu không hợp lệ.')
            return probability
        output_bgr = cv2.imread(str(output))
        if output_bgr is None or output_bgr.shape != image.shape:
            raise ValueError('LaMa trả về ảnh không hợp lệ.')
        return cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
