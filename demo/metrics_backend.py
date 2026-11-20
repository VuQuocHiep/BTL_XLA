"""Optional LPIPS in a CPU worker using project-local extras, not Conda installs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2

ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT=ROOT/'pretrain/restoration/torch/hub/checkpoints/alexnet-owt-7be5be79.pth'


def available():
    return CHECKPOINT.is_file() and (ROOT/'pretrain/restoration/python/lpips').is_dir()


def lpips_files(reference,others):
    if not available():
        raise ImportError('Project-local LPIPS extras or AlexNet checkpoint unavailable')
    with tempfile.TemporaryDirectory(prefix='ddcolor-lpips-') as directory:
        folder=Path(directory)
        output=folder/'scores.json'
        request={'mode':'lpips','image':str(reference),'others':[str(path) for path in others],
                 'model':str(CHECKPOINT),'output':str(output)}
        path=folder/'request.json';path.write_text(json.dumps(request),encoding='utf-8')
        environment=os.environ.copy();environment['CUDA_VISIBLE_DEVICES']=''
        try:
            result=subprocess.run([os.environ.get('DDCOLOR_RESTORATION_PYTHON',sys.executable),
                                   str(ROOT/'demo/restoration_worker.py'),str(path)],cwd=str(ROOT),
                                  env=environment,capture_output=True,text=True,encoding='utf-8',errors='replace',
                                  timeout=180,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        except subprocess.TimeoutExpired as error:
            raise RuntimeError('LPIPS worker exceeded 180 seconds') from error
        if result.returncode:
            raise RuntimeError(result.stderr[-1200:])
        scores=json.loads(output.read_text(encoding='utf-8'))
        if len(scores)!=len(others):
            raise RuntimeError('LPIPS response length mismatch')
        return scores


def lpips_reference_files(reference, others):
    with tempfile.TemporaryDirectory(prefix='ddcolor-lpips-reference-') as directory:
        path=Path(directory)/'reference.png'
        cv2.imwrite(str(path),cv2.cvtColor(reference,cv2.COLOR_RGB2BGR))
        return lpips_files(path,others)


def lpips_pair(reference,processed):
    with tempfile.TemporaryDirectory(prefix='ddcolor-lpips-input-') as directory:
        folder=Path(directory)
        first,second=folder/'reference.png',folder/'processed.png'
        cv2.imwrite(str(first),cv2.cvtColor(reference,cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(second),cv2.cvtColor(processed,cv2.COLOR_RGB2BGR))
        return lpips_files(first,[second])[0]
