"""CPU-only pretrained inference in an isolated, bounded subprocess.

No package installation, network access, CUDA allocation or parent Torch
settings are performed here. Protocol: request JSON path, response JSON path.
"""
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['TORCH_HOME'] = str(ROOT / 'pretrain/restoration/torch')

import cv2
import numpy as np
import torch


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    torch.set_num_threads(2)
    image = cv2.cvtColor(cv2.imread(request['image']), cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    limit = int(request.get('max_side', 512))
    scale = min(1., limit / max(height, width))
    small = cv2.resize(image, (max(32, round(width * scale)), max(32, round(height * scale))),
                       interpolation=cv2.INTER_AREA)
    model_path = Path(request['model'])
    if request['mode'] == 'lpips':
        if not model_path.is_file():
            raise ValueError('AlexNet checkpoint unavailable; runtime downloading is disabled')
        sys.path.insert(0, str(ROOT / 'pretrain/restoration/python'))
        import lpips
        model = lpips.LPIPS(net='alex',version='0.1',verbose=False).cpu().eval()
        def tensor(rgb):
            return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).float()[None]/127.5-1
        reference=tensor(image)
        scores=[]
        with torch.inference_mode():
            for path in request['others']:
                other=cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
                if other.shape != image.shape:
                    raise ValueError('LPIPS inputs must have identical shapes')
                scores.append(float(model(reference,tensor(other)).item()))
        Path(request['output']).write_text(json.dumps(scores),encoding='utf-8')
    elif request['mode'] == 'scratch':
        from PIL import Image
        from third_party.microsoft_scratch.networks import UNet

        model = UNet(in_channels=1, out_channels=1, depth=4, conv_num=2, wf=6,
                     padding=True, batch_norm=True, up_mode='upsample', with_tanh=False,
                     sync_bn=False, antialiasing=True)
        checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state'], strict=True)
        del checkpoint
        model.cpu().eval()
        sh, sw = small.shape[:2]
        # Official detector: grayscale, bicubic multiple-of-16 sizing, [-1,1], sigmoid.
        size = (max(32, round(sw / 16) * 16), max(32, round(sh / 16) * 16))
        pil = Image.fromarray(small).resize(size, Image.Resampling.BICUBIC).convert('L')
        tensor = torch.from_numpy(np.asarray(pil).copy()).float()[None, None] / 127.5 - 1
        with torch.inference_mode():
            probability = torch.sigmoid(model(tensor))[0, 0].numpy()
        probability = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
        np.save(request['output'], probability)
    elif request['mode'] == 'lama':
        mask = cv2.imread(request['mask'], cv2.IMREAD_GRAYSCALE)
        mask = np.uint8(cv2.resize(mask, (small.shape[1], small.shape[0]),
                                   interpolation=cv2.INTER_AREA) > 0) * 255
        sh, sw = small.shape[:2]
        ph, pw = (-sh) % 8, (-sw) % 8
        rgb = cv2.copyMakeBorder(small, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
        mask = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
        rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()[None] / 255
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask > 0)).float()[None, None]
        model = torch.jit.load(str(model_path), map_location='cpu').eval()
        with torch.inference_mode():
            output = model(rgb_tensor, mask_tensor)[0].permute(1, 2, 0).cpu().numpy()
        output = np.clip(output[:sh, :sw] * 255, 0, 255).round().astype(np.uint8)
        output = cv2.resize(output, (width, height), interpolation=cv2.INTER_CUBIC)
        # The caller pastes only original-resolution mask pixels, never this whole resize.
        if not cv2.imwrite(request['output'], cv2.cvtColor(output, cv2.COLOR_RGB2BGR)):
            raise RuntimeError('Cannot save LaMa output')
    else:
        raise ValueError('Unsupported worker mode')


if __name__ == '__main__':
    main()
