"""Download only the selected pretrained assets; never install dependencies.

LaMa URL/checksum: IOPaint's iopaint/model/lama.py.
Scratch weights/source: Microsoft's official v1.0 release and MIT repository.
"""
import argparse
import hashlib
import io
import json
import struct
import zlib
import shutil
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'pretrain' / 'restoration'
LAMA_URL = 'https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt'
MS_URL = 'https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/global_checkpoints.zip'


def request(url, start=None, end=None):
    headers = {'User-Agent': 'DDColor-restoration/1.0'}
    if start is not None:
        headers['Range'] = f'bytes={start}-{end}'
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90)


class RemoteZip(io.RawIOBase):
    """Seek using HTTP ranges; do not download the 2 GB global restoration bundle."""
    def __init__(self, url):
        self.url, self.position = url, 0
        with request(url, 0, 0) as response:
            if response.status != 206:
                raise RuntimeError('Server does not support selective range downloads')
            self.size = int(response.headers['Content-Range'].split('/')[-1])

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        size = min(self.size - self.position, size if size >= 0 else self.size - self.position)
        if size <= 0:
            return b''
        with request(self.url, self.position, self.position + size - 1) as response:
            if response.status != 206:
                raise RuntimeError('Range request was not honored; refusing full bundle download')
            value = response.read(size)
        self.position += len(value)
        return value


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['lama', 'scratch', 'probe-scratch', 'lpips'], required=True)
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    if args.model in ('lama', 'scratch'):
        cached_target = DEST / ('big-lama.pt' if args.model == 'lama' else 'microsoft-scratch.pt')
        manifest = cached_target.with_suffix('.json')
        source_ready = args.model == 'lama' or (ROOT / 'third_party/microsoft_scratch/networks.py').is_file()
        if cached_target.exists() and manifest.exists() and source_ready:
            info = json.loads(manifest.read_text(encoding='utf-8'))
            if sha256(cached_target) == info.get('sha256'):
                print(json.dumps({'already_verified': str(cached_target)}), flush=True)
                return
    if args.model == 'lama':
        target = DEST / 'big-lama.pt'
        part = target.with_suffix('.part')
        digest = hashlib.md5()
        with request(LAMA_URL) as response, part.open('wb') as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != 'e3aa4aaa15225a33ec84f9f4bc47e500':
            raise RuntimeError('LaMa checksum mismatch; downloaded file was not activated')
        part.replace(target)
        info = {'source': LAMA_URL, 'md5': digest.hexdigest(), 'sha256': sha256(target)}
    elif args.model == 'lpips':
        target = DEST / 'torch/hub/checkpoints/alexnet-owt-7be5be79.pth'
        target.parent.mkdir(parents=True, exist_ok=True)
        url = 'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth'
        cached = ROOT / 'tmp/torch/hub/checkpoints' / target.name
        if not target.exists():
            part = target.with_suffix('.part')
            if cached.exists():
                shutil.copyfile(cached, part)
            else:
                with request(url) as response, part.open('wb') as output:
                    shutil.copyfileobj(response, output, 1024 * 1024)
            if not sha256(part).startswith('7be5be79'):
                raise RuntimeError('AlexNet checksum mismatch')
            part.replace(target)
        digest = sha256(target)
        if not digest.startswith('7be5be79'):
            raise RuntimeError('Cached AlexNet checksum mismatch')
        info = {'source': url, 'sha256': digest}
    else:
        with zipfile.ZipFile(RemoteZip(MS_URL)) as archive:
            members = [member for member in archive.infolist() if member.filename.endswith('detection/FT_Epoch_latest.pt')]
            if len(members) != 1:
                raise RuntimeError('Cannot uniquely locate Microsoft scratch checkpoint')
            member = members[0]
            print(json.dumps({'member': member.filename, 'compressed_bytes': member.compress_size,
                              'uncompressed_bytes': member.file_size}), flush=True)
            if args.model == 'probe-scratch':
                return
            target = DEST / 'microsoft-scratch.pt'
            part = target.with_suffix('.part')
            # One range stream instead of hundreds of small ZIP read requests.
            archive.fp.seek(member.header_offset)
            header = archive.fp.read(30)
            name_length, extra_length = struct.unpack_from('<HH', header, 26)
            start = member.header_offset + 30 + name_length + extra_length
            compressed = DEST / 'microsoft-scratch.deflate.part'
            have = compressed.stat().st_size if compressed.exists() else 0
            if have < member.compress_size:
                with request(MS_URL, start + have, start + member.compress_size - 1) as response, compressed.open('ab') as output:
                    if response.status != 206:
                        raise RuntimeError('Selected checkpoint range not honored')
                    while have < member.compress_size:
                        chunk = response.read(min(1024 * 1024, member.compress_size - have))
                        if not chunk:
                            raise RuntimeError('Incomplete checkpoint download; rerun to resume')
                        output.write(chunk)
                        have += len(chunk)
            decoder = zlib.decompressobj(-zlib.MAX_WBITS)
            crc, size = 0, 0
            if member.compress_type != zipfile.ZIP_DEFLATED:
                raise RuntimeError('Unexpected checkpoint ZIP compression')
            with compressed.open('rb') as stream, part.open('wb') as output:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    data = decoder.decompress(chunk)
                    output.write(data)
                    crc = zlib.crc32(data, crc)
                    size += len(data)
            if crc != member.CRC or size != member.file_size or not decoder.eof:
                raise RuntimeError('Microsoft checkpoint CRC/size mismatch')
            part.replace(target)
            compressed.unlink()
        vendor = ROOT / 'third_party' / 'microsoft_scratch'
        vendor.mkdir(parents=True, exist_ok=True)
        (vendor / '__init__.py').write_text('', encoding='utf-8')
        repo = 'https://raw.githubusercontent.com/microsoft/Bringing-Old-Photos-Back-to-Life/master/'
        sources = {}
        for name in ('networks.py', 'antialiasing.py', 'LICENSE'):
            url = repo + ('Global/detection_models/' if name != 'LICENSE' else '') + name
            with request(url) as response:
                source = response.read().decode('utf-8')
            if name == 'networks.py':
                source = source.replace('from detection_models.sync_batchnorm import DataParallelWithCallback',
                                        'from torch.nn import DataParallel as DataParallelWithCallback')
                source = source.replace('from detection_models.antialiasing import Downsample',
                                        'from .antialiasing import Downsample')
            (vendor / name).write_text(source, encoding='utf-8')
            sources[name] = {'url': url, 'sha256': sha256(vendor / name)}
        (vendor / 'NOTICE.md').write_text(
            'Microsoft Bringing Old Photos Back to Life, MIT license.\n'
            'networks.py: relative antialiasing import; DataParallel alias replaces unused sync-BN import.\n'
            'Inference uses sync_bn=False on CPU; no network layers or weight keys changed.\n', encoding='utf-8')
        info = {'source': MS_URL, 'member': member.filename, 'crc32': member.CRC,
                'sha256': sha256(target), 'sources': sources}
    target.with_suffix('.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps({'saved': str(target), 'bytes': target.stat().st_size}), flush=True)


if __name__ == '__main__':
    main()
