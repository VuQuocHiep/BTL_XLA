"""Full-reference quality metrics for aligned uint8 RGB image pairs."""

from functools import lru_cache

import numpy as np


def validate_pair(reference, processed, aligned):
    if reference is None or processed is None:
        raise ValueError('Vui lòng tải đủ ảnh tham chiếu và ảnh sau xử lý.')
    if not aligned:
        raise ValueError('Hãy xác nhận hai ảnh cùng nội dung, góc nhìn và đã căn chỉnh trước khi đánh giá.')
    reference, processed = np.asarray(reference), np.asarray(processed)
    for image in (reference, processed):
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError('Hai ảnh phải là ảnh RGB 8 bit.')
        if min(image.shape[:2]) < 64:
            raise ValueError('Mỗi ảnh cần có chiều rộng và chiều cao ít nhất 64 pixel để tính LPIPS.')
    if reference.shape != processed.shape:
        raise ValueError('Hai ảnh phải cùng kích thước và đã căn chỉnh. '
                         'Hãy cắt/chỉnh ảnh tham chiếu tương ứng với ảnh sau xử lý; hệ thống không tự co giãn ảnh.')
    return reference, processed


@lru_cache(maxsize=1)
def _lpips_model():
    import lpips

    # Lazy loading keeps the colorization/scanning UI usable without LPIPS weights.
    return lpips.LPIPS(net='alex', version='0.1', verbose=False).cpu().eval()


def calculate_lpips(reference, processed):
    import torch

    def tensor(image):
        rgb = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)
        return torch.from_numpy(rgb).unsqueeze(0) / 127.5 - 1.0

    try:
        model = _lpips_model()
    except ImportError:
        from demo.metrics_backend import lpips_pair
        return lpips_pair(reference, processed)
    with torch.no_grad():
        return float(model(tensor(reference), tensor(processed)).item())


def evaluate_quality(reference, processed, aligned=False, compute_lpips=True):
    from skimage.metrics import structural_similarity
    from skimage.color import rgb2lab, deltaE_ciede2000

    reference, processed = validate_pair(reference, processed, aligned)
    mse = float(np.mean((reference.astype(np.float64) - processed.astype(np.float64)) ** 2))
    psnr = float('inf') if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))
    ssim = float(structural_similarity(reference, processed, data_range=255, channel_axis=-1,
                                      gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
    delta_e = float(np.mean(deltaE_ciede2000(rgb2lab(reference), rgb2lab(processed))))
    status = ''
    perceptual = None
    if compute_lpips:
        try:
            perceptual = calculate_lpips(reference, processed)
            if not np.isfinite(perceptual):
                raise RuntimeError('LPIPS trả về giá trị không hợp lệ.')
        except Exception:
            status = 'LPIPS chưa khả dụng; không có điểm thay thế.'
    return {'PSNR': psnr, 'SSIM': ssim, 'LPIPS': perceptual, 'DeltaE00': delta_e}, status


def comparison_metrics(reference, comparison, evaluator=evaluate_quality):
    """Reference metrics for two real branches; never use input as its own GT."""
    from demo.restoration_workflow import read_rgb

    from demo.metrics_backend import available as metrics_available, lpips_reference_files

    batch_lpips = reference is not None and evaluator is evaluate_quality and metrics_available()
    results = []
    messages = []
    for key in ('baseline', 'enhanced'):
        branch = comparison[key]
        scores = dict.fromkeys(('PSNR', 'SSIM', 'LPIPS', 'DeltaE00'))
        if reference is not None:
            if batch_lpips:
                scores, message = evaluator(reference, read_rgb(branch['output']), aligned=True, compute_lpips=False)
            else:
                scores, message = evaluator(reference, read_rgb(branch['output']), aligned=True)
            if message:
                messages.append(message)
        scores['Time'] = branch['total_seconds']
        results.append(scores)
    if batch_lpips:
        try:
            values = lpips_reference_files(reference, [comparison[key]['output'] for key in ('baseline', 'enhanced')])
            for score, value in zip(results, values):
                if not np.isfinite(value):
                    raise ValueError('Invalid LPIPS result')
                score['LPIPS'] = value
        except Exception:
            for score in results:
                score['LPIPS'] = None
            messages.append('LPIPS unavailable')
    if reference is None:
        note = 'Chưa có ảnh chuẩn: chỉ so sánh hình ảnh và thời gian; chưa kết luận cải thiện chất lượng.'
    else:
        note = 'So với ảnh chuẩn đã cung cấp; thời gian gồm phục hồi và tô màu.'
        if messages:
            note += ' LPIPS chưa khả dụng.'
    names = ('PSNR ↑ (dB)', 'SSIM ↑', 'LPIPS ↓', 'ΔE00 ↓', 'Time (s)')
    keys = ('PSNR', 'SSIM', 'LPIPS', 'DeltaE00', 'Time')
    def value(item):
        return 'N/A' if item is None else '∞' if np.isinf(item) else f'{item:.3f}'
    table = [[label, value(results[0][key]), value(results[1][key])] for label, key in zip(names, keys)]
    return table, comparison_plot(results, reference is not None), note


def comparison_plot(results, has_reference):
    from matplotlib.figure import Figure

    metrics = [('PSNR', 'dB'), ('SSIM', 'Score'), ('LPIPS', 'Distance'), ('Time', 'Seconds')]
    if not has_reference:
        metrics = [('Time', 'Seconds')]
    figure = Figure(figsize=(9, 3.2), layout='constrained')
    axes = np.atleast_1d(figure.subplots(1, len(metrics)))
    for axis, (key, unit) in zip(axes, metrics):
        values = [result[key] for result in results]
        axis.bar([0,1], [item if item is not None and np.isfinite(item) else 0 for item in values],
                 color=['#94a3b8','#3578b5'], width=0.6)
        for x, item in enumerate(values):
            label = 'N/A' if item is None else '∞' if np.isinf(item) else f'{item:.3f}'
            height = item if item is not None and np.isfinite(item) else 0
            axis.annotate(label, (x,height), xytext=(0,4), textcoords='offset points', ha='center', fontsize=9)
        axis.set_title(key)
        axis.set_ylabel(unit)
        axis.set_xticks([0,1], ['DDColor','Restored\n+ DDColor'], fontsize=8)
        axis.grid(axis='y', alpha=0.15)
        axis.margins(y=0.25)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    return figure
