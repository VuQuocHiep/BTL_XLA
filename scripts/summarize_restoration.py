"""Produce a reproducible human-readable summary of saved benchmark outputs."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    root=ROOT/'reports/restoration'
    paired=json.loads((root/'paired-final/results.json').read_text(encoding='utf-8'))
    colored=json.loads((root/'color-comparison/results.json').read_text(encoding='utf-8'))
    def find(data,case,method,stage):
        return next(row for row in data['rows'] if (row['case'],row['method'],row['stage'])==(case,method,stage))
    def number(value):
        if value is None:return 'N/A'
        return f'{value:.3f}' if value!=float('inf') else '∞'
    lines=['# Restoration benchmark — 2026-11-02','',
           'Paired synthetic corruptions of one `skimage.data.astronaut` photograph, 256 × 256 pixels; '
           'seed 20261102. Restoration is evaluated against the clean grayscale source, '
           'colorization against its clean color version. This is a controlled diagnostic, '
           'not a broad real-photo benchmark.','',
           '## Restoration only','',
           '| Case | Input PSNR | Old PSNR | New PSNR | Input SSIM | New SSIM | Input LPIPS | New LPIPS | New seconds |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for case in paired['details']:
        source=find(paired,case,'input','source');old=find(paired,case,'old','final');new=find(paired,case,'new','final')
        values=[source['psnr'],old['psnr'],new['psnr'],source['ssim'],new['ssim'],source['lpips'],new['lpips'],new['seconds']]
        lines.append('| '+case+' | '+' | '.join(number(value) for value in values)+' |')
    lines+=['','PSNR: dB (higher); SSIM: higher; LPIPS AlexNet v0.1: lower. '
            'All scores use corresponding clean references. Restoration timings include '
            'CPU subprocess startup/model load and individual filter steps, excluding saved-stage PNG export.','',
            '## DDColor on RTX 3050 Laptop 4 GB','',
            'The same existing DDColor-large checkpoint and 512-pixel inference input are used. '
            'One warm-up is excluded, then colorization time is the median of three calls. '
            'Restoration is measured once per case; its CPU time is added to the enhanced branch. '
            'Evaluation/LPIPS and model startup for DDColor itself are not included.','',
            '| Case | Method | PSNR (dB) | SSIM | LPIPS | Total seconds |',
            '|---|---|---:|---:|---:|---:|']
    cases=list(colored['details'])
    for case in cases:
        for method in ('DDColor','Restoration-DDColor'):
            row=find(colored,case,method,'color')
            lines.append('| '+case+' | '+method+' | '+' | '.join(number(row[key]) for key in ('psnr','ssim','lpips','seconds'))+' |')
    details=next(iter(colored['details'].values()))
    lines+=['',f"Peak CUDA allocated: {details['peak_cuda_allocated_mb']:.1f} MiB; "
            f"reserved: {details['peak_cuda_reserved_mb']:.1f} MiB. CPU restoration/LPIPS workers do not use CUDA.",
            '', '## Interpretation and failures','',
            '* Curved scratches improve in this diagnostic; this does not establish accuracy on all real photos.',
            '* The wide-damage automatic mask recall is only '
            f"{find(paired,'wide_damage','new','final')['mask_recall']*100:.2f}%. A learned inpainting model cannot repair pixels absent from its mask.",
            '* With the **oracle** wide-damage mask, Telea versus LaMa PSNR is '
            f"{find(paired,'wide_damage','oracle-telea','repair')['psnr']:.2f} versus "
            f"{find(paired,'wide_damage','oracle-lama','repair')['psnr']:.2f} dB. These are diagnostic upper-control results, not automatic detection results.",
            '* In noisy colorization, SSIM/LPIPS improve while PSNR worsens. Do not call this an unconditional improvement in color accuracy.',
            '* Wide-damage colorization degrades slightly. Use the manual mask override or disable restoration for this case.',
            '* Default sharpening is disabled after an ablation showed it amplified remaining noise. Blur alone is not repaired by the default pipeline.',
            '* The clean control is slightly changed by automatic detection/denoise; the unmodified input remains the perfect reference. Use stage switches for undamaged images.',
            '* The generated real-photo example has no clean reference, so no quality scores are assigned to it. Some damage can remain.',
            '', '## Artifacts','',
            '* [Paired rows and stage timings](paired-final/results.csv)',
            '* [Paired JSON, options and notes](paired-final/results.json)',
            '* [CUDA comparison rows](color-comparison/results.csv)',
            '* [CUDA timings, repeat measurements and peak memory](color-comparison/results.json)',
            '* [Real-photo before/after](real-photo/comparison.png)',
            '* [Real-photo mask](real-photo/mask.png)',
            '* [Real-photo stage times/settings](real-photo/metadata.json)',
            '* [Audit and repository selection](../../docs/restoration_audit.md)',
            '', '![End-to-end comparison](color-comparison/summary.png)','']
    (root/'SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
    from matplotlib.figure import Figure
    import numpy as np
    figure=Figure(figsize=(11,6),layout='constrained')
    for axis,(key,title) in zip(figure.subplots(2,2).flat,
                                [('psnr','PSNR (dB) ↑'),('ssim','SSIM ↑'),('lpips','LPIPS ↓'),('seconds','Total time (s)')]):
        x=np.arange(len(cases))
        for offset,method,color in [(-.18,'DDColor','#94a3b8'),(.18,'Restoration-DDColor','#3578b5')]:
            values=[find(colored,case,method,'color')[key] for case in cases]
            bars=axis.bar(x+offset,values,width=.34,label=method,color=color)
            for bar,value in zip(bars,values):
                axis.annotate(f'{value:.2f}',(bar.get_x()+bar.get_width()/2,bar.get_height()),
                              xytext=(0,3),textcoords='offset points',ha='center',fontsize=8)
        axis.set_xticks(x,[case.replace('_',' ') for case in cases],fontsize=9)
        axis.set_title(title,fontsize=11)
        axis.grid(axis='y',alpha=.15)
        axis.margins(y=.25)
        axis.spines['top'].set_visible(False);axis.spines['right'].set_visible(False)
    figure.legend(*figure.axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=2,frameon=False)
    figure.savefig(root/'color-comparison/summary.png',dpi=140)
    print('Saved',root/'SUMMARY.md')


if __name__=='__main__':main()
