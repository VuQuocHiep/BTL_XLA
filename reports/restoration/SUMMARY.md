# Restoration benchmark — 2026-11-02

Paired synthetic corruptions of one `skimage.data.astronaut` photograph, 256 × 256 pixels; seed 20261102. Restoration is evaluated against the clean grayscale source, colorization against its clean color version. This is a controlled diagnostic, not a broad real-photo benchmark.

## Restoration only

| Case | Input PSNR | Old PSNR | New PSNR | Input SSIM | New SSIM | Input LPIPS | New LPIPS | New seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | ∞ | 22.093 | 41.245 | 1.000 | 0.999 | 0.000 | 0.003 | 10.743 |
| noise | 28.511 | 23.129 | 29.367 | 0.719 | 0.820 | 0.235 | 0.140 | 9.880 |
| curved_scratch | 25.852 | 22.916 | 35.022 | 0.972 | 0.991 | 0.064 | 0.006 | 10.205 |
| wide_damage | 19.357 | 17.586 | 19.334 | 0.929 | 0.928 | 0.099 | 0.101 | 9.344 |
| dust | 32.205 | 22.597 | 35.657 | 0.980 | 0.988 | 0.072 | 0.012 | 9.832 |
| faded | 17.578 | 21.162 | 19.655 | 0.750 | 0.798 | 0.081 | 0.048 | 9.515 |
| blur | 25.611 | 22.733 | 25.611 | 0.871 | 0.871 | 0.246 | 0.246 | 4.234 |
| mixed | 23.985 | 22.397 | 28.926 | 0.701 | 0.807 | 0.267 | 0.153 | 10.800 |

PSNR: dB (higher); SSIM: higher; LPIPS AlexNet v0.1: lower. All scores use corresponding clean references. Restoration timings include CPU subprocess startup/model load and individual filter steps, excluding saved-stage PNG export.

## DDColor on RTX 3050 Laptop 4 GB

The same existing DDColor-large checkpoint and 512-pixel inference input are used. One warm-up is excluded, then colorization time is the median of three calls. Restoration is measured once per case; its CPU time is added to the enhanced branch. Evaluation/LPIPS and model startup for DDColor itself are not included.

| Case | Method | PSNR (dB) | SSIM | LPIPS | Total seconds |
|---|---|---:|---:|---:|---:|
| noise | DDColor | 17.709 | 0.616 | 0.363 | 0.224 |
| noise | Restoration-DDColor | 16.633 | 0.699 | 0.296 | 11.573 |
| curved_scratch | DDColor | 21.299 | 0.895 | 0.161 | 0.226 |
| curved_scratch | Restoration-DDColor | 23.143 | 0.912 | 0.116 | 10.468 |
| wide_damage | DDColor | 17.667 | 0.855 | 0.202 | 0.222 |
| wide_damage | Restoration-DDColor | 17.636 | 0.853 | 0.204 | 10.393 |

Peak CUDA allocated: 1770.7 MiB; reserved: 2004.0 MiB. CPU restoration/LPIPS workers do not use CUDA.

## Interpretation and failures

* Curved scratches improve in this diagnostic; this does not establish accuracy on all real photos.
* The wide-damage automatic mask recall is only 1.30%. A learned inpainting model cannot repair pixels absent from its mask.
* With the **oracle** wide-damage mask, Telea versus LaMa PSNR is 30.75 versus 31.62 dB. These are diagnostic upper-control results, not automatic detection results.
* In noisy colorization, SSIM/LPIPS improve while PSNR worsens. Do not call this an unconditional improvement in color accuracy.
* Wide-damage colorization degrades slightly. Use the manual mask override or disable restoration for this case.
* Default sharpening is disabled after an ablation showed it amplified remaining noise. Blur alone is not repaired by the default pipeline.
* The clean control is slightly changed by automatic detection/denoise; the unmodified input remains the perfect reference. Use stage switches for undamaged images.
* The generated real-photo example has no clean reference, so no quality scores are assigned to it. Some damage can remain.

## Artifacts

* [Paired rows and stage timings](paired-final/results.csv)
* [Paired JSON, options and notes](paired-final/results.json)
* [CUDA comparison rows](color-comparison/results.csv)
* [CUDA timings, repeat measurements and peak memory](color-comparison/results.json)
* [Real-photo before/after](real-photo/comparison.png)
* [Real-photo mask](real-photo/mask.png)
* [Real-photo stage times/settings](real-photo/metadata.json)
* [Audit and repository selection](../../docs/restoration_audit.md)

![End-to-end comparison](color-comparison/summary.png)
