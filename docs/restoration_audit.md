# Restoration audit — 2026-11-02

This audit was written before changing the restoration pipeline. The original
implementation is preserved in `reports/restoration/baseline_v2.py` for reproducible
comparisons. No existing Conda packages are installed, removed or upgraded.

## Findings in the original implementation

| Component | Evidence in code | Consequence |
|---|---|---|
| Source routing | `colorize` prefers `restored_path`, then straightened input, then upload; RGB↔BGR conversion is explicit | No original-image overwrite was found. A test verifies the exact saved restored pixels reach the model. |
| Scratch detection | 7 px median residual, 9 px top/black-hat; thresholds 25–65; Hough line-length proportional to image dimensions; ≥65% line support | Rules favor thin straight high-contrast lines. Faint, curved, wide or textured damage can be missed. Hair/highlights can be false positives. Pixel-scale kernels are not resolution invariant. |
| Dust | 2–36 px connected regions, width/height ≤9, high residual cutoff | Larger defects are deliberately excluded; isolated grain may be confused with dust. |
| Mask | 3×3 closing and dilation; abort at >12% coverage | Coverage is not accuracy. No precision/recall or user override was previously available. An empty mask silently skips repair. |
| Inpainting | `cv2.INPAINT_TELEA`, radius 3 | Local interpolation works for narrow defects, not semantic reconstruction of missing scene content. It cannot repair an undetected defect. |
| Denoise | Fixed NLM h=4×one global strength, L channel only | Noise levels differ; increasing strength can erase texture. Chromatic noise is not addressed by the old method. |
| Contrast | Global 1st–99th percentile stretch blended on every nonconstant image | Can change a correctly exposed image, hence reduce reference metrics despite a more dramatic appearance. |
| Sharpen | Unconditional unsharp mask | Can sharpen remaining grain, edges of unrepaired scratches and halos. It cannot recover missing information. |
| DDColor | Original input L is concatenated with predicted ab in `ddcolor/pipeline.py` | DDColor predicts chroma; it does not remove defects in the original luminance. Restoration must happen first. |
| Evaluation | Input was its own reference: ∞ PSNR, SSIM=1, LPIPS=0 | This baseline is not evidence for restoration quality or color accuracy. A corresponding clean reference is required. |
| LPIPS environment | `find_spec('lpips')` is false in the running `ddcolor` Conda environment | Earlier real LPIPS tests used separate temporary dependencies; the actual demo cannot currently calculate LPIPS there. Never substitute zero. |
| Tests | Mostly synthetic lines/flat fields; one generated photo checked a local scratch residual | These tests establish behavior, not broad photographic restoration quality. No ablation or paired benchmark existed. |

The original ordering (detect → repair → denoise → contrast → sharpen → DDColor)
is reasonable for repairing thin defects before sharpening. The main problems are
mask quality, local-only filling, fixed strengths, and insufficient evaluation,
not an image-routing overwrite.

## Repository review and selection

* **DDColor:** multi-scale features with color/image decoders; retain the existing
  local pretrained checkpoint and inference implementation. Do not change its
  environment or weights. [Official repository](https://github.com/piddnad/DDColor).
* **Microsoft:** its detector is an anti-aliased U-Net, depth 4, width factor 6,
  one-channel input/output, sigmoid probability and threshold 0.4. Official code
  uses grayscale normalized to [-1,1]. Reuse only this detector and MIT architecture
  files, not the full latent-translation/face-enhancement stack. The official
  2,036,400,762-byte ZIP contains a 451,663,663-byte scratch checkpoint (411,699,862
  compressed); download that member alone with HTTP ranges and verify ZIP CRC.
  CPU inference avoids sharing 4 GB VRAM with DDColor. Bounded resizing may lose
  very fine defects; the threshold and mask growth must remain adjustable.
  [Detection](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/blob/master/Global/detection.py),
  [architecture](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/blob/master/Global/detection_models/networks.py),
  [dependencies](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/blob/master/requirements.txt).
* **IOPaint/LaMa:** reuse the 205,669,692-byte TorchScript model from the URL
  published by IOPaint; verify MD5 `e3aa4aaa15225a33ec84f9f4bc47e500` and record
  SHA256. Input RGB [0,1], binary mask, padding to a multiple of 8. Run on CPU,
  initially capped at 512 pixels on the long side, then composite only masked
  original-resolution pixels. Larger missing regions remain ambiguous. The full
  IOPaint package pins Gradio 4.21.0, Pillow 9.5.0 and a diffusion stack, so do not
  install it over the working Gradio 4.44.1 environment. The repository is archived.
  [Adapter](https://github.com/Sanster/IOPaint/blob/main/iopaint/model/lama.py),
  [resize/compositing](https://github.com/Sanster/IOPaint/blob/main/iopaint/model/base.py),
  [dependencies](https://github.com/Sanster/IOPaint/blob/main/requirements.txt).
* **CodeFormer:** codebook/transformer face restoration with an adjustable
  quality/fidelity tradeoff; optional detectors, alignment, parsing and background
  upsampling add complexity. Its own BasicSR package name conflicts with the
  repository's vendored BasicSR. Do not install or enable it here. If added later,
  use a separate environment/process, no background upsampler by default, and
  explicitly warn that inferred facial features may differ from the real person.
  [Inference](https://github.com/sczhou/CodeFormer/blob/master/inference_codeformer.py),
  [installation](https://github.com/sczhou/CodeFormer#dependencies-and-installation).

## Integration rules

Workers are CPU-only subprocesses with two Torch threads, explicit size bounds,
timeouts, separate temporary files and no runtime downloading. They exit after
each request to release model memory. Their startup/loading overhead is included
in stage timings. `DDCOLOR_RESTORATION_PYTHON` can point to a separate compatible
Python environment if needed, without changing DDColor's interpreter. Missing or
failed explicitly selected models produce an error, not a silently substituted
method. `auto` may select a documented OpenCV fallback when assets are absent.

No CodeFormer model, entire third-party repository, training job, diffusion model
or video processing is installed. Model manifests preserve URLs/checksums and
source attribution; model binaries are excluded from Git.

## Evaluation plan

Create paired corruptions from a clean, known photograph: noise, dust, thin/curved
scratches, wide damage, fading, blur, and combined corruption. Measure the mask
against known spatial damage. Evaluate detection, inpainting with automatic mask,
inpainting with oracle mask, denoise, contrast and sharpen separately. Keep all
stage images and timings, plus CSV/JSON results. Report failures/degradations as
well as improvements. A single source with synthetic corruptions does not establish
generalization to real historical photos. The supplied old-photo example has no
clean reference and is therefore visual evidence only.

For end-to-end comparison, both DDColor branches must share the exact source and
the same clean color reference, inference size, checkpoint and device. Compare
**DDColor** versus **Restoration + DDColor**, including restoration time. No
reference means PSNR/SSIM/LPIPS/ΔE00 are N/A; show timings and images only. Never
claim an improvement merely from greater contrast, smoother skin, or a lower
local scratch residual.

## Implemented and measured

The production UI now uses `restore_pipeline` via `RestorationWorkflow`. `auto`
selects the downloaded Microsoft detector and LaMa. Each stage has its own
enable switch and parameters; stage images, raw mask and red overlay are saved
as lossless PNGs. A manual mask replaces automatic detection when supplied.
Noise strength is estimated from high-frequency residuals in low-gradient areas;
contrast is adjusted only for a restricted low-dynamic-range case. Sharpening is
off by default after its ablation worsened noise metrics. Suspiciously extensive
automatic masks remain visible but are not inpainted (>12%; manual masks >30%).
An explicit backend failure never silently returns the original as a successful
repair. Cache keys include source pixels, all options and mask pixels.

The optional LPIPS package was placed under `pretrain/restoration/python` with
`--no-deps`; AlexNet's verified existing cache was reused. It runs in a CPU worker,
not the active environment's site-packages. Both comparison branches share one
LPIPS model load. Installed Conda versions were checked again: NumPy 1.24.3,
Torch 2.2.0+cu118, OpenCV 4.7.0, Gradio 4.44.1.

Eight conditions generated 72 paired stage/ablation measurements; three actual
CUDA colorization cases generated another 34 measurements. All 106 rows have
PSNR/SSIM/LPIPS against corresponding clean references. Full settings, per-stage
times, known/detected masks and images are under
[`reports/restoration`](../reports/restoration/SUMMARY.md).

Key outcomes: curved-scratch restoration PSNR 25.85 → 35.02 dB; curved-scratch
colorization PSNR 21.30 → 23.14 dB. Noisy colorization has higher SSIM and lower
LPIPS but lower PSNR. Wide-damage automatic mask recall is only 1.30%; its color
output degrades slightly. With an oracle wide-damage mask, LaMa reaches 31.62 dB
versus Telea's 30.75 dB; this control does not solve automatic mask generation.
The default does not deblur. Undamaged control pixels are slightly changed by
automatic processing. These limitations are retained in the report.

The original DDColor model runs unchanged on RTX 3050: measured peak allocated
1770.7 MiB and reserved 2004 MiB. Median warmed color inference is about 0.22–0.23 s
for these 256-pixel output cases at 512 inference resolution. Enhanced totals are
10.39–11.57 s including isolated CPU restoration. The larger 1144×1375 generated
example took 17.52 s for restoration; it has no ground truth, so only images/masks
and stage timings are reported. Startup and weight loading are a substantial
cost of the deliberately isolated workers.

No full Microsoft restoration stack, IOPaint installation, CodeFormer, training,
or video pipeline was added. Further automatic wide-region detection and
real-historical-photo validation remain open work; the current evidence does not
justify claiming reliable improvement across arbitrary old photos.
