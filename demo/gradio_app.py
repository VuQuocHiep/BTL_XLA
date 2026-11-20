import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
import torch
from gradio_imageslider import ImageSlider

from ddcolor import DDColor, ColorizationPipeline, build_ddcolor_model
from demo.document_scanner import straighten_document
from demo.photo_restoration import RestorationOptions
from demo.quality_metrics import comparison_metrics, validate_pair
from demo.restoration_workflow import RestorationWorkflow, stage_gallery
from demo.recoloring_ui import build_recoloring_tab, build_removal_tab

model_path = str(ROOT / 'modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt')
_model = build_ddcolor_model(DDColor, model_path=model_path, input_size=512, model_size='large',
                            decoder_type='MultiScaleColorDecoder',
                            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
colorizer = ColorizationPipeline(_model, input_size=512)
_output_dir = tempfile.TemporaryDirectory(prefix='ddcolor-')
workflow = RestorationWorkflow(_output_dir.name, colorizer)
queue = dict(concurrency_id='image-pipeline', concurrency_limit=1)

OPTION_FIELDS = ['detector', 'repair_backend', 'detect', 'repair', 'denoise', 'contrast', 'sharpen',
                 'detection_threshold', 'mask_grow', 'denoise_h', 'contrast_amount', 'sharpen_amount',
                 'max_model_side']


def options_from_values(values):
    values = dict(zip(OPTION_FIELDS, values))
    for key in ('mask_grow', 'max_model_side'):
        if key in values:
            values[key] = int(values[key])
    return RestorationOptions(**values).validate()


def source_image(original, straightened):
    image = straightened if straightened is not None else original
    if image is None:
        raise ValueError('Vui lòng tải ảnh lên trước khi xử lý.')
    return image


def render(run=None, restored=None, table=None, plot=None, note=''):
    preview, label = None, 'Ảnh đầu vào / kết quả màu'
    if run:
        if 'baseline' in run:
            preview = (run['baseline']['output'], run['enhanced']['output'])
            label = 'DDColor gốc / Phục hồi + DDColor'
        else:
            preview = (run['single']['source'], run['single']['output'])
        restored = restored or run.get('restored')
    restoration_pair = (restored['source'], restored['restored']) if restored else None
    mask = restored['mask'] if restored else None
    return (gr.update(value=preview, label=label), restoration_pair, stage_gallery(restored), mask,
            restored, run, table or [], plot, note)


def restore_click(original, straightened, manual_mask, *settings):
    try:
        state = workflow.prepare(source_image(original, straightened), options_from_values(settings), manual_mask)
        note = f"Đã phục hồi trong {state['timings']['Total']:.2f}s; mask {state['coverage']*100:.2f}%."
        warnings = [item for item in state['notes'] if item.startswith(('Bỏ qua', 'Không phát hiện'))]
        if warnings:
            note += ' ' + ' '.join(warnings)
        return render(restored=state, note=note)
    except ValueError as error:
        gr.Warning(str(error))
        return render(note=str(error))


def colorize_click(original, straightened, restored, use_restoration, manual_mask, *settings):
    try:
        result = workflow.colorize(source_image(original, straightened), options_from_values(settings),
                                   manual_mask, restored, use_restoration)
        return render(run={'single': result, 'restored': result['restored']},
                      note=f"{result['method']}: {result['total_seconds']:.2f}s (gồm thời gian phục hồi nếu có).")
    except ValueError as error:
        gr.Warning(str(error))
        return render(note=str(error))


def compare_click(original, straightened, restored, manual_mask, reference, *settings):
    try:
        source = source_image(original, straightened)
        if reference is not None:
            validate_pair(reference, source, True)
        result = workflow.compare(source, options_from_values(settings), manual_mask, restored)
        table, plot, note = comparison_metrics(reference, result)
        return render(run=result, table=table, plot=plot, note=note)
    except ValueError as error:
        gr.Warning(str(error))
        return render(note=str(error))


def straighten_click(original):
    try:
        return straighten_document(original), *render()
    except ValueError as error:
        gr.Warning(str(error))
        return None, *render(note=str(error))


UI_CSS = """
.gradio-container { max-width: 1440px !important; margin: auto; }
#app-header { padding: 12px 4px 18px; }
#app-header h1 { margin-bottom: 6px; }
#app-header p { color: var(--body-text-color-subdued); }
#workspace { align-items: flex-start; gap: 24px; }
#controls-panel { background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
    border-radius: 16px; padding: 18px; gap: 14px; }
#controls-panel .panel-title { margin-bottom: -4px; }
#controls-panel .panel-title h3 { font-size: 16px; }
#result-panel { min-width: 0; gap: 16px; }
#result-tabs { border-radius: 16px; overflow: hidden; }
#result-tabs > .tab-nav { gap: 4px; padding: 8px; }
#result-tabs > .tab-nav button { border-radius: 8px; padding: 10px 14px; }
.result-preview { height: clamp(240px, calc(100vh - 360px), 460px) !important; min-height: 240px; }
.result-preview .slider-wrap { height: 100%; max-height: 100%; }
.result-preview .slider-wrap > div { height: 100%; }
.result-preview img { object-fit: contain !important; }
#recolor-comparison, #remove-comparison { height: 360px !important; min-height: 360px; }
#remove-comparison .slider-wrap,
#remove-comparison .slider-wrap > div,
#recolor-comparison .slider-wrap,
#recolor-comparison .slider-wrap > div {
    width: 100% !important; height: 100% !important; max-height: none !important;
}
#recolor-comparison img, #remove-comparison img {
    display: block; width: 100% !important; height: 100% !important;
    max-height: 100% !important; object-fit: contain !important; object-position: center;
}
#run-actions { gap: 12px; }
#run-actions button { min-height: 46px; }
#status-note { border-left: 3px solid var(--color-accent); padding-left: 12px; }
@media (max-width: 800px) {
    #workspace { gap: 16px; }
    #controls-panel { min-width: min(100%, 320px) !important; }
}
"""


with gr.Blocks(title='DDColor · Phục hồi & tô màu ảnh', css=UI_CSS) as demo:
    restored_state = gr.State(None)
    run_state = gr.State(None)
    gr.Markdown(
        '# DDColor Studio\nPhục hồi ảnh cũ và tô màu. Tải ảnh ở cột trái, chọn thao tác và xem kết quả bên phải.',
        elem_id='app-header',
    )
    with gr.Row(elem_id='workspace'):
        with gr.Column(scale=1, min_width=310, elem_id='controls-panel'):
            gr.Markdown('### 01 · Ảnh đầu vào', elem_classes=['panel-title'])
            original = gr.Image(label='Ảnh đầu vào', type='numpy', image_mode='RGB', height=250)
            with gr.Accordion('Căn chỉnh ảnh chụp', open=False):
                straighten_btn = gr.Button('Làm thẳng ảnh')
                straightened = gr.Image(label='Ảnh đã làm thẳng', type='numpy', image_mode='RGB',
                                        interactive=False, height=220)

            gr.Markdown('### 02 · Tùy chỉnh xử lý', elem_classes=['panel-title'])
            use_restoration = gr.Checkbox(value=True, label='Phục hồi trước khi tô màu')
            with gr.Accordion('Các bước phục hồi', open=False):
                detect = gr.Checkbox(value=True, label='Phát hiện hư hỏng')
                repair = gr.Checkbox(value=True, label='Vá vùng hỏng')
                denoise = gr.Checkbox(value=True, label='Khử nhiễu')
                contrast = gr.Checkbox(value=True, label='Tăng tương phản khi ảnh bạc màu')
                sharpen = gr.Checkbox(value=False, label='Làm nét nhẹ')
            with gr.Accordion('Thông số nâng cao', open=False):
                detector = gr.Dropdown(['auto','microsoft','opencv'], value='auto', label='Bộ phát hiện vết xước')
                backend = gr.Dropdown(['auto','lama','telea'], value='auto', label='Phương pháp vá ảnh')
                threshold = gr.Slider(0.1,0.9,value=0.4,step=0.05,label='Ngưỡng phát hiện vết xước')
                grow = gr.Slider(0,5,value=1,step=1,label='Mở rộng vùng vá (px)')
                noise = gr.Slider(0,12,value=0,step=0.5,label='Mức khử nhiễu (0 = tự động)')
                contrast_amount = gr.Slider(0,0.5,value=0.2,step=0.05,label='Mức tăng tương phản')
                sharpen_amount = gr.Slider(0,0.5,value=0.2,step=0.05,label='Mức làm nét')
                max_side = gr.Slider(256,768,value=512,step=128,label='Cạnh tối đa của mô hình CPU (px)')
            with gr.Accordion('Vùng vá thủ công', open=False):
                manual_mask = gr.Image(label='Mask thủ công (tùy chọn, cùng kích thước nguồn)',
                                       type='numpy', image_mode='L', height=220)

        with gr.Column(scale=3, min_width=320, elem_id='result-panel'):
            gr.Markdown('### 03 · Xử lý & kết quả')
            with gr.Row(elem_id='run-actions'):
                colorize_btn = gr.Button('Tô màu DDColor', variant='primary', scale=2)
                restore_btn = gr.Button('Phục hồi ảnh cũ', scale=1)
            with gr.Tabs(elem_id='result-tabs'):
                with gr.Tab('Kết quả màu'):
                    gr.Markdown('Kéo thanh trượt để đối chiếu ảnh trước và sau khi tô màu.')
                    color_preview = ImageSlider(position=0.5, label='Ảnh đầu vào / kết quả màu',
                                                height=460, elem_classes=['result-preview'])
                build_recoloring_tab(run_state, _output_dir.name, queue)
                build_removal_tab(run_state, _output_dir.name, queue)
                with gr.Tab('Phục hồi'):
                    gr.Markdown('Xem ảnh trước và sau bước phục hồi, trước khi tô màu.')
                    restoration_preview = ImageSlider(position=0.5, label='Trước / sau phục hồi',
                                                      height=460, elem_classes=['result-preview'])
                with gr.Tab('Các giai đoạn'):
                    stages = gr.Gallery(label='Source → Mask → Repair → Denoise → Contrast → Sharpen',
                                        columns=3, height=420, object_fit='contain')
                    with gr.Accordion('Xem vùng hư hỏng được phát hiện', open=False):
                        mask_preview = gr.Image(label='Mask: trắng = vùng cần vá', type='filepath',
                                                interactive=False, height=360)
                with gr.Tab('Đánh giá & so sánh'):
                    gr.Markdown('So sánh DDColor gốc với Phục hồi + DDColor trên cùng ảnh đầu vào.')
                    with gr.Accordion('Ảnh màu chuẩn (tùy chọn)', open=False):
                        reference = gr.Image(label='Ảnh màu chuẩn đã căn chỉnh (tùy chọn)',
                                             type='numpy', image_mode='RGB', height=240)
                    compare_btn = gr.Button('So sánh DDColor gốc')
                    gr.Markdown('Ảnh đối chiếu hiển thị trong tab **Kết quả màu** sau khi so sánh.')
                    results = gr.Dataframe(headers=['Metric','DDColor','Restoration + DDColor'],
                                           datatype=['str']*3, interactive=False, wrap=True)
                    plot = gr.Plot(show_label=False)
            comment = gr.Markdown(elem_id='status-note')
    settings = [detector,backend,detect,repair,denoise,contrast,sharpen,threshold,grow,noise,
                contrast_amount,sharpen_amount,max_side]
    views = [color_preview,restoration_preview,stages,mask_preview,restored_state,run_state,results,plot,comment]
    queue = dict(concurrency_id='image-pipeline', concurrency_limit=1)
    restore_btn.click(restore_click, [original,straightened,manual_mask,*settings], views, **queue)
    colorize_btn.click(colorize_click, [original,straightened,restored_state,use_restoration,manual_mask,*settings], views, **queue)
    compare_btn.click(compare_click, [original,straightened,restored_state,manual_mask,reference,*settings], views, **queue)
    straighten_btn.click(straighten_click, original, [straightened,*views], **queue)
    original.change(lambda: (None,*render()), outputs=[straightened,*views], **queue)
    for control in [use_restoration,manual_mask,*settings]:
        control.change(lambda: render(), outputs=views, **queue)
    reference.change(lambda: ([],None,''), outputs=[results,plot,comment], **queue)

if __name__ == '__main__':
    demo.launch()
