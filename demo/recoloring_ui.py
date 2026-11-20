"""Compact object recoloring tab for the existing DDColor Blocks app."""
from pathlib import Path
import uuid

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageOps
from gradio_imageslider import ImageSlider

from demo.object_recoloring import RecolorSession
from demo.object_segmentation import shared_segmenter
from demo.object_removal import RemovalSession
from demo.restoration_backends import available_models
from demo.restoration_workflow import read_rgb


def build_recoloring_tab(run_state, output_dir, queue):
    return _build_object_tab(run_state, output_dir, queue)


def build_removal_tab(run_state, output_dir, queue):
    return _build_object_tab(run_state, output_dir, queue, removal=True)


def _build_object_tab(run_state, output_dir, queue, removal=False):
    segmenter = shared_segmenter()
    prefix = 'remove' if removal else 'recolor'
    session_type = RemovalSession if removal else RecolorSession
    session = gr.State(None)
    download_path = gr.State(None)
    with gr.Tab('Xóa vật thể' if removal else 'Đổi màu vật thể'):
        with gr.Row():
            upload = gr.UploadButton('Tải ảnh', file_types=['image'], type='filepath')
            use_result = gr.Button('Dùng kết quả DDColor')
        with gr.Row():
            canvas = gr.Image(label='Click chọn vật thể', type='numpy', image_mode='RGB',
                              interactive=False, format='png', height=360)
            overlay = gr.Image(label='Vùng chọn', interactive=False, format='png', height=360)
        with gr.Row():
            mode = gr.Radio(['+ Thêm vùng', '− Loại vùng'], value='+ Thêm vùng', label='Điểm chọn')
            clear = gr.Button('Xóa lựa chọn')
            confirm = gr.Button('Xác nhận vùng')
        with gr.Row():
            if removal:
                choices = [('OpenCV · Telea', 'telea'), ('OpenCV · Navier–Stokes', 'ns')]
                if available_models()['lama']:
                    choices.append(('LaMa · CPU (đã có)', 'lama'))
                color = gr.Dropdown(choices, value='telea', label='Phương pháp xóa')
                strength = gr.Slider(1, 9, value=3, step=1, label='Bán kính vá OpenCV (px)')
                feather = gr.Slider(0, 15, value=3, step=1, label='Mở rộng vùng xóa (px)')
            else:
                color = gr.ColorPicker(value='#d83030', label='Màu')
                strength = gr.Slider(0, 100, value=100, step=1, label='Cường độ (%)')
                feather = gr.Slider(0, 30, value=3, step=1, label='Mềm cạnh (px)')
        with gr.Row():
            apply = gr.Button('Xóa vật thể' if removal else 'Đổi màu', variant='primary')
            undo = gr.Button('Hoàn tác')
            reset = gr.Button('Đặt lại')
            download = gr.DownloadButton('Tải ảnh', interactive=False)
        comparison = ImageSlider(label='Trước / sau', position=.5, height=360,
                                 elem_id=prefix + '-comparison')
        note = gr.Markdown()

    outputs = [session, canvas, overlay, comparison, download_path, download, note]

    def view(state, path=None, message=''):
        if state is None:
            return None, None, None, None, None, gr.update(value=None, interactive=False), message
        return (state, state.current, state.overlay(), (state.original, state.current), path,
                gr.update(value=path, interactive=path is not None), message)

    def save(state):
        path = Path(output_dir) / (prefix + '-' + uuid.uuid4().hex + '.png')
        if not cv2.imwrite(str(path), cv2.cvtColor(state.current, cv2.COLOR_RGB2BGR)):
            raise ValueError('Không thể lưu ảnh kết quả.')
        return str(path)

    def load(image, margin=3):
        try:
            if isinstance(image, (str, Path)):
                with Image.open(image) as uploaded:
                    image = np.array(ImageOps.exif_transpose(uploaded).convert('RGB'))
            state = session_type.load(image)
            if removal:
                state.set_margin(margin)
            return view(state, save(state), 'Click + trên vật thể để chọn vùng.')
        except (ValueError, OSError) as error:
            gr.Warning(str(error))
            return (*[gr.skip() for _ in range(6)], str(error))

    def from_result(run, margin=3):
        if not run:
            raise gr.Error('Chạy DDColor trước hoặc tải ảnh trực tiếp vào vùng chọn.')
        branch = run.get('enhanced') or run.get('single')
        if not branch:
            raise gr.Error('Chưa có kết quả DDColor.')
        return load(read_rgb(branch['output']), margin)

    def require(state):
        if state is None:
            raise ValueError('Tải ảnh hoặc dùng kết quả DDColor trước.')
        return state

    def select(state, mode, path, evt: gr.SelectData):
        try:
            require(state).select(evt.index, mode.startswith('+'), segmenter)
            fallback = ' Đã chuyển MobileSAM sang CPU.' if segmenter.fell_back else ''
            return view(state, path, f'{len(state.points)} điểm; hãy xác nhận vùng.{fallback}')
        except ValueError as error:
            gr.Warning(str(error))
            return view(state, path, str(error))

    def edit(action, state, path, color=None, strength=None, feather=None):
        try:
            require(state)
            if action == 'apply':
                if removal:
                    state.remove(color, strength, feather)
                else:
                    state.apply(color, strength, feather)
                path = save(state)
                message = ('Đã xóa vật thể. Click + để chọn vật thể tiếp theo.' if removal else
                           'Đã đổi màu. Click điểm + để chọn vật thể tiếp theo.')
            else:
                getattr(state, action)()
                if action in ('undo', 'reset'):
                    path = save(state)
                message = 'Đã xác nhận vùng.' if action == 'confirm' else ''
            return view(state, path, message)
        except ValueError as error:
            gr.Warning(str(error))
            return view(state, path, str(error))

    upload.upload(load, [upload, feather] if removal else upload, outputs, api_name=prefix + '_load', **queue)
    use_result.click(from_result, [run_state, feather] if removal else run_state,
                     outputs, api_name=prefix + '_from_ddcolor', **queue)
    canvas.select(select, [session, mode, download_path], outputs, api_name=prefix + '_select', **queue)
    if removal:
        def margin_change(state, path, margin):
            if state is not None:
                state.set_margin(margin)
            return view(state, path, 'Hãy xác nhận lại vùng xóa.' if state and state.mask is not None else '')
        feather.change(margin_change, [session, download_path, feather], outputs,
                       api_name='remove_margin', **queue)
    for button, action in [(clear, 'clear'), (confirm, 'confirm'), (undo, 'undo'), (reset, 'reset')]:
        def callback(state, path, action=action):
            return edit(action, state, path)
        button.click(callback, [session, download_path], outputs, api_name=prefix + '_' + action, **queue)
    apply.click(lambda s, p, c, a, f: edit('apply', s, p, c, a, f),
                [session, download_path, color, strength, feather], outputs, api_name=prefix + '_apply', **queue)
