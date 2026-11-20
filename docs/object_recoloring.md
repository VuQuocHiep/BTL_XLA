# Đổi màu vật thể

Chạy trong giao diện DDColor hiện có:

```powershell
conda activate ddcolor
Set-Location D:\DDColor
python demo/gradio_app.py
```

Mở `http://127.0.0.1:7860`, chọn tab **Đổi màu vật thể**:

1. **Tải ảnh**, hoặc chạy DDColor rồi chọn **Dùng kết quả DDColor**. Khi so sánh hai nhánh, dùng kết quả **Phục hồi + DDColor**.
2. Click **ảnh bên trái** để chọn vật thể. Các tọa độ do Gradio 4.44.1 trả về đã ở kích thước ảnh thật, bao gồm bù khoảng trống do `object-fit: contain`. Không nhân thêm tỷ lệ CSS. Ảnh có EXIF được xoay đúng trước khi chọn.
3. Chọn **+ Thêm vùng** hoặc **− Loại vùng** rồi click để sửa. Ảnh bên phải hiển thị overlay xanh trong suốt và các điểm xanh/đỏ. Click chỉ phân đoạn, chưa đổi màu.
4. **Xác nhận vùng**, chọn màu, cường độ 0–100%, độ mềm cạnh, rồi **Đổi màu**.
5. Thay màu/cường độ và bấm **Đổi màu** tiếp để tính từ ảnh trước lần chỉnh vật thể này, tránh tích lũy màu. Không chạy lại MobileSAM hoặc DDColor.
6. Click + sau khi đổi màu để bắt đầu vật thể tiếp theo. **Hoàn tác** quay lại lần đổi màu trước; **Đặt lại** về ảnh được tải vào tab. **Xóa lựa chọn** chỉ xóa mask/điểm.
7. Thanh trượt đối chiếu ảnh lúc tải vào tab với kết quả hiện tại. Nút **Tải ảnh** ở hàng thao tác tải PNG đầy đủ độ phân giải.

## Mô hình và môi trường

Đã kiểm tra Conda `C:\Users\ximen\miniconda3\envs\ddcolor\python.exe`:

| Thành phần | Phiên bản |
|---|---|
| Python | 3.9.25 |
| PyTorch | 2.2.0+cu118 |
| torchvision | 0.17.0+cu118 |
| Gradio | 4.44.1 |
| gradio_imageslider | 0.0.17 |
| pydantic | 2.10.6 |
| huggingface_hub | 0.36.2 |
| OpenCV | 4.7.0.72 |
| NumPy | 1.24.3 |

Đã bổ sung **timm 0.6.13** bằng `--no-deps --target third_party/mobilesam_runtime`. Không cài vào Conda, không nâng cấp dependency hiện có, CUDA hoặc PyTorch. MobileSAM được dùng trực tiếp từ checkout chính thức, không cần `pip install`.

- Nguồn: https://github.com/ChaoningZhang/MobileSAM
- Commit đã kiểm tra: `f706ad9c4eb7f219c00d9050e46328518ffb65d2`.
- License: xem `third_party/MobileSAM/LICENSE` và copyright trong các file upstream.
- Mô hình: **vit_t / Tiny-ViT**, checkpoint `pretrain/mobile_sam.pt`, 40,728,226 byte.
- SHA256: `6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f`.

Các thư mục tải về và checkpoint nằm trong `.gitignore`. Khôi phục ở một checkout mới bằng các lệnh sau (không cần chạy lại trên máy này):

```powershell
git clone https://github.com/ChaoningZhang/MobileSAM.git third_party/MobileSAM
git -C third_party/MobileSAM checkout f706ad9c4eb7f219c00d9050e46328518ffb65d2
python -m pip install --no-deps --target third_party/mobilesam_runtime timm==0.6.13
Copy-Item third_party/MobileSAM/weights/mobile_sam.pt pretrain/mobile_sam.pt
Get-FileHash pretrain/mobile_sam.pt -Algorithm SHA256
```

MobileSAM mặc định dùng **CPU**, giữ GPU 4 GB cho DDColor. Đổi thiết bị trước khi chạy app nếu cần:

```powershell
$env:DDCOLOR_SAM_DEVICE = 'cuda'  # cpu (mặc định), cuda, auto
python demo/gradio_app.py
```

`auto` chỉ chọn CUDA khi còn hơn 2 GiB trống tại thời điểm nạp MobileSAM. Thiếu CUDA hoặc CUDA OOM sẽ chuyển MobileSAM về CPU; model chỉ được nạp một lần. Cache giữ embedding của một ảnh, tự thay khi ảnh hoặc phiên khác sử dụng ảnh khác; lock bảo vệ predictor dùng chung. Các thao tác dùng cùng hàng đợi với DDColor/phục hồi để tránh chạy mô hình đồng thời. Undo giữ tối đa khoảng 128 MiB lịch sử, tối thiểu một ảnh gần nhất.

## Các file

| File | Vai trò |
|---|---|
| `demo/object_segmentation.py` | Nạp MobileSAM lazily, embedding cache, chọn mask theo điểm/score, khóa và CPU fallback |
| `demo/object_recoloring.py` | Tọa độ, mask, LAB, feather vào trong, trạng thái phiên, undo/reset |
| `demo/recoloring_ui.py` | Tab và callback Gradio; upload riêng và ảnh click ở chế độ chỉ xem để tương thích Gradio 4.44.1 |
| `demo/gradio_app.py` | Gắn tab, dùng chung `run_state` và hàng đợi hiện tại |
| `tests/test_object_recoloring.py` | Kiểm thử pixel, độ sáng, điểm, lịch sử, cache, OOM và cấu hình UI |
| `scripts/benchmark_recoloring.py` | Inference thật trên năm tình huống, tùy chọn giữ DDColor trên GPU |
| `scripts/smoke_recoloring_api.py` | Kiểm thử callback của app đang chạy qua HTTP, gồm `SelectData`, PNG, undo/reset |

LAB giữ kênh L và một phần biến thiên chroma, giữ bóng và kết cấu. Chuyển lại RGB có thể gây clipping với màu bão hòa; màu hiển thị phụ thuộc độ sáng ban đầu, không nhất thiết bằng RGB của color picker. Feather dùng distance transform **vào trong mask**. Chỉ gán pixel trong mask vào ảnh copy, nên ngoài mask giữ nguyên từng byte, kể cả sau khi lưu PNG.

MobileSAM có thể chọn cả người hoặc một bộ phận xe khi click mơ hồ. Cần kiểm tra overlay và thêm điểm âm vào mặt/nền/kính/bánh xe, thêm điểm dương ở phần còn thiếu trước khi xác nhận. Các vùng rời không chứa điểm + được loại khỏi mask để tránh đổi màu các đốm nền không liên quan; nếu vật thể có nhiều phần rời, click + vào từng phần cần giữ. Điểm +/− là prompt cho mô hình; không phải cam kết ranh giới chính xác trên mọi ảnh. Không có ground truth thủ công nên báo cáo không cung cấp IoU hoặc tuyên bố đạt độ chính xác tuyệt đối.

## Kiểm thử và minh chứng

```powershell
python -m unittest discover -s tests -v
python scripts/benchmark_recoloring.py --ddcolor
# Khi app đang chạy ở 7860:
python scripts/smoke_recoloring_api.py
```

Xem `reports/recoloring/REPORT.md`, `results-ddcolor.json`, `api-smoke.json`, ảnh `*-before.png`, `*-mask.png`, `*-selection.png`, `*-after.png` và `ui-verified.png`.
