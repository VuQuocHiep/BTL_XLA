# Kiểm chứng đổi màu vật thể — 2026-11-03

## Kết quả

- Toàn bộ suite: **51 tests, 48 đạt, 3 skipped**. Các bài skipped là kiểm thử pretrained phục hồi/LPIPS cần bật riêng; không phải bài kiểm thử recoloring.
- **12 bài kiểm thử mới**: ánh xạ tọa độ, feather vào trong, ngoài mask giữ nguyên, cường độ 0%, độ sáng/kết cấu LAB, nhiều điểm +/−, xác nhận bị vô hiệu hóa sau khi sửa mask, chỉnh màu không inference lại, undo/reset, cô lập phiên, cache embedding, CUDA OOM mô phỏng, ưu tiên mask phù hợp điểm, loại vùng rời không có điểm + và cấu hình click/upload của Gradio.
- Inference thật trên 5 ảnh; DDColor large được giữ trên GPU, MobileSAM Tiny-ViT trên CPU. Trường hợp áo được chạy qua DDColor trước khi phân đoạn.
- **0 pixel ngoài mask thay đổi** ở cả 5 ảnh. Ảnh sau và mask ở kích thước gốc, lưu PNG.
- Bộ nhớ PyTorch GPU cấp phát khoảng **966 MiB**, đỉnh **1,768 MiB** trong tiến trình benchmark. Đây là số liệu allocator, không phải tổng bộ nhớ toàn hệ thống theo nvidia-smi.
- Cold start MobileSAM khoảng **6.51 s** trong lần chạy cuối; tạo embedding mới khoảng **1.20–2.52 s**, điểm bổ sung khoảng **0.07–0.24 s**, đổi màu khoảng **0.05–0.74 s**. Benchmark giới hạn PyTorch ở 4 luồng CPU; app dùng thiết lập luồng hiện có, thời gian thực tế có thể khác.
- HTTP smoke test trên bản ứng dụng tích hợp: upload → chọn điểm với `SelectData` → xác nhận → đổi màu → tải PNG → hoàn tác → đặt lại đều đạt. Xem `api-smoke.json`; ảnh tải thực tế: `api-after.png`.
- Đã thao tác trực tiếp trong trình duyệt: tải ảnh chó, click lên thân, xem overlay, xác nhận, đổi màu đỏ và thấy slider trước/sau. `ui-verified.png` ghi lại giao diện sau đổi màu.

## Năm tình huống

| Mã | Ảnh / vật thể | Điểm cuối | Nhận xét quan sát |
|---|---|---:|---|
| shirt | `skimage.data.astronaut`, áo phi hành gia, qua DDColor | 2 dương, 2 âm | Chọn áo; dùng điểm âm để loại mặt và mũ cầm tay |
| car | MobileSAM `app/assets/picture3.jpg` | 6 dương, 5 âm | Phần thân xe; nhiều điểm để phân biệt các panel, kính và bánh xe |
| nearby-objects | MobileSAM `app/assets/picture5.jpg`, hai gấu sát nhau | 1 dương, 1 âm | Chọn gấu bên phải, loại gấu bên trái |
| similar-background | MobileSAM `app/assets/picture4.jpg`, chó nâu trên cỏ vàng/nâu | 1 dương, 1 âm | Chọn chó, loại nền gần màu |
| small-object | MobileSAM `MobileSAMv2/test_images/2.jpg`, số 1 | 1 dương, 1 âm | Vật thể ~0.23% diện tích ảnh, không chọn nền/chữ sát bên |

Điểm cụ thể, kích thước, thời gian và số pixel nằm trong `results-ddcolor.json`. Tập ảnh nhỏ không có ground truth nên không đo IoU; ranh giới có thể còn sai số, cần kiểm tra overlay trước khi xác nhận. Không khẳng định chỉ một click luôn đủ trên ảnh bất kỳ. Ngoài mask giữ nguyên tuyệt đối; “ngoài vật thể” phụ thuộc độ đúng của mask người dùng xác nhận.

Kênh L được giữ trong LAB; sai khác độ sáng đầu ra RGB trung bình trên mask ở lần chạy cuối khoảng 0.21–2.18 đơn vị L, do feather, clipping gamut và lượng tử hóa. Bài kiểm thử dùng màu trung tính kiểm tra tương quan kết cấu >0.99 và sai khác L trung bình <0.3.

## Phạm vi thiết bị đã kiểm tra

Máy RTX 3050 Laptop 4 GB; PyTorch 2.2.0+cu118, Python 3.9.25, Gradio 4.44.1. Đã xác nhận cấu hình mặc định DDColor GPU + MobileSAM CPU hoạt động cùng nhau. Nhánh CUDA OOM được kiểm tra bằng lỗi mô phỏng, không cố ý làm cạn VRAM trên máy người dùng. Chế độ MobileSAM CUDA/auto có triển khai nhưng chưa đo benchmark riêng.

Không thay đổi phiên bản PyTorch, CUDA, Gradio, gradio_imageslider, pydantic, huggingface_hub. Chỉ bổ sung timm 0.6.13 trong thư mục dự án và checkout/checkpoint MobileSAM chính thức.

## File minh chứng

- `*-before.png`, `*-selection.png`, `*-mask.png`, `*-after.png`: năm bộ ảnh tại độ phân giải gốc.
- `results-ddcolor.json`: số đo thật và hash checkpoint.
- `api-smoke.json`, `api-after.png`: kiểm thử ứng dụng đang chạy.
- `ui-verified.png`: screenshot giao diện sau thao tác thật.
- `comparison.png`: bảng ảnh trước / mask / sau.

Chạy lại theo `docs/object_recoloring.md`. Các ảnh PNG bị `.gitignore` của dự án bỏ qua nhưng đã được lưu trong thư mục báo cáo trên máy.
