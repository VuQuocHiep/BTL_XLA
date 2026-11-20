# Kiểm thử xóa vật thể — 2026-11-03

- Suite: **57 bài, 54 đạt, 3 skipped** (các bài pretrained phục hồi/LPIPS cần bật riêng). Sáu bài mới kiểm tra OpenCV thật, mặt nạ mở rộng, pixel ngoài mask, backend LaMa qua mock, yêu cầu xác nhận, undo/reset, lỗi backend và dùng chung MobileSAM.
- Benchmark MobileSAM thật trên hai ảnh upstream MobileSAM: `MobileSAMv2/test_images/2.jpg` (xóa số 1 nhỏ), `app/assets/picture4.jpg` (xóa chó trên cỏ).
- Mỗi ảnh chạy cả **Telea, Navier–Stokes và LaMa CPU đã có**, tổng sáu kết quả. Cả sáu có **0 pixel ngoài mask mở rộng thay đổi**. Số đo và kích thước gốc ở `results.json`.
- OpenCV xử lý khoảng **0.45–0.73 giây** trong lần đo; LaMa CPU khoảng **8.3–17.9 giây**, gồm nạp checkpoint trong worker. Không coi đây là bảo đảm tốc độ trên mọi ảnh.
- HTTP smoke test trên tab tích hợp: upload, chọn điểm, xác nhận, xóa, tải PNG, undo và reset. `api-smoke.json` ghi kết quả; `api-after.png` là PNG tải qua endpoint của app.
- Đã kiểm tra lại HTTP smoke test tab đổi màu để phát hiện hồi quy sau khi chia sẻ giao diện/MobileSAM.
- Screenshot `ui-verified.png` ghi lại thao tác chọn và xóa vật thể trong trình duyệt.

Ảnh `*-before.png`, `*-mask.png`, `*-selection.png`, `*-telea-after.png`, `*-ns-after.png`, `*-lama-after.png` được lưu tại độ phân giải gốc. PNG bị gitignore của dự án bỏ qua nhưng có trên máy.

Quan sát: nền trắng và vật nhỏ phù hợp với OpenCV; vùng chó lớn trên cỏ vẫn có thể lộ vệt/viền, kể cả LaMa. Không dùng chỉ số “ngoài mask không đổi” để suy ra chất lượng tái tạo nền. Chưa có ground truth nền thật để đo chất lượng xóa tuyệt đối.

Không cài thư viện, không đổi CUDA/PyTorch/Gradio, không tải checkpoint mới. Tái sử dụng MobileSAM, OpenCV 4.7.0 và checkpoint LaMa của chức năng phục hồi.
