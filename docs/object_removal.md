# Xóa vật thể

Chức năng nằm trong tab **Xóa vật thể** của `python demo/gradio_app.py`.

1. **Tải ảnh** hoặc **Dùng kết quả DDColor**. Nếu đã chạy so sánh hai nhánh, lấy kết quả Phục hồi + DDColor.
2. Click vật thể trên ảnh bên trái; dùng điểm **+ / −** để sửa mask. Click chưa xóa ảnh.
3. Điều chỉnh **Mở rộng vùng xóa** nếu viền vật thể còn sót. Overlay bên phải hiển thị chính xác mask đã mở rộng. Thay đổi mức mở rộng sẽ yêu cầu xác nhận lại.
4. **Xác nhận vùng** → chọn phương pháp → **Xóa vật thể**.
5. Xem thanh trượt trước/sau, **Tải ảnh**, **Hoàn tác** hoặc **Đặt lại**. Click + để xóa tiếp vật thể khác.

## Phương pháp

- **OpenCV Telea** (mặc định): nhẹ, chạy CPU, phù hợp vật nhỏ và nền đơn giản.
- **OpenCV Navier–Stokes**: phương án OpenCV thứ hai để thử khi Telea tạo vệt không đẹp.
- **LaMa CPU (đã có)**: chỉ xuất hiện khi checkpoint phục hồi đã có trên máy. Tái sử dụng `demo.restoration_backends.run_model` và CPU worker hiện tại; không tải model mới. Bán kính vá OpenCV không áp dụng cho LaMa.

Sử dụng `cv2.inpaint`, `cv2.INPAINT_TELEA`, `cv2.INPAINT_NS`, `cv2.dilate` và `cv2.getStructuringElement` đã có trong OpenCV 4.7.0. Không clone/build OpenCV, không cài thêm dependency, không nâng cấp môi trường. Tham khảo [repo OpenCV](https://github.com/opencv/opencv) và [hướng dẫn inpainting chính thức](https://docs.opencv.org/4.x/df/d3d/tutorial_py_inpainting.html).

Xóa vật thể và đổi màu dùng chung một MobileSAM lazily loaded và cache embedding có lock. Mỗi tab có trạng thái/undo riêng; dùng chung hàng đợi với DDColor để hạn chế tranh bộ nhớ. Không chạy lại DDColor khi xóa.

`demo/object_removal.py` chứa xử lý mask, backend và lịch sử chỉnh sửa. `demo/recoloring_ui.py` dùng chung cấu trúc giao diện cho hai tab, với prefix API `remove_*` / `recolor_*`. `demo/gradio_app.py` gắn tab và CSS 360 px giữ toàn bộ ảnh trong khung.

Chỉ ghi pixel trong **mask sau mở rộng**; ngoài vùng này giữ nguyên từng byte. Nền thật phía sau vật thể không thể khôi phục chắc chắn từ một ảnh: kết quả là vùng vá ước lượng. Vật lớn, bóng đổ, tóc/lông và nền phức tạp có thể còn viền, vệt nhòe hoặc kết cấu không tự nhiên; tăng mức mở rộng, bổ sung điểm chọn hoặc thử LaMa. Không có cam kết xóa hoàn hảo trên mọi ảnh.

## Kiểm thử

```powershell
conda activate ddcolor
python -m unittest discover -s tests
python scripts/benchmark_removal.py
# App phải đang chạy trên 7860:
python scripts/smoke_recoloring_api.py --remove
python scripts/smoke_recoloring_api.py
```

Xem `reports/removal/REPORT.md`, số đo `results.json`, `api-smoke.json`, ảnh trước/mask/sau và screenshot giao diện trong cùng thư mục.
