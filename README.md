# HỆ THỐNG PHỤC HỒI, TÔ MÀU VÀ CHỈNH SỬA ẢNH DỰA TRÊN DDCOLOR


## 1. THÔNG TIN CHUNG & THÀNH VIÊN

**Tên đề tài**: XÂY DỰNG HỆ THỐNG PHỤC HỒI, TÔ MÀU VÀ CHỈNH SỬA ẢNH DỰA TRÊN DDCOLOR

**Nhóm thực hiện**: Nhóm 14 - Nhóm lớp 12

**Danh sách thành viên**:

| STT | Họ và tên | Mã sinh viên | Phân công nhiệm vụ |
|:---:|:---|:---:|:---|
| 1 | **Vũ Quốc Hiệp** | B23DCCN295 | - Tô màu ảnh bằng DDColor. <br>- So sánh kết quả DDColor trực tiếp với kết quả phục hồi và tô màu. |
| 2 | **Lê Đức Duy** | B23DCCN239 | - Quy trình phục hồi ảnh cũ. <br>- Làm phẳng ảnh chụp. |
| 3 | **Phạm Minh** | B23DCCN561 | - Chọn vùng để tô màu. <br>- Xóa vật thể.|
---

## 2. MÔ TẢ HỆ THỐNG

Dự án xây dựng một **chuỗi xử lý ảnh số toàn diện (End-to-End Image Processing & Restoration Pipeline)** nhằm phục hồi và tái tạo chất lượng cao cho các tài liệu chụp xiên lệch, ảnh đen trắng và ảnh tư liệu lịch sử bị suy biến qua thời gian:

1. **Hiệu chỉnh hình học (Geometric Rectification)**: Tự động phát hiện 4 góc tài liệu và nắn thẳng mép phối cảnh (Deskew/Homography), chuyển đổi về góc nhìn trực diện chuẩn xác.
2. **Tái cấu trúc & Phục hồi bề mặt (Restoration)**: Loại bỏ các khuyết tật vật lý (vết xước, rách nát, bụi bẩn) thông qua mô hình học sâu kết hợp thuật toán thị giác kinh điển, đi kèm khử nhiễu thích ứng và cân bằng tương phản.
3. **Tô màu tự nhiên (Deep Colorization)**: Ứng dụng mô hình nơ-ron tích hợp **DDColor (ICCV 2023)** với backbone ConvNeXt và kiến trúc Dual Decoders (Color Decoder & Spatial Decoder) trong không gian màu CIE LAB, giữ nguyên kênh sáng $L$ và tối ưu dự đoán hai kênh sắc độ $a, b$.
4. **Tương tác & Biên tập thông minh**: Tích hợp phân đoạn đối tượng tương tác nhẹ **MobileSAM** phục vụ đổi màu hoặc xóa vật thể cục bộ chính xác, hỗ trợ công cụ vẽ/tẩy mặt nạ (Mask Editor) trực quan.

---

## 3. CHỨC NĂNG CHÍNH

Hệ thống được thiết kế theo dạng module hóa, tích hợp 5 khối chức năng chính:

| STT | Chức năng | Nguyên lý / Mô hình kỹ thuật | Đặc tính & Đầu ra |
|:---:|:---|:---|:---|
| **1** | **Làm thẳng / Làm phẳng ảnh** *(Document Scanner)* | Dò biên Canny đa tỉ lệ + Phân ngưỡng Otsu $\rightarrow$ Đa giác lồi Douglas–Peucker $\rightarrow$ Biến đổi phối cảnh Homography ($3 \times 3$). | Khắc phục góc chụp nghiêng, bảo toàn độ phân giải gốc qua phép nội suy Bicubic. |
| **2** | **Phục hồi ảnh cũ** *(Photo Restoration)* | Pipeline 5 giai đoạn: Dò hư hỏng (Microsoft Scratch Detector / Morphology) $\rightarrow$ Điền khuyết tật (LaMa FFC Inpainting / Telea / NS) $\rightarrow$ Khử nhiễu thích ứng $\rightarrow$ Cân bằng tương phản $\rightarrow$ Tăng độ nét có kiểm soát. | Làm sạch bề mặt ảnh, tái tạo vùng khuyết thiếu tự nhiên, giữ trọn cấu trúc gốc. |
| **3** | **Tô màu ảnh đen trắng** *(Deep Colorization)* | Mô hình học sâu **DDColor Large** (trích xuất đặc trưng ConvNeXt, Dual Decoders dự đoán phân bố màu trong không gian LAB). | Tái tạo màu sắc chân thực, giải quyết triệt để lỗi lem màu (color bleeding) và mất cân đối bão hòa. |
| **4** | **Đổi màu & Xóa vật thể** *(Interactive Editing)* | Phân đoạn đối tượng bằng **MobileSAM** (tương tác điểm $+/-$) $\rightarrow$ Chuyển dịch sắc độ trong không gian LAB (Đổi màu) hoặc Inpainting (Xóa vật thể). | Chỉnh sửa cục bộ chính xác theo ngữ nghĩa; hỗ trợ hoàn tác (Undo/Reset) và chỉnh sửa mask thủ công bằng bút/tẩy. |
| **5** | **Đánh giá & So sánh chất lượng** *(Quality Metrics)* | So sánh song song hai nhánh (DDColor trực tiếp vs. Phục hồi + DDColor) với bộ đo: **PSNR**, **SSIM**, **$\Delta E_{00}$** (sai khác màu chuẩn CIE) và **LPIPS** (AlexNet). | Đánh giá định lượng khách quan hiệu năng phục hồi và độ chân thực khi có ảnh gốc đối chứng. |

---

## 4. CÔNG NGHỆ SỬ DỤNG

### 4.1. Ngôn ngữ & Nền tảng cốt lõi
- **Ngôn ngữ**: Python 3.9 (64-bit)
- **Deep Learning Framework**: PyTorch 2.2.0, TorchVision 0.17.0 (tương thích cả GPU NVIDIA CUDA 11.8 và CPU)
- **Thị giác máy tính & Xử lý số**: OpenCV (`opencv-python` 4.7.0.72), NumPy 1.24.3, SciPy 1.11.4, scikit-image 0.22.0, Pillow
- **Giao diện người dùng**: Gradio 4.44.1, Gradio ImageSlider 0.0.17, Matplotlib 3.8.4

### 4.2. Các mô hình học sâu (Pre-trained Deep Learning Models)
- **DDColor** (*ICCV 2023, DAMO Academy*): Mạng nơ-ron hai bộ giải mã tối ưu hóa tái tạo màu sắc ảnh chụp.
- **MobileSAM** (*Faster Segment Anything*): Mô hình phân đoạn đối tượng tương tác siêu nhẹ chạy mượt trên CPU/GPU.
- **LaMa** (*Large Mask Inpainting, WACV 2022*): Mạng tích chập biến đổi Fourier nhanh (Fast Fourier Convolutions) cho phép vá hoàn hảo các vùng khuyết tật lớn.
- **Microsoft Scratch Detector**: Mạng nơ-ron chuyên dụng nhận diện vết xước và nếp gấp trên ảnh cổ.
- **LPIPS** (*Learned Perceptual Image Patch Similarity*): Trích xuất đặc trưng mạng AlexNet để chấm điểm tương đồng cảm nhận thị giác.

---

## 5. HƯỚNG DẪN CÀI ĐẶT

### 5.1. Yêu cầu hệ thống
- **Hệ điều hành**: Windows 10/11 hoặc Linux / macOS.
- **Môi trường Python**: Python 3.9 (64-bit) - Khuyến nghị quản lý thông qua **Conda**.
- **Phần cứng**: Tối thiểu 8 GB RAM, khuyến nghị GPU NVIDIA $\ge$ 4 GB VRAM nếu muốn tăng tốc độ xử lý.

### 5.2. Các bước cài đặt

#### Bước 1: Khởi tạo và kích hoạt môi trường Conda
Mở **Anaconda PowerShell Prompt** (hoặc PowerShell đã khởi tạo conda) và chạy:
```powershell
# Chuyển vào thư mục mã nguồn dự án
cd "<duong-dan-thu-muc-repo>"

# Khởi tạo môi trường ảo Python 3.9
conda create -n ddcolor python=3.9 -y

# Kích hoạt môi trường
conda activate ddcolor
```

#### Bước 2: Cài đặt PyTorch
Cài đặt bản PyTorch tương thích với phần cứng của bạn:

- **Nếu máy tính có GPU NVIDIA (CUDA 11.8)**:
  ```powershell
  python -m pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
  ```
- **Nếu máy tính chỉ sử dụng CPU**:
  ```powershell
  python -m pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
  ```

#### Bước 3: Cài đặt các thư viện phụ thuộc
```powershell
# Cài đặt toàn bộ thư viện phục vụ ứng dụng Web UI và xử lý ảnh
python -m pip install -r requirements.demo.txt

# Kiểm tra tính toàn vẹn phụ thuộc
python -m pip check
```

#### Bước 4: Tải checkpoint mô hình bắt buộc (DDColor Checkpoint)
Mô hình DDColor Large (~912 MB) bắt buộc phải có để khởi chạy ứng dụng. Tải về và đặt đúng vị trí mặc định:
```powershell
curl.exe --fail --location --create-dirs "https://huggingface.co/piddnad/DDColor-models/resolve/main/ddcolor_modelscope.pth" --output "modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt"
```

#### Bước 5: (Tùy chọn) Tải các mô hình nâng cao
Để kích hoạt đầy đủ các tính năng LaMa Inpainting, MobileSAM và đo lường LPIPS:

```powershell
# 1. Tải mô hình phục hồi Microsoft Scratch Detector & LaMa Inpainting
python scripts/fetch_restoration_models.py --model scratch
python scripts/fetch_restoration_models.py --model lama

# 2. Cài đặt runtime và tải trọng số MobileSAM (cho Đổi màu & Xóa vật thể)
git clone https://github.com/ChaoningZhang/MobileSAM.git third_party/MobileSAM
python -m pip install --no-deps --target third_party/mobilesam_runtime timm==0.6.13
Copy-Item third_party/MobileSAM/weights/mobile_sam.pt pretrain/mobile_sam.pt

# 3. Cài đặt thư viện LPIPS và checkpoint AlexNet
python -m pip install --target pretrain/restoration/python --no-deps lpips==0.1.4
python scripts/fetch_restoration_models.py --model lpips
```

---

## 6. HƯỚNG DẪN CHẠY

### 6.1. Khởi chạy giao diện

```powershell
python demo/gradio_app.py
```
Sau khi khởi chạy thành công, mở trình duyệt web và truy cập: **`http://127.0.0.1:7860`**


### 6.2. Chạy suy luận qua giao diện dòng lệnh 
Xử lý tô màu hàng loạt ảnh trong thư mục:
```powershell
python scripts/infer.py --model_path modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt --input assets/test_images --output results
```

### 6.3. Chạy kiểm thử tự động & Benchmark đánh giá 
```powershell
# Kiểm thử toàn bộ các module chức năng
python -m unittest discover -s tests -v

# Chạy benchmark đánh giá thuật toán phục hồi ảnh
python scripts/benchmark_restoration.py --out reports/restoration/paired-final --side 256
```

---

## 7. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
├── assets/                 # Dữ liệu ảnh mẫu, sơ đồ kiến trúc hệ thống
├── basicsr/                # Hạ tầng mạng nơ-ron kế thừa cho huấn luyện
├── ddcolor/                # Kiến trúc mô hình DDColor và pipeline suy luận màu
├── demo/                   # Mã nguồn ứng dụng Gradio và các module xử lý
│   ├── gradio_app.py       # Tệp khởi chạy giao diện Web chính
│   ├── document_scanner.py # Module nắn phẳng và hiệu chỉnh phối cảnh
│   ├── photo_restoration.py# Pipeline phục hồi ảnh cũ (5 giai đoạn)
│   ├── object_segmentation.py # Tích hợp phân đoạn ảnh MobileSAM
│   ├── object_recoloring.py# Module đổi màu đối tượng trong không gian LAB
│   ├── object_removal.py   # Module xóa vật thể (Inpainting)
│   ├── mask_editor.py      # Trình biên tập mask thủ công (Bút vẽ / Tẩy)
│   └── quality_metrics.py  # Đánh giá chỉ số chất lượng: PSNR, SSIM, DeltaE, LPIPS
├── docs/                   # Tài liệu phân tích chuyên sâu các module
├── pretrain/               # Trọng số mô hình phụ trợ (MobileSAM, LaMa, Scratch)
├── reports/                # Báo cáo thực nghiệm và số liệu benchmark
├── scripts/                # Script tiện ích: tải model, kiểm tra môi trường, suy luận CLI
├── tests/                  # Bộ kiểm thử tự động (Unit tests)
├── third_party/            # Microsoft Scratch
├── requirements.txt        # Danh mục thư viện nền tảng cốt lõi
├── requirements.demo.txt   # Danh mục thư viện đầy đủ cho Gradio Demo
```

