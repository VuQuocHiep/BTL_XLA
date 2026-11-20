# DDColor Studio

[![arXiv](https://img.shields.io/badge/arXiv-2212.11613-b31b1b.svg)](https://arxiv.org/abs/2212.11613)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models-FF8000)](https://huggingface.co/piddnad/DDColor-models)
[![ModelScope](https://img.shields.io/badge/ModelScope-Demo-8A2BE2)](https://www.modelscope.cn/models/damo/cv_ddcolor_image-colorization/summary)

README này tập trung phân tích **cấu trúc, luồng xử lý và thuật toán** để bạn hiểu project trước khi tự đọc mã nguồn. Tài liệu không đưa lệnh chạy hay đoạn code; tên mô hình, thông số và file chỉ được giữ làm định danh để giải thích hoặc tìm phần triển khai tương ứng.

## 1. Project này làm gì?

Đây là project **tô màu ảnh xám bằng mô hình DDColor**, dựa trên bài báo *DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders* (ICCV 2023). Trong repository hiện tại, DDColor còn được tích hợp vào một ứng dụng Gradio có thêm các công cụ xử lý ảnh:

- Tô màu ảnh bằng DDColor.
- Phục hồi ảnh cũ theo các bước phát hiện hư hỏng, vá vùng hỏng, khử nhiễu và điều chỉnh tương phản.
- So sánh kết quả DDColor trực tiếp với kết quả phục hồi rồi tô màu.
- Chọn vùng bằng điểm nhấn để đổi màu hoặc xóa vật thể.
- Làm thẳng ảnh tài liệu bị chụp xiên.
- Đánh giá kết quả bằng ảnh màu tham chiếu đã căn chỉnh.

Có thể hình dung repository gồm hai lớp:

1. **Lõi DDColor**: mô hình, cách chuẩn bị ảnh, nạp trọng số và chạy suy luận.
2. **DDColor Studio**: giao diện và các bước xử lý bao quanh mô hình.

Phục hồi, đổi màu vật thể và xóa vật thể là các công cụ hỗ trợ; chúng không phải là các thành phần của kiến trúc DDColor trong bài báo.

## 2. Bức tranh tổng thể

Luồng tổng quát bắt đầu từ ảnh tải lên. Người dùng có thể làm thẳng ảnh tài liệu trước; sau đó chọn phục hồi, tô màu trực tiếp, hoặc phục hồi rồi tô màu. Hai nhánh so sánh dùng chung ảnh nguồn và cùng DDColor. Đổi màu/xóa vật thể là nhánh riêng: MobileSAM tạo mask từ điểm người dùng chọn rồi thuật toán chỉnh sửa ảnh chỉ tác động lên mask đó.

## 3. Cấu trúc repository

| Thư mục / tệp | Vai trò |
|---|---|
| `ddcolor/` | Giao diện mô hình và pipeline suy luận dùng chung cho script, ứng dụng và Cog. |
| `basicsr/` | Hạ tầng huấn luyện dựa trên BasicSR: mô hình, kiến trúc đăng ký, dữ liệu, loss, metric và tiện ích train. |
| `demo/` | Ứng dụng Gradio và các workflow phục hồi, quét tài liệu, chọn vùng, đổi màu, xóa vật thể và đánh giá. |
| `scripts/` | Các điểm chạy từ dòng lệnh: suy luận, huấn luyện, xuất ONNX, tải model phục hồi và benchmark. |
| `options/train/` | Cấu hình huấn luyện DDColor. |
| `assets/` | Hình minh họa, sơ đồ kiến trúc và ảnh mẫu dùng để thử nghiệm. |
| `pretrain/` | Vị trí dành cho trọng số pretrained; nhiều tệp trọng số không được lưu trong Git. |
| `modelscope/` | Vị trí cache/checkpoint DDColor mà giao diện hiện tại mặc định tìm. Thư mục này có thể phải được tạo hoặc tải model riêng. |
| `third_party/` | Mã nguồn bên thứ ba được dùng có chọn lọc, ví dụ kiến trúc detector scratch của Microsoft. |
| `docs/` | Hướng dẫn chi tiết cho các tính năng mở rộng và ghi chú đánh giá. |
| `tests/` | Kiểm thử luồng dữ liệu, xử lý ảnh, giao diện và các trường hợp lỗi. |
| `reports/` | Kết quả benchmark, ảnh trung gian và báo cáo thực nghiệm; không phải đầu vào bắt buộc để chạy app. |
| `requirements.txt` | Các dependency cơ bản của project. |
| `requirements.demo.txt` | Các dependency bổ sung cho giao diện và đánh giá demo. |
| `setup.py` | Cấu hình đóng gói phần BasicSR vendored; không phải điểm chạy chính của ứng dụng. |

### Các file quan trọng trong `demo/`

| File | Trách nhiệm ở mức tổng quan |
|---|---|
| `gradio_app.py` | Tạo giao diện chính, nạp DDColor và kết nối các tab với workflow. |
| `restoration_workflow.py` | Điều phối phục hồi → tô màu, lưu ảnh trung gian và tái sử dụng kết quả phục hồi còn hợp lệ. |
| `photo_restoration.py` | Các bước xử lý ảnh và cấu hình phục hồi. |
| `restoration_backends.py`, `restoration_worker.py` | Chạy detector/LaMa tùy chọn trong tiến trình CPU riêng. |
| `document_scanner.py` | Phát hiện tờ giấy và hiệu chỉnh phối cảnh. |
| `object_segmentation.py` | Dùng MobileSAM để tạo mask từ ảnh và các điểm người dùng chọn. |
| `object_recoloring.py`, `object_removal.py` | Xử lý mask, đổi màu hoặc xóa vùng được chọn và quản lý lịch sử thao tác. |
| `recoloring_ui.py` | Thành phần giao diện cho hai tab đổi màu và xóa vật thể. |
| `quality_metrics.py`, `metrics_backend.py` | So sánh ảnh đầu ra với ảnh tham chiếu và quản lý phần tính LPIPS tùy chọn. |

## 4. DDColor hoạt động như thế nào?

Ảnh xám không cung cấp đủ thông tin để biết màu gốc của từng vật thể. DDColor dự đoán màu hợp lý dựa trên nội dung và ngữ cảnh ảnh; vì vậy kết quả là một cách tô màu có tính suy đoán, không phải khôi phục chính xác màu lịch sử.

Ở mức khái niệm, kiến trúc gồm:

1. **Encoder thị giác ConvNeXt** trích xuất đặc trưng ở nhiều độ phân giải. Có biến thể ConvNeXt-Tiny và ConvNeXt-Large.
2. **Nhánh decoder ảnh** kết hợp đặc trưng đa tỉ lệ để giữ thông tin không gian và chi tiết.
3. **Nhánh decoder màu** dùng các color query có thể học được. Transformer cập nhật các query bằng cách chú ý đến đặc trưng ảnh ở nhiều mức.
4. **Tạo màu và tinh chỉnh**: các query tạo ra thông tin sắc độ; mạng tinh chỉnh kết hợp thông tin này với ảnh đầu vào để tạo đầu ra.

Encoder tạo một kim tự tháp đặc trưng: các tầng sâu có ngữ cảnh rộng để nhận biết cảnh/vật thể, còn tầng nông giữ vị trí và chi tiết. Decoder ảnh đi ngược qua các tầng, kết hợp đặc trưng sâu với đặc trưng cùng độ phân giải từ encoder (skip connections), rồi tăng độ phân giải. Đây là phần giữ cấu trúc không gian để màu dự đoán bám đúng vùng ảnh.

Decoder màu khởi tạo một tập query có thể học. Mỗi query được cập nhật bằng cross-attention với đặc trưng ảnh ở các mức khác nhau; các tầng decoder lần lượt chuyển qua các mức đặc trưng, rồi self-attention cho phép các query trao đổi thông tin với nhau. Sau feed-forward, một mạng nhỏ biến mỗi query thành vector màu. Tích vô hướng giữa vector query và đặc trưng không gian tạo các bản đồ màu thô; mạng tinh chỉnh kết hợp chúng với đặc trưng ảnh để dự đoán hai kênh sắc độ. Nhiều query giúp mô hình biểu diễn nhiều khả năng màu/ngữ nghĩa đồng thời, thay vì gán một màu duy nhất cho toàn ảnh.

Hai nhánh giải quyết hai vấn đề bổ trợ: decoder ảnh trả lời **màu được đặt ở vị trí nào**, còn decoder query học **nên dùng những thông tin màu nào**. Mạng tinh chỉnh cuối cùng làm mượt và hòa trộn các bản đồ màu với đầu vào. Màu vẫn là dự đoán từ dữ liệu huấn luyện; cấu trúc dual-decoder không thể suy ra màu lịch sử đã mất nếu ảnh xám không chứa thông tin đó.

Sơ đồ kiến trúc trong repository:

![Sơ đồ kiến trúc DDColor](assets/network_arch.jpg)

### Vì sao pipeline dùng không gian màu LAB?

Trong pipeline suy luận, ảnh được chuyển từ RGB/BGR sang LAB:

- **L** chứa độ sáng và cấu trúc sáng tối.
- **a, b** chứa thông tin sắc độ.

Mô hình dự đoán hai kênh sắc độ `a,b`. Pipeline giữ kênh sáng `L` của ảnh nguồn, ghép với sắc độ dự đoán rồi chuyển lại thành ảnh màu. Nhờ vậy, DDColor chủ yếu quyết định màu chứ không thay thế toàn bộ độ sáng và kết cấu của ảnh gốc.

Pipeline tạo ảnh xám đầu vào bằng cách giữ L và đặt a,b về trung tính, đổi ảnh đó thành RGB để đưa vào mạng. Ảnh được chuẩn hóa về kích thước vuông của mô hình; sau suy luận, chỉ hai kênh sắc độ được nội suy về kích thước nguồn và ghép với L gốc ở độ phân giải đầy đủ. Điều này giữ nguyên phần lớn độ sáng/chi tiết ban đầu, nhưng nội suy sắc độ có thể làm màu biên hơi mềm. Phần xử lý màu dùng OpenCV; quy ước ảnh đầu vào/đầu ra của `ColorizationPipeline` là **BGR**, trong khi giao diện Gradio làm việc với **RGB** và chuyển đổi tại ranh giới giữa các thành phần.

## 5. Hai con đường chạy mô hình

### Suy luận

Suy luận đi qua `ddcolor/`: xây mô hình, nạp checkpoint, chuẩn bị ảnh, gọi mô hình để dự đoán màu và chuyển kết quả về ảnh có kích thước ban đầu. `scripts/infer.py` dùng pipeline này để xử lý ảnh trong một thư mục. Giao diện Gradio cũng dùng cùng pipeline cho thao tác tô màu.

Repository có nhiều checkpoint/biến thể. Mô tả từng biến thể và liên kết tải nằm trong [Model Zoo](MODEL_ZOO.md). Trọng số là tệp riêng, thường không nằm trong Git. Script suy luận có thể nhận checkpoint cục bộ hoặc tên model Hugging Face; giao diện hiện tại mặc định tìm checkpoint ModelScope tại `modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt`.

Vì vậy, nếu ứng dụng báo không tìm thấy trọng số thì đó thường là vấn đề model chưa được tải/đặt đúng vị trí, không phải lỗi kiến trúc.

#### Chọn checkpoint nào?

| Model | Điểm khác nhau | Khi nào chọn |
|---|---|---|
| `ddcolor_paper` | Bản Large huấn luyện theo thiết lập trong bài báo. | Muốn đối chiếu/tái tạo kết quả gần với paper. |
| `ddcolor_modelscope` | Bản Large được nhóm tác giả khuyến nghị dùng mặc định trên ảnh ngoài ImageNet. | Lựa chọn tổng quát để bắt đầu thử ảnh thường ngày. |
| `ddcolor_artistic` | Bản Large được huấn luyện thêm với dữ liệu nghệ thuật; phong cách/màu có thể khác. | Thử kết quả thiên hướng hình minh họa/nghệ thuật. |
| `ddcolor_paper_tiny` | Bản Tiny gọn hơn. | Khi ưu tiên mô hình nhẹ hoặc muốn thử suy luận trên phần cứng hạn chế. |

Các tên này là những checkpoint huấn luyện khác nhau, không phải tùy chọn cường độ của cùng một model. Kết quả giữa chúng có thể khác nhau; hãy thử trên ảnh mục tiêu và đọc ghi chú chi tiết trong [Model Zoo](MODEL_ZOO.md).

### Huấn luyện

Huấn luyện đi qua `basicsr/`:

1. Dataset đọc ảnh màu và chuyển chúng sang dạng LAB.
2. Kênh sáng làm đầu vào; kênh sắc độ là mục tiêu mà generator cần dự đoán.
3. Generator DDColor học cách dự đoán màu. Trong mỗi vòng tối ưu, generator được cập nhật trước khi discriminator được huấn luyện phân biệt ảnh màu thật với ảnh do generator tạo ra; tín hiệu đối kháng khuyến khích kết quả giống phân bố ảnh thật hơn.
4. BasicSR điều phối dataloader, tối ưu hóa, checkpoint, log và validation.

Cấu hình mẫu trong [`options/train/train_ddcolor.yml`](options/train/train_ddcolor.yml) dùng ImageNet, crop 256×256, generator ConvNeXt-L với MultiScaleColorDecoder, discriminator và 400.000 iteration. Các loss có vai trò khác nhau: L1 hạn chế sai lệch sắc độ theo pixel; perceptual loss so sánh đặc trưng nhiều tầng VGG để giữ độ tương đồng cảm nhận; GAN loss khuyến khích ảnh tạo ra giống ảnh màu thật; colorfulness loss khuyến khích kết quả đủ phong phú về màu. Trọng số loss là sự cân bằng giữa giống tham chiếu và vẻ tự nhiên, không đảm bảo màu lịch sử chính xác. Đây là cấu hình tham khảo, không phải một bộ cấu hình đã tự dò phù hợp với mọi máy/dataset.

Trước khi train cần chuẩn bị ảnh màu và file danh sách đường dẫn; đặt đúng `dataroot_gt`/`meta_info_file`, số GPU, batch size, đường dẫn trọng số khởi tạo và validation trong YAML. `scripts/train.sh` là launcher shell hiện cấu hình chạy distributed với bốn GPU; đây không phải quy trình huấn luyện một lệnh dành cho Windows thuần. Khi đổi số GPU, phải đồng bộ cấu hình launcher với `num_gpu` và batch size trong YAML.

Các bước này chỉ cần thiết nếu bạn muốn huấn luyện hoặc sửa mô hình. Để dùng checkpoint có sẵn, hãy dùng đường suy luận thay vì khởi chạy training.

## 6. Ứng dụng Gradio và luồng xử lý ảnh

Điểm vào của ứng dụng là `demo/gradio_app.py`. Ứng dụng tạo một pipeline DDColor dùng chung, sau đó kết nối các nút giao diện với workflow tương ứng. Các tác vụ nặng được xếp hàng để tránh nhiều bước cùng lúc tranh chấp bộ nhớ GPU.

Để chạy giao diện, môi trường cần có dependency của demo và checkpoint DDColor ở vị trí ứng dụng mong đợi. Điểm khởi chạy là `demo/gradio_app.py`; khi ứng dụng hoạt động, Gradio hiển thị địa chỉ cục bộ, mặc định là `http://127.0.0.1:7860`.

Chi tiết cài đặt/tài nguyên bổ sung cho từng tính năng nằm trong các tài liệu riêng:

- [Đổi màu vật thể](docs/object_recoloring.md)
- [Xóa vật thể](docs/object_removal.md)
- [Kiểm toán, lựa chọn backend và đánh giá phục hồi](docs/restoration_audit.md)

Ứng dụng hiện tìm checkpoint DDColor tại `modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt`. Checkpoint không nhất thiết được Git lưu trong repository; SDK ModelScope có thể tải model vào cache, sau đó cần đặt/kiểm tra checkpoint đúng vị trí app tìm. Môi trường demo đã được kiểm tra với Python 3.9, PyTorch 2.2.0+cu118, OpenCV 4.7 và Gradio 4.44.1. `requirements.demo.txt` dành cho việc dựng môi trường demo mới; tránh cài lại trọn bộ dependency vào môi trường đang hoạt động nếu không muốn pip thay đổi phiên bản PyTorch/Gradio.

### Những khu vực chính trên giao diện

| Khu vực | Chức năng |
|---|---|
| **Ảnh đầu vào** | Tải ảnh nguồn lên. Có thể dùng **Làm thẳng ảnh** trước khi xử lý nếu ảnh là tờ giấy bị chụp xiên. |
| **Phục hồi trước khi tô màu** | Bật/tắt nhánh phục hồi. Mặc định bật: nút tô màu sẽ phục hồi rồi mới gọi DDColor. Tắt để chạy DDColor trực tiếp trên ảnh nguồn. |
| **Các bước phục hồi** | Bật/tắt riêng phát hiện hư hỏng, vá ảnh, khử nhiễu, chỉnh tương phản và làm nét. |
| **Thông số nâng cao** | Chọn detector/backend và điều chỉnh ngưỡng, độ nở mask, cường độ xử lý, kích thước tối đa của model CPU. |
| **Vùng vá thủ công** | Tải mask cùng kích thước ảnh nguồn; vùng trắng là vùng cần vá. Mask thủ công thay detector tự động. |
| **Tô màu DDColor** | Chạy theo lựa chọn phục hồi đang bật/tắt và hiển thị ảnh nguồn/kết quả trong thanh trượt. |
| **Phục hồi ảnh cũ** | Chỉ chạy phục hồi, không gọi DDColor; hữu ích để kiểm tra mask và ảnh đã vá trước. |
| **So sánh DDColor gốc** | Chạy DDColor trực tiếp và Phục hồi + DDColor trên cùng ảnh, lần lượt từng nhánh. |
| **Các tab kết quả** | Xem ảnh màu, ảnh trước/sau phục hồi, các giai đoạn trung gian, mask và bảng metric. |

Nếu vừa chạy so sánh, những tính năng **Dùng kết quả DDColor** trong hai tab chọn vùng lấy kết quả của nhánh **Phục hồi + DDColor**. Nếu chỉ chạy tô màu một nhánh, chúng lấy kết quả của nhánh đó.

### Phục hồi ảnh cũ

Luồng phục hồi đi theo thứ tự: ảnh nguồn → tạo mask hư hỏng (tự động hoặc thủ công) → vá vùng mask → khử nhiễu → điều chỉnh tương phản khi cần → làm nét nếu người dùng bật. Mỗi giai đoạn có thể tắt riêng; các ảnh trung gian cho phép xác định bước nào tạo cải thiện hay gây mất chi tiết.

- **Phát hiện hư hỏng**: có detector dựa trên OpenCV và tùy chọn detector scratch pretrained của Microsoft.
- **Vá ảnh**: có OpenCV Telea và tùy chọn LaMa. Nếu không có model ngoài, chế độ `auto` có thể dùng phương án OpenCV.
- **Khử nhiễu/tương phản/làm nét**: các bước xử lý ảnh truyền thống với thông số điều chỉnh được; làm nét mặc định tắt.
- **Mask thủ công**: dùng khi detector bỏ sót hoặc chọn sai vùng. Mask trắng chỉ vùng cần vá.

##### Detector tạo mask ra sao?

- **OpenCV** là bộ dò theo quy tắc ảnh, không phải mô hình hiểu ngữ nghĩa. Nó so ảnh với median cục bộ để đo phần lệch sáng/tối, dùng top-hat/black-hat để nhấn các vệt sáng/tối mảnh, rồi tìm các ứng viên có cấu trúc đường thẳng đủ dài và được pixel hỗ trợ liên tục. Bụi được tìm riêng bằng các thành phần liên thông rất nhỏ có độ tương phản cao. Cuối cùng phép đóng hình thái nối khe nhỏ và phép giãn nở phủ thêm viền. Ưu điểm là không cần checkpoint và chạy nhanh; nhược điểm là vết cong, mờ, rộng hoặc lẫn với tóc/đường nét ảnh dễ bị bỏ sót/nhận nhầm.
- **Microsoft Scratch Detection** là U-Net đã huấn luyện để dự đoán xác suất vùng hư hỏng từ ảnh xám. Ảnh được thu nhỏ trong giới hạn cạnh tối đa để chạy CPU; xác suất được đưa về kích thước gốc rồi so với ngưỡng. Hạ ngưỡng tăng độ nhạy nhưng thường tạo mask rộng hơn và dễ nhận nhầm chi tiết. Đây là detector scratch chứ không phải bộ phân đoạn hiểu mọi dạng mất ảnh.
- **Mask thủ công** thay thế detector tự động, hữu ích với hư hỏng lớn/không theo dạng scratch. Sau khi tạo mask, tham số mở rộng vùng dùng phép giãn nở hình elip để mask trùm qua mép hư hỏng. Mask tự động phủ trên 12% ảnh và mask thủ công phủ trên 30% ảnh không được tự động vá, nhằm hạn chế thao tác có thể phá hỏng nhiều nội dung.

##### Thuật toán vá và xử lý ảnh

- **Telea** dùng fast marching: bắt đầu từ biên mask, lần lượt điền pixel gần biên trước và ước lượng màu từ vùng lân cận còn nguyên. Phương pháp này nhẹ, hợp với vết nhỏ/hẹp và nền biến đổi từ từ; vùng lớn có thể bị nhòe hoặc lặp kết cấu.
- **Navier–Stokes** lan truyền hướng biên/đường đẳng sáng từ vùng còn nguyên vào vùng khuyết theo một mô hình dòng chảy. Nó có thể giữ hướng đường nét tốt trong vài trường hợp, nhưng vẫn là nội suy cục bộ; không thể dựng lại vật thể hay kết cấu lớn đã mất.
- **LaMa** là mô hình inpainting học từ dữ liệu, dùng ngữ cảnh rộng hơn để dự đoán vùng bị che. Nó có thể tạo kết cấu hợp lý hơn khi vùng vá phức tạp, nhưng nội dung mới chỉ là phỏng đoán. Project chạy model CPU ở kích thước giới hạn rồi ghép dự đoán vào các pixel trong mask ở độ phân giải nguồn; pixel ngoài mask không bị lấy từ ảnh đã resize.
- **Khử nhiễu** ước lượng độ nhiễu từ thành phần tần số cao ở vùng ít cạnh, rồi áp dụng Non-Local Means lên kênh sáng. NLM tìm các mảng ảnh tương tự thay vì chỉ làm mờ lân cận; cường độ quá cao vẫn làm mất hạt/texture nhỏ. Kênh sắc độ không được khử nhiễu riêng trong bước này.
- **Tăng tương phản** lấy các phân vị sáng tối để kéo giãn dải L, nhưng chỉ khi dải sáng rơi vào khoảng được coi là ảnh tương phản thấp. Ảnh vốn có tương phản cao hoặc gần như một màu được giữ nguyên để tránh tự ý làm gắt.
- **Làm nét** là unsharp masking: lấy chi tiết cao tần bằng cách trừ bản Gaussian blur khỏi ảnh rồi cộng một phần chi tiết trở lại. Ngưỡng theo độ nhiễu loại bớt hạt nhỏ; tuy vậy nó không phục hồi thông tin mất do out-of-focus và có thể làm rõ cả vết xước còn sót. Vì vậy mặc định tắt.

Khi có LaMa, nó được chạy trong worker CPU riêng để không dùng chung bộ nhớ GPU với DDColor. Kết quả LaMa chỉ được ghép vào vùng mask; kích thước ảnh đầu ra được giữ nguyên. Việc khởi động worker và nạp model làm tăng thời gian xử lý.

Ứng dụng có thể lưu các ảnh trung gian như mask, vùng phủ mask và kết quả của từng giai đoạn để người dùng xem quá trình, thay vì chỉ thấy ảnh cuối. Kết quả phục hồi được cache theo ảnh nguồn, tùy chọn và mask; khi một trong các yếu tố này đổi, kết quả cũ không còn được dùng.

#### Các thông số phục hồi trên giao diện

Giá trị dưới đây là phạm vi/giá trị mặc định của giao diện hiện tại:

| Điều khiển | Phạm vi / mặc định | Tác động |
|---|---|---|
| **Bộ phát hiện vết xước** | `auto` (mặc định), `microsoft`, `opencv` | `auto` dùng detector Microsoft nếu model đã có; nếu không thì dùng detector OpenCV. Chọn rõ backend để buộc dùng detector đó. |
| **Phương pháp vá ảnh** | `auto` (mặc định), `lama`, `telea` | `auto` chọn LaMa nếu model đã có, nếu không dùng Telea. |
| **Phát hiện hư hỏng** | Bật mặc định | Tắt nếu không muốn tạo mask tự động; vẫn có thể tải mask thủ công. |
| **Vá vùng hỏng** | Bật mặc định | Tắt để chỉ xem các bước còn lại mà không inpaint vùng mask. |
| **Khử nhiễu** | Bật mặc định | Xử lý nhiễu trên độ sáng. |
| **Tăng tương phản** | Bật mặc định | Chỉ kéo giãn trong trường hợp dải sáng phù hợp; ảnh không cần tăng tương phản có thể được giữ nguyên. |
| **Làm nét nhẹ** | Tắt mặc định | Chỉ bật khi cần; làm nét có thể khuếch đại nhiễu hoặc vết xước còn sót. |
| **Ngưỡng phát hiện** | `0.1–0.9`, mặc định `0.4` | Chủ yếu áp dụng cho xác suất mask của detector Microsoft. Ngưỡng thấp thường nhận nhiều pixel hơn, nhưng cũng có thể tăng vùng báo nhầm. |
| **Mở rộng vùng vá** | `0–5 px`, mặc định `1 px` | Nới mask trước khi vá để tránh viền hỏng còn lại. Giá trị lớn hơn cũng sửa nhiều pixel hơn. |
| **Mức khử nhiễu** | `0–12`, mặc định `0` (tự động) | `0` để hệ thống ước lượng nhiễu; giá trị lớn hơn yêu cầu khử mạnh hơn và có thể làm mất chi tiết. |
| **Mức tăng tương phản** | `0–0.5`, mặc định `0.2` | Cường độ pha trộn giữa ảnh gốc và ảnh đã kéo giãn tương phản. `0` tương đương không áp dụng bước này. |
| **Mức làm nét** | `0–0.5`, mặc định `0.2` | Chỉ có tác dụng khi bật **Làm nét nhẹ**. Nên tăng từ từ và kiểm tra vùng nhiễu/viền. |
| **Cạnh tối đa của model CPU** | `256–768 px`, mặc định `512 px` | Giới hạn kích thước ảnh đưa vào detector/LaMa CPU; tăng có thể giữ chi tiết nhỏ nhưng tốn thời gian hơn. |

Mask detector tự động phủ hơn 12% ảnh có thể bị từ chối vá; mask thủ công phủ hơn 30% cũng bị chặn để tránh thay đổi diện rộng. Nếu detector bỏ sót vết xước, có thể tải mask thủ công hoặc chuyển detector. Trong tab **Các giai đoạn**, xem mask và lớp phủ đỏ trước khi quyết định dùng kết quả. Chọn đích danh Microsoft/LaMa khi thiếu model sẽ báo lỗi; chế độ `auto` mới tự chọn phương án thay thế được ghi trong bảng.

#### Chọn thông số theo tình huống

- **Ảnh gần như sạch, chỉ cần tô màu**: tắt **Phục hồi trước khi tô màu** để tránh các bước xử lý không cần thiết.
- **Ảnh có nhiễu hạt**: giữ **Khử nhiễu** bật và để mức `0` tự động trước; nếu còn nhiễu, thử tăng nhẹ. So sánh chi tiết nhỏ để tránh làm ảnh bệt.
- **Ảnh bạc màu**: bật **Tăng tương phản**; nếu kết quả quá gắt, giảm mức tăng. Hệ thống có thể tự bỏ qua nếu dải sáng không cần kéo giãn.
- **Vết xước mảnh**: xem mask trước. Nếu đường xước bị bỏ sót, thử thay detector hoặc cung cấp mask thủ công; chỉ tăng mở rộng vùng vừa đủ để che viền.
- **Vật thể/vùng hỏng lớn**: dùng mask thủ công chính xác và xem xét kết quả sau vá. Inpainting không thể biết chính xác nội dung bị mất.
- **Ảnh mờ**: pipeline mặc định không khử mờ. Làm nét không thể phục hồi thông tin chưa có và có thể làm lộ nhiễu.

### So sánh DDColor với phục hồi + DDColor

Chế độ so sánh chạy hai nhánh tuần tự trên cùng ảnh nguồn và cùng một instance DDColor: nhánh thứ nhất đưa ảnh gốc thẳng vào mô hình; nhánh thứ hai phục hồi ảnh trước rồi mới tô màu. Việc chạy tuần tự tránh tranh chấp bộ nhớ GPU, đồng thời làm cho khác biệt giữa hai kết quả chủ yếu đến từ bước phục hồi.

Hai nhánh được chạy tuần tự và dùng cùng mô hình DDColor. Nếu không cung cấp ảnh màu tham chiếu đã căn chỉnh, ứng dụng chỉ nên dùng hình ảnh và thời gian để đối chiếu; các điểm PSNR/SSIM/LPIPS/ΔE00 sẽ không có ý nghĩa khi thiếu ground truth.

#### Ví dụ các luồng thường dùng

1. **Chỉ tô màu**: tải ảnh lên, tắt **Phục hồi trước khi tô màu**, bấm **Tô màu DDColor**.
2. **Chỉ xem phục hồi**: tải ảnh lên, chỉnh các bước/thông số, bấm **Phục hồi ảnh cũ**. Mở tab **Các giai đoạn** để kiểm tra mask và ảnh trung gian.
3. **Phục hồi rồi tô màu**: giữ **Phục hồi trước khi tô màu** bật, chỉnh thông số nếu cần, bấm **Tô màu DDColor**.
4. **So sánh hai nhánh**: bấm **So sánh DDColor gốc**. Thanh trượt hiển thị hai đầu ra; bảng bên dưới so sánh PSNR/SSIM/LPIPS/ΔE00 và thời gian. Nếu không có ảnh chuẩn, các metric chất lượng hiện `N/A`.
5. **Đánh giá với ảnh chuẩn**: mở **Ảnh màu chuẩn**, tải ảnh cùng nội dung đã căn chỉnh và cùng kích thước, rồi bấm **So sánh DDColor gốc**. PSNR/SSIM cao thường tốt hơn; LPIPS/ΔE00 thấp thường gần ảnh tham chiếu hơn. Đây là độ giống tham chiếu, không phải đánh giá vẻ đẹp.

### Chọn vùng, đổi màu và xóa vật thể

Hai tab này dùng MobileSAM để tạo mask từ ảnh và các điểm dương/âm người dùng đặt:

#### Cách chọn vùng chung

1. Mở tab **Đổi màu vật thể** hoặc **Xóa vật thể**.
2. Tải ảnh riêng vào tab đó, hoặc bấm **Dùng kết quả DDColor** để chuyển ảnh từ workflow chính.
3. Click lên vật thể trong ảnh bên trái khi đang ở chế độ **+ Thêm vùng**. MobileSAM dự đoán mask; ảnh **Vùng chọn** hiển thị lớp phủ và các điểm đã đặt.
4. Chọn **− Loại vùng** rồi click lên phần nền/chi tiết không muốn chọn; quay lại **+ Thêm vùng** để bổ sung phần còn thiếu. Click chỉ cập nhật mask, chưa áp dụng chỉnh sửa.
5. Kiểm tra lớp phủ, bấm **Xác nhận vùng**, sau đó mới bấm **Đổi màu** hoặc **Xóa vật thể**.
6. Thanh trượt **Trước / sau** cho phép đối chiếu kết quả. **Tải ảnh** tải ảnh PNG hiện tại; **Hoàn tác** lùi thao tác gần nhất; **Đặt lại** quay về ảnh được nạp vào tab; **Xóa lựa chọn** chỉ xóa mask/điểm đang chọn.

Một số đối tượng có nhiều phần rời (ví dụ vật thể có lỗ hoặc các bộ phận cách nhau). Thêm điểm `+` vào từng phần cần giữ. Hãy xem mask trước khi xác nhận, vì điểm chỉ là gợi ý cho mô hình phân đoạn chứ không đảm bảo đường biên hoàn hảo.

Về thuật toán, MobileSAM Tiny tạo embedding ảnh một lần rồi dùng tọa độ điểm và nhãn dương/âm làm prompt. Mô hình trả nhiều mask ứng viên; ứng dụng ưu tiên mask thỏa nhiều prompt nhất, dùng điểm chất lượng của SAM để phân xử các ứng viên tương đương. Sau đó nó loại các vùng liên thông rời rạc không chứa điểm dương, tránh lấy nhầm đốm nền cách xa vật thể. Embedding của ảnh được cache để thêm điểm không phải mã hóa lại ảnh; khi ảnh đổi thì embedding được tạo lại. Nếu CUDA thiếu bộ nhớ, segmenter có thể chuyển sang CPU.

#### Đổi màu vật thể

Tab này có ba điều khiển chính:

| Điều khiển | Phạm vi / mặc định | Cách hiểu |
|---|---|---|
| **Màu** | Màu bất kỳ qua color picker; mặc định `#d83030` | Màu đích của vùng được chọn. |
| **Cường độ** | `0–100%`, mặc định `100%` | Mức pha màu mới vào ảnh. Giảm nếu muốn giữ lại nhiều sắc độ cũ hơn. |
| **Mềm cạnh** | `0–30 px`, mặc định `3 px` | Làm chuyển tiếp mềm ở phía trong mask để giảm đường biên gắt; không làm màu tràn ra ngoài mask. |

Ví dụ: muốn đổi áo sang xanh, đặt điểm `+` trên áo, thêm điểm `−` trên da/nền nếu bị chọn lẫn, xác nhận mask, chọn màu xanh, giữ cường độ cao rồi bấm **Đổi màu**. Nếu màu quá mạnh, giảm cường độ; nếu mép áo bị gắt, tăng mềm cạnh một ít.

Thuật toán đổi màu chuyển ảnh RGB sang LAB, giữ nguyên L để bảo toàn sáng/tối, bóng đổ và kết cấu. Màu đã chọn được chuyển thành mục tiêu sắc độ a,b; một phần nhỏ biến thiên sắc độ ban đầu được giữ lại để bề mặt không biến thành mảng phẳng. Cường độ điều khiển mức pha giữa màu cũ và màu mới. Độ mềm cạnh tạo alpha giảm dần **bên trong mask** theo khoảng cách đến biên; nó không làm màu loang ra ngoài vùng chọn. Chỉ pixel trong mask được ghi vào ảnh kết quả, nên ngoài mask giữ nguyên giá trị ảnh.

Khi chỉnh lại màu/cường độ cho cùng vùng sau lần áp dụng đầu, kết quả được tính từ ảnh nền của lần chỉnh vùng đó thay vì cộng dồn màu qua nhiều lần. Sau khi đổi xong một vật thể, đặt thêm điểm `+` để bắt đầu chọn vật thể tiếp theo.

#### Xóa vật thể: ba phương pháp

Tab **Xóa vật thể** có tối đa ba lựa chọn. **OpenCV · Telea** và **OpenCV · Navier–Stokes** luôn có sẵn; **LaMa · CPU (đã có)** chỉ xuất hiện khi checkpoint LaMa đã được chuẩn bị trên máy.

| Phương pháp | Khi nào thử | Thông số bán kính |
|---|---|---|
| **OpenCV · Telea** (mặc định) | Thử đầu tiên với vật nhỏ, vết mảnh hoặc nền tương đối đơn giản. Chạy CPU và không cần model bổ sung. | Có tác dụng; `1–9 px`, mặc định `3 px`. Bán kính lớn hơn ảnh hưởng vùng lân cận rộng hơn. |
| **OpenCV · Navier–Stokes** | Phương án OpenCV thứ hai nếu Telea để lại vệt hoặc kết quả chưa tự nhiên. Nên thử cả hai và so sánh trên ảnh cụ thể. | Có tác dụng; `1–9 px`, mặc định `3 px`. |
| **LaMa · CPU (đã có)** | Có thể thử với vùng cần vá phức tạp hơn khi model đã có. Chạy trong worker CPU, nên thường chậm hơn do khởi động/nạp model. | Không áp dụng; LaMa không dùng bán kính vá OpenCV. |

Ngoài phương pháp, có điều khiển **Mở rộng vùng xóa** (`0–15 px`, mặc định `3 px`). Đây là độ nở mask trước khi inpaint, không phải bán kính của thuật toán. Nếu còn viền vật thể, tăng nhẹ độ mở rộng và xác nhận lại vùng; nếu vùng xung quanh bị sửa quá nhiều, giảm giá trị này. Ảnh phủ kín bởi mask sẽ bị từ chối vì không còn pixel nền để nội suy.

Sau khi backend dự đoán ảnh đã vá, ứng dụng chỉ thay pixel thuộc mask đã mở rộng vào ảnh hiện tại. Vì vậy, vùng ngoài mask được bảo toàn chính xác; chọn thiếu mask có thể để lại viền, còn chọn quá rộng làm thuật toán sửa cả nền thật xung quanh vật thể. Bán kính OpenCV điều chỉnh vùng lân cận dùng để nội suy, trong khi độ mở rộng điều chỉnh pixel nào bị thay đổi — đây là hai tham số khác nhau.

**Ví dụ so sánh ba cách:** chọn vùng và chạy Telea trước. Để thử Navier–Stokes hoặc LaMa trên cùng ảnh gốc, bấm **Đặt lại**, chọn lại vùng tương tự, dùng cùng mức mở rộng rồi chọn phương pháp khác. Sau **Hoàn tác**, vùng chọn bị xóa nên cũng cần chọn/xác nhận lại. Không có phương pháp nào đảm bảo tái tạo đúng nền bị che; với vật thể lớn, bóng đổ hoặc nền nhiều kết cấu, có thể cần chỉnh mask và chấp nhận kết quả ước lượng.

#### Làm thẳng tài liệu

Trong khu vực **Căn chỉnh ảnh chụp**, bấm **Làm thẳng ảnh** để tìm bốn góc của tờ giấy và hiệu chỉnh phối cảnh. Ảnh đầu vào cần là RGB 8-bit, mỗi chiều ít nhất 32 pixel; bước tìm biên thu nhỏ ảnh lớn để phát hiện, nhưng phép biến đổi cuối cùng lấy từ ảnh độ phân giải ban đầu. Hệ thống tìm tờ giấy có bốn góc nhìn thấy và diện tích tối thiểu khoảng 10% ảnh. Sau khi làm thẳng, pipeline dùng ảnh đã hiệu chỉnh làm nguồn cho phục hồi/tô màu và so sánh.

Thuật toán thu nhỏ ảnh lớn để phát hiện ổn định hơn, chuyển sang xám và làm mờ nhẹ để giảm nhiễu. Nó tạo ứng viên biên bằng Canny, đồng thời tạo mặt nạ sáng/tối bằng ngưỡng Otsu; phép đóng hình thái nối các đoạn biên bị đứt. Từ các contour đủ lớn, thuật toán xấp xỉ đa giác với các mức dung sai khác nhau, giữ tứ giác lồi có bốn góc nằm bên trong ảnh, rồi chọn ứng viên lớn nhất. Bốn góc được sắp thứ tự và ánh xạ thành hình chữ nhật bằng phép biến đổi phối cảnh; phép nội suy cuối dùng ảnh nguồn gốc để không giới hạn độ phân giải theo ảnh thu nhỏ phát hiện.

Đây là **hiệu chỉnh phối cảnh**, không phải nhận dạng ký tự (OCR), xóa bóng, làm nét chữ hay tự động sửa nội dung. Nếu giấy không tương phản với nền, một góc bị khuất, hoặc đường viền nền bị nhận nhầm thành giấy, thuật toán có thể báo không tìm thấy tài liệu hoặc chọn sai tứ giác. Hãy chụp lại với toàn bộ mép giấy hiện rõ trên nền tương phản; kết quả không đạt thì quay lại ảnh nguồn thay vì tiếp tục xử lý ảnh bị cắt sai.

Đổi màu tác động lên sắc độ trong LAB để cố gắng giữ ánh sáng và kết cấu. Xóa vật thể dùng inpainting để ước lượng nội dung lấp vào vùng mask. Cả hai chỉ sửa vùng đã chọn; độ chính xác phụ thuộc chất lượng mask, và vùng bị che khuất không thể được khôi phục chắc chắn từ một ảnh đơn.

MobileSAM, checkpoint của nó và một số model phục hồi là tài nguyên riêng, không nhất thiết có sẵn sau khi clone repository. Xem [hướng dẫn MobileSAM](docs/object_recoloring.md) để hiểu tài nguyên cần cho chọn vùng. Detector Microsoft và LaMa được chuẩn bị riêng bằng tiện ích tải model trong project; nếu không có chúng, vẫn dùng được detector OpenCV và Telea. Nếu người dùng chọn tường minh Microsoft/LaMa nhưng tài nguyên chưa sẵn sàng, ứng dụng báo lỗi thay vì giả vờ thành công.

## 7. Các hình thức sử dụng và tích hợp

### Suy luận hàng loạt

`scripts/infer.py` là bộ suy luận theo thư mục: người dùng cung cấp thư mục ảnh, chọn checkpoint cục bộ hoặc tên model Hugging Face, chọn kích thước kiến trúc tương ứng và thư mục đầu ra. Script đọc lần lượt từng file, bỏ qua ảnh OpenCV không giải mã được với thông báo lỗi, tô màu ảnh hợp lệ và lưu kết quả cùng tên file. Kích thước suy luận mặc định là 512 pixel mỗi cạnh; đầu ra được nội suy về kích thước ảnh gốc. Dùng Large cho khả năng biểu đạt cao hơn nếu phần cứng đáp ứng; Tiny giảm tài nguyên tính toán nhưng có thể cho chất lượng khác.

Các tên checkpoint pretrained gồm ddcolor_paper, ddcolor_modelscope, ddcolor_artistic và ddcolor_paper_tiny; chúng khác nhau ở dữ liệu/thiết lập huấn luyện chứ không chỉ khác cường độ. Xem [Model Zoo](MODEL_ZOO.md) để hiểu mục tiêu của từng checkpoint trước khi chọn.

### Các cách tích hợp khác

- **ModelScope**: SDK tải/định tuyến model, nhận ảnh đầu vào và trả ảnh màu trong trường kết quả OUTPUT_IMG. Đây là pipeline đóng gói sẵn; khác với việc gọi trực tiếp ColorizationPipeline trong project.
- **Hugging Face**: checkpoint được lấy từ Hub qua tên model hoặc nạp cục bộ. Cách này thuận tiện khi muốn dùng các biến thể trong Model Zoo mà không tự quản lý file checkpoint.
- **ONNX**: `scripts/export_onnx.py` chuyển trọng số PyTorch và kiến trúc DDColor thành đồ thị ONNX, sau đó chạy kiểm tra shape và đơn giản hóa đồ thị. Notebook [`demo/colorization_pipeline_onnxruntime.ipynb`](demo/colorization_pipeline_onnxruntime.ipynb) minh họa suy luận ONNX Runtime. ONNX là cách đóng gói/chạy mô hình; nó không thay đổi thuật toán tô màu và vẫn cần checkpoint tương thích khi export.
- **Cog/Replicate**: `cog.yaml` mô tả môi trường đóng gói; `demo/cog_predict.py` là adapter nhận một ảnh, cho chọn Large hoặc Tiny, rồi trả về ảnh PNG. Khi khởi động, adapter nạp cả hai model để tái sử dụng qua nhiều yêu cầu. Khi triển khai, phải cung cấp checkpoint đúng các đường dẫn `checkpoints/ddcolor_modelscope.pth` và `checkpoints/ddcolor_paper_tiny.pth`; các file này không được tự tạo bởi adapter.

### Huấn luyện

Luồng train chính nằm trong BasicSR và được cấu hình tại [`options/train/train_ddcolor.yml`](options/train/train_ddcolor.yml). Dataset cần có danh sách đường dẫn ảnh; `scripts/get_meta_file.py` duyệt thư mục dữ liệu và tạo danh sách này. Sau đó cấu hình YAML phải trỏ đúng ảnh train/validation, số GPU, batch size và checkpoint khởi tạo. `scripts/train.sh` hiện dùng launcher distributed bốn GPU trên Linux/POSIX; cần đổi launcher và các tham số khi chuyển môi trường. Huấn luyện đòi hỏi dữ liệu, GPU và checkpoint phù hợp; không cần thực hiện nếu chỉ chạy model có sẵn.

### Các script hỗ trợ

| Script | Dùng để làm gì |
|---|---|
| `scripts/get_meta_file.py` | Tạo danh sách đường dẫn ảnh để cấu hình dataset train/validation sử dụng. |
| `scripts/fetch_restoration_models.py` | Tải riêng detector scratch, LaMa hoặc tài nguyên LPIPS; không tự chạy khi mở ứng dụng. |
| `scripts/export_onnx.py` | Xuất checkpoint DDColor sang ONNX và kiểm tra/simplify đồ thị. |
| `scripts/benchmark_restoration.py` | Đo pipeline phục hồi theo các ca thử nghiệm; có thể bật thêm nhánh DDColor. |
| `scripts/score_restoration_lpips.py` | Bổ sung điểm LPIPS cho kết quả benchmark khi có dependency/model cần thiết. |
| `scripts/summarize_restoration.py` | Tổng hợp các kết quả benchmark phục hồi đã tạo. |
| `scripts/benchmark_recoloring.py` | Đo một số tình huống đổi màu vật thể, tùy chọn chạy cùng DDColor. |
| `scripts/benchmark_removal.py` | Đo các phương pháp xóa vật thể trên ca kiểm thử. |
| `scripts/smoke_recoloring_api.py` | Kiểm tra API của app Gradio đang chạy; có chế độ dành riêng cho luồng xóa vật thể. |

Các giao diện này khác nhau ở cách nạp model và đóng gói, nhưng mục đích chung là đưa ảnh vào mô hình DDColor rồi trả ảnh đã tô màu.

## 8. Đánh giá và giới hạn

Các metric trong giao diện là **metric so với ảnh tham chiếu**, không phải thước đo tuyệt đối cho vẻ đẹp của ảnh:

- **PSNR (dB)**: tính sai số bình phương trung bình giữa các kênh RGB rồi đổi sang thang log. Sai số pixel càng nhỏ thì PSNR càng cao; ảnh trùng hoàn toàn có PSNR vô hạn. Metric nhạy với dịch chuyển/căn chỉnh dù mắt người có thể thấy hai ảnh gần giống.
- **SSIM**: so sánh cục bộ độ sáng, tương phản và cấu trúc qua các cửa sổ ảnh, thay vì chỉ tính sai số tuyệt đối từng pixel. Gần 1 thường là giống hơn, nhưng SSIM không biết màu nào đúng về mặt lịch sử.
- **LPIPS**: đưa ảnh qua mạng đặc trưng AlexNet và so khoảng cách giữa các biểu diễn ở nhiều tầng. Thấp hơn thường gần tham chiếu hơn về đặc trưng cảm nhận; nó phụ thuộc kiến trúc/trọng số mạng và có thể không đồng thuận với PSNR. Cần package và checkpoint AlexNet; nếu không có thì hiển thị `N/A`, không thay bằng điểm giả.
- **ΔE00**: chuyển màu sang CIELAB rồi tính độ lệch màu CIEDE2000, có điều chỉnh độ nhạy theo vùng màu/sắc độ. Thấp hơn thường ít lệch màu hơn; đây là trung bình toàn ảnh nên vùng nhỏ sai màu có thể bị che lấp bởi phần lớn ảnh đúng.
- **Time**: thời gian xử lý của mỗi nhánh; nhánh phục hồi + DDColor bao gồm cả thời gian phục hồi, kể cả khi kết quả phục hồi được dùng lại từ cache theo thời gian đã ghi nhận.

Hai ảnh đầu vào metric phải là RGB 8-bit, cùng kích thước, mỗi chiều ít nhất 64 pixel và đã căn chỉnh cùng nội dung/góc nhìn. Ứng dụng không tự co ảnh tham chiếu vì co giãn có thể làm sai phép so sánh. Không có tham chiếu thì bảng chỉ so sánh thời gian và hiển thị ảnh; PSNR/SSIM/LPIPS/ΔE00 là `N/A`. Ảnh xám gốc không thể tự đóng vai trò ảnh chuẩn màu. Không có tham chiếu màu thật thì không thể kết luận màu dự đoán có đúng với màu lịch sử hay không.

Các metric có thể không đồng thuận: ví dụ một ảnh có SSIM/LPIPS tốt hơn nhưng PSNR kém hơn. Hãy xem ảnh và mục tiêu đánh giá thay vì coi một con số là kết luận duy nhất. LPIPS có thể chạy bằng worker CPU riêng; thiếu dependency/checkpoint hoặc worker lỗi thì ứng dụng báo không khả dụng, không giả định chất lượng.

LPIPS là phần tùy chọn. Khi cần, project có thể đặt thư viện LPIPS và checkpoint AlexNet trong thư mục tài nguyên riêng rồi chạy tính điểm trên CPU worker riêng. Cách ly này giúp tránh thay đổi dependency chính của DDColor. Nếu thiếu package, checkpoint hoặc worker gặp lỗi, điểm LPIPS được đánh dấu không khả dụng thay vì gán giá trị mặc định.

Một số giới hạn cần nhớ:

- Màu DDColor dự đoán có thể khác màu thật; mô hình không biết chính xác màu nguyên bản.
- Detector scratch tự động phù hợp với một số loại hư hỏng nhất định, có thể bỏ sót vết rộng/mờ hoặc đánh dấu nhầm chi tiết ảnh.
- Inpainting chỉ ước lượng phần bị che; không thể đảm bảo tái tạo đúng vật thể/nền đã mất.
- Phục hồi không đồng nghĩa với khử mờ; tăng cường ảnh có thể làm thay đổi chi tiết hoặc màu.
- Kết quả benchmark trong `reports/` là chẩn đoán trên tập ảnh và điều kiện cụ thể, không chứng minh chất lượng tổng quát trên mọi ảnh lịch sử.
- Xóa/đổi màu phụ thuộc mask. Hãy xem mask trước khi áp dụng và chỉnh điểm nếu vùng chọn sai.

## 9. Kiểm thử

`tests/` được chia theo các hành vi: luồng Gradio/workflow phục hồi, các giai đoạn phục hồi, backend pretrained, đổi màu, xóa vật thể, làm thẳng tài liệu và metric. Kiểm thử đơn vị chủ yếu kiểm tra điều kiện đầu vào, định tuyến ảnh, mask, trạng thái phiên và xử lý pixel; chúng không chứng minh mô hình tạo màu/phục hồi tốt trên mọi ảnh thực tế. Có thể chạy toàn bộ bằng unittest discovery. `test_pretrained_restoration.py` là kiểm thử tích hợp có opt-in: cần tải model phục hồi và bật biến môi trường RUN_RESTORATION_INTEGRATION. Các script benchmark tạo đánh giá riêng; benchmark không giống unit test và kết quả trong `reports/` chỉ đại diện cho dữ liệu/thiết lập đã chạy.

## 10. Cách đọc project nếu mới bắt đầu

Để hiểu luồng chính, nên đọc theo thứ tự:

1. README này để nắm nhiệm vụ và các thành phần.
2. [`ddcolor/pipeline.py`](ddcolor/pipeline.py) để hiểu cách checkpoint và ảnh được đưa qua pipeline suy luận.
3. [`ddcolor/model.py`](ddcolor/model.py) và sơ đồ `assets/network_arch.jpg` để hiểu cấu trúc mô hình.
4. [`demo/gradio_app.py`](demo/gradio_app.py) để xem giao diện nối các tác vụ ra sao.
5. [`demo/restoration_workflow.py`](demo/restoration_workflow.py) và [`demo/photo_restoration.py`](demo/photo_restoration.py) để theo dõi nhánh phục hồi rồi tô màu.
6. [`basicsr/data/lab_dataset.py`](basicsr/data/lab_dataset.py), [`basicsr/models/color_model.py`](basicsr/models/color_model.py) và [`options/train/train_ddcolor.yml`](options/train/train_ddcolor.yml) nếu mục tiêu là hiểu huấn luyện.
7. Các tài liệu trong `docs/` nếu cần đi sâu vào một tính năng giao diện cụ thể.

## 11. Bài báo và ghi nhận

DDColor được giới thiệu bởi Xiaoyang Kang, Tao Yang, Wenqi Ouyang, Peiran Ren, Lingzhi Li và Xuansong Xie trong bài báo *DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders*, Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, trang 328–338. Thông tin bài báo và mã trích dẫn có tại trang [arXiv](https://arxiv.org/abs/2212.11613).

Phần huấn luyện sử dụng hạ tầng BasicSR. Một số ý tưởng/mã nguồn liên quan được ghi nhận trong repository gốc, gồm ColorFormer, BigColor, ConvNeXt, Mask2Former và DETR. Khi phân phối lại mã nguồn, checkpoint hoặc tài nguyên bên thứ ba, hãy xem [`LICENSE`](LICENSE) và giấy phép/ghi công đi kèm với từng thành phần.
