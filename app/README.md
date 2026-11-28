# TV4 (Trưởng nhóm) - Ứng dụng & Tổng hợp

**Người phụ trách:** Bùi Đình Phúc (2312664), nhóm trưởng
**Nhánh làm việc:** `feature/app`
**Tương ứng:** Bước 4 của đề, Giai đoạn 4-5, rubric mục 3 (20%) và mục 4 (15%)

## Mục tiêu

Xây giao diện báo cáo phân khúc và đề xuất chiến lược marketing tự động cho từng cụm; ghép sản phẩm của cả nhóm thành bản hoàn chỉnh và tổng hợp báo cáo, slide.

## Công việc

### 1. App Streamlit
- [X] **Tổng quan:** số khách, số phân khúc, tổng doanh thu, bảng tỷ trọng khách và doanh thu theo phân khúc
- [X] **Hồ sơ phân khúc:** radar chart, bảng đặc trưng trung bình, mô tả persona
- [X] **Chiến lược marketing:** với mỗi phân khúc đưa ra gợi ý khuyến mãi và nội dung email marketing
- [X] **Tra cứu khách hàng:** nhập `Customer ID`, hiện phân khúc và chiến lược gợi ý
- [X] **Phân khúc dữ liệu mới:** upload file CSV giao dịch, app tự xử lý (dùng hàm của TV1 + scaler), dự đoán cụm (dùng model của TV2), gán persona (dùng hàm của TV3), cho tải kết quả về

Rubric yêu cầu app "có khả năng nhận dữ liệu thực tế", nên phần upload là bắt buộc.

### 2. Chiến lược marketing theo phân khúc
Gợi ý ban đầu, chỉnh lại sau khi TV3 chốt persona:

| Phân khúc | Khuyến mãi | Email marketing |
|---|---|---|
| VIP | Chương trình thành viên hạng cao, quà tặng, ưu tiên sản phẩm mới | Thư cảm ơn cá nhân hóa, mời giới thiệu bạn bè |
| Tiềm năng | Voucher cho đơn thứ 2-3, gợi ý mua kèm | Chuỗi email chào mừng, gợi ý sản phẩm liên quan |
| Cần níu kéo | Ưu đãi quay lại có thời hạn | Email "lâu rồi không gặp", khảo sát lý do |
| Ngủ đông | Giảm giá sâu, chi phí thấp | Email kích hoạt lại, gửi thưa |

### 3. Tích hợp & quản lý repo
- [X] Tạo khung repo, `requirements.txt`, `.gitignore`
- [X] Review và merge Pull Request của các thành viên
- [X] Đảm bảo pipeline chạy liền mạch từ dữ liệu thô đến app trên một máy sạch
- [X] Viết `README.md` hướng dẫn cài đặt và chạy

### 4. Báo cáo & slide
- [ ] Lập mục lục báo cáo (15-25 trang) và giao khung cho mọi người từ tuần 1
- [ ] Viết Chương 1 (Giới thiệu bài toán), Chương 6 (Ứng dụng), Chương 7 (Kết luận, hướng phát triển)
- [ ] Ghép các chương, thống nhất định dạng, đánh số hình/bảng, trích dẫn tài liệu
- [ ] Ghép slide thuyết trình, chia thời lượng cho từng người

## Mục lục báo cáo dự kiến

| Chương | Nội dung | Người viết |
|---|---|---|
| 1 | Giới thiệu bài toán, mục tiêu, ý nghĩa thực tiễn | TV4 |
| 2 | Mô tả dữ liệu & EDA | TV1 |
| 3 | Tiền xử lý & trích xuất đặc trưng | TV1 |
| 4 | Mô hình gom cụm & tinh chỉnh tham số | TV2 |
| 5 | Đánh giá, so sánh, phân tích persona | TV3 |
| 6 | Ứng dụng demo & chiến lược marketing | TV4 |
| 7 | Kết luận & hướng phát triển | TV4 |
| | Tài liệu tham khảo, phụ lục | TV4 |

## Đầu ra (hạn: cuối tuần 5)

| File | Mô tả |
|---|---|
| `app/streamlit_app.py` | App demo |
| `app/strategies.py` | Bảng chiến lược marketing theo persona |
| `app/segmentation.py` | Phân khúc dữ liệu mới bằng pipeline TV1 → TV2 → TV3 |
| `app/build_artifacts.py` | Lưu `kmeans.joblib` và bảng phân khúc cho app |
| `app/sample/giao_dich_mau.csv` | Dữ liệu mẫu để thử chức năng upload |
| `README.md` | Hướng dẫn cài đặt, chạy |
| `report/` | Báo cáo PDF + slide |

## Câu hỏi nên chuẩn bị khi vấn đáp
- Khách mới upload lên thì được xử lý qua những bước nào trước khi dự đoán cụm?
- Vì sao phải dùng lại scaler đã fit thay vì fit lại trên dữ liệu mới?
- Doanh nghiệp dùng kết quả phân khúc này như thế nào trong thực tế, cập nhật lại bao lâu một lần?
