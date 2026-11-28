# TV1 - Dữ liệu & Tiền xử lý

**Người phụ trách:** Hoàng Hữu Nhân (2312419)
**Nhánh làm việc:** `feature/preprocessing`
**Tương ứng:** Bước 1 của đề, Giai đoạn 1-2 (Business & Data Understanding, Data Preprocessing), rubric mục 1 (20%)

## Mục tiêu

Từ file giao dịch thô tạo ra bảng **mỗi khách hàng một dòng** với các chỉ số RFM và đặc trưng hành vi, đã biến đổi và chuẩn hóa sẵn để nhóm gom cụm dùng ngay.

## Công việc

### 1. EDA trên dữ liệu thô
- [ ] Đọc cả 2 sheet `Year 2009-2010` và `Year 2010-2011`, gộp lại
- [ ] Thống kê mô tả: số dòng, số hóa đơn, số khách, số sản phẩm, số quốc gia, khoảng thời gian
- [ ] Tỷ lệ thiếu dữ liệu theo cột (đặc biệt `Customer ID`, `Description`)
- [ ] Phân bố theo quốc gia, theo tháng (có tính mùa vụ không)
- [ ] Tìm các trường hợp bất thường: `Quantity` âm, `Price` = 0 hoặc âm, hóa đơn bắt đầu bằng `C`, dòng trùng, `StockCode` không phải sản phẩm (`POST`, `M`, `BANK CHARGES`, `AMAZONFEE`...)

### 2. Làm sạch
- [ ] Bỏ dòng thiếu `Customer ID`
- [ ] Bỏ dòng trùng lặp
- [ ] Bỏ các mã không phải sản phẩm
- [ ] Tách hóa đơn hủy/trả (Invoice bắt đầu bằng `C`) ra riêng, không bỏ hẳn vì còn dùng để tính tỷ lệ hủy
- [ ] Chỉ giữ giao dịch có `Quantity > 0` và `Price > 0`
- [ ] Ghi lại số dòng còn lại sau mỗi bước (đưa vào báo cáo dạng bảng)

### 3. Trích xuất đặc trưng theo khách hàng
Ngày mốc tính Recency = ngày giao dịch cuối cùng trong data + 1 ngày.

| Đặc trưng | Cách tính |
|---|---|
| Recency | Số ngày từ lần mua cuối đến ngày mốc |
| Frequency | Số hóa đơn khác nhau |
| Monetary | Tổng `Quantity * Price` |
| AvgOrderValue | Monetary / Frequency |
| UniqueProducts | Số `StockCode` khác nhau đã mua |
| AvgBasketSize | Tổng số lượng / số hóa đơn |
| Tenure | Số ngày từ lần mua đầu tiên đến ngày mốc |
| CancelRate | Số hóa đơn hủy / (số hóa đơn mua + số hóa đơn hủy) |

Có thể đề xuất thêm đặc trưng khác nếu thấy hợp lý, nhưng báo trước cho TV2, TV3.

### 4. Biến đổi & chuẩn hóa
- [ ] Vẽ histogram, boxplot, tính skewness từng đặc trưng
- [ ] Log-transform (`np.log1p`) để giảm lệch phải, so sánh skewness trước/sau
- [ ] Chuẩn hóa bằng `RobustScaler` (giải thích vì sao chọn Robust thay vì Standard/MinMax: ít bị ảnh hưởng bởi ngoại biên)
- [ ] Ma trận tương quan giữa các đặc trưng, xem có cặp nào tương quan quá cao cần bỏ bớt không

## Đầu ra bàn giao (hạn: cuối tuần 2)

| File | Mô tả |
|---|---|
| `preprocessing/eda.ipynb` | Notebook EDA có biểu đồ và nhận xét |
| `preprocessing/preprocess.py` | Hàm làm sạch + trích xuất đặc trưng, gọi lại được từ phần khác |
| `data/processed/customers_features.csv` | Bảng đặc trưng gốc (chưa scale), index là `Customer ID` |
| `data/processed/X_scaled.csv` | Đặc trưng đã log + RobustScaler |
| `models/scaler.joblib` | Scaler đã fit, app cần dùng để xử lý dữ liệu mới |

File trong `data/` và `models/` không commit, gửi qua Drive nhóm.

## Phần báo cáo phụ trách
- Chương 2: Mô tả dữ liệu & EDA
- Chương 3: Tiền xử lý & trích xuất đặc trưng

## Câu hỏi nên chuẩn bị khi vấn đáp
- Vì sao bỏ khách thiếu `Customer ID` mà không điền giá trị?
- Vì sao cần log-transform trước khi scale?
- RobustScaler khác StandardScaler ở đâu?
- Ngày mốc tính Recency chọn thế nào, chọn khác thì có ảnh hưởng không?
