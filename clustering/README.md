# TV2 - Mô hình gom cụm

**Người phụ trách:** Nguyễn Cảnh Nguyên (2312357)
**Nhánh làm việc:** `feature/clustering`
**Tương ứng:** Bước 2 của đề, Giai đoạn 3 (Modeling), rubric mục 2 (30%, làm cùng TV3)

## Mục tiêu

Cài đặt 3 thuật toán gom cụm trên dữ liệu đã chuẩn hóa của TV1, tinh chỉnh siêu tham số cho từng thuật toán và bàn giao nhãn cụm cho TV3 đánh giá.

Rubric yêu cầu: tối thiểu 3 thuật toán, **giải thích rõ nguyên lý**, **có tinh chỉnh siêu tham số**.

## Công việc

### 1. K-Means
- [X] Chạy k từ 2 đến 10
- [X] Vẽ Elbow (inertia theo k)
- [X] Vẽ Silhouette score theo k, thêm silhouette plot cho 2-3 giá trị k tốt nhất
- [X] Chọn k và giải thích (kết hợp Elbow, Silhouette và ý nghĩa kinh doanh: đề hướng tới khoảng 4 nhóm)
- [X] Thử thêm `init` (k-means++ / random) và `n_init` xem kết quả có ổn định không

### 2. Hierarchical Clustering
- [X] Vẽ Dendrogram (nên lấy mẫu khoảng 2.000 khách vì vẽ toàn bộ rất chậm và khó nhìn)
- [X] So sánh các kiểu linkage: `ward`, `complete`, `average`, `single`
- [X] Chọn số cụm bằng cách cắt dendrogram, đối chiếu với Silhouette

### 3. DBSCAN
- [X] Vẽ k-distance graph để chọn `eps` tại điểm gãy
- [X] Grid search `eps` và `min_samples`
- [X] Ghi lại số cụm và tỷ lệ điểm nhiễu (`-1`) cho từng bộ tham số
- [X] Mục đích chính của DBSCAN trong đề là **phát hiện ngoại biên**, nên không cần ép ra đúng 4 cụm

### 4. Tinh chỉnh tham số
Clustering không có nhãn nên không dùng `GridSearchCV` trực tiếp được. Cách làm: tự viết vòng lặp qua lưới tham số, chấm điểm bằng Silhouette / Davies-Bouldin / Calinski-Harabasz, lưu thành bảng. Có thể dùng `sklearn.model_selection.ParameterGrid` để tạo lưới.

## Đầu ra bàn giao (hạn: cuối tuần 3)

| File | Mô tả |
|---|---|
| `clustering/kmeans.ipynb` | Elbow, Silhouette, chọn k |
| `clustering/hierarchical.ipynb` | Dendrogram, so sánh linkage |
| `clustering/dbscan.ipynb` | k-distance, grid search |
| `clustering/models.py` | Hàm fit từng thuật toán với tham số đã chọn |
| `clustering/results/tuning_*.csv` | Bảng kết quả tinh chỉnh của từng thuật toán |
| `clustering/results/labels.csv` | Cột `Customer ID`, `kmeans`, `hierarchical`, `dbscan` |
| `models/kmeans.joblib` | Model K-Means cuối, app cần để dự đoán khách mới |

## Phần báo cáo phụ trách
- Chương 4: Nguyên lý 3 thuật toán (có công thức, mã giả hoặc hình minh họa), quá trình tinh chỉnh tham số, hình Elbow / Silhouette / Dendrogram / k-distance

## Câu hỏi nên chuẩn bị khi vấn đáp
- K-Means có nhược điểm gì với dữ liệu lệch và có ngoại biên? Vì sao phải scale trước?
- Ward linkage khác complete/average linkage thế nào?
- DBSCAN chọn `eps` dựa vào đâu? Điểm core, border, noise là gì?
- Vì sao Hierarchical khó chạy trên toàn bộ dữ liệu (độ phức tạp)?
