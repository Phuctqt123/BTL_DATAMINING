# TV3 - Đánh giá & Phân tích cụm

**Người phụ trách:** ...
**Nhánh làm việc:** `feature/evaluation`
**Tương ứng:** Bước 3 của đề, Giai đoạn 3 (Evaluation), rubric mục 2 (30%, làm cùng TV2)

## Mục tiêu

So sánh khách quan 3 thuật toán của TV2, chọn ra mô hình cuối cùng, rồi diễn giải từng cụm thành persona khách hàng có ý nghĩa kinh doanh.

## Công việc

### 1. So sánh mô hình
- [X] Tính cho cả 3 thuật toán: Silhouette (càng cao càng tốt), Davies-Bouldin (càng thấp càng tốt), Calinski-Harabasz (càng cao càng tốt)
- [X] Với DBSCAN, tính chỉ số sau khi bỏ điểm nhiễu và ghi rõ tỷ lệ nhiễu
- [X] Kiểm tra độ ổn định: chạy K-Means với nhiều `random_state`, so sánh nhãn bằng Adjusted Rand Index
- [X] So sánh mức độ giống nhau giữa các thuật toán (ARI giữa K-Means và Hierarchical)
- [X] Trực quan hóa cụm bằng PCA 2D (hoặc t-SNE) cho từng thuật toán
- [X] Lập bảng tổng hợp và chọn mô hình cuối, có lý do

### 2. Phân tích đặc trưng cụm
- [X] Bảng giá trị trung bình / trung vị của từng đặc trưng theo cụm
- [X] Kích thước mỗi cụm, tỷ trọng số khách và tỷ trọng doanh thu
- [X] **Radar Chart** so sánh các cụm (chuẩn hóa min-max các trung bình trước khi vẽ)
- [X] Boxplot từng đặc trưng theo cụm

### 3. Gán persona
- [X] Đặt tên cho từng cụm dựa trên đặc trưng, tối thiểu các nhóm trong đề: VIP, Tiềm năng, Ngủ đông, Cần níu kéo
- [X] Viết mô tả chi tiết cho mỗi persona: hành vi mua, mức chi tiêu, độ gắn bó, rủi ro rời bỏ
- [X] Xác định quy tắc gán nhãn rõ ràng (mapping nhãn mô hình K-Means đã lưu sang persona) để app dùng lại

### 4. Phân tích ngoại biên
- [X] Xem các điểm DBSCAN đánh dấu nhiễu là khách thế nào (khách sỉ chi tiêu cực lớn? khách mua 1 lần?)
- [X] Đề xuất cách xử lý nhóm này (giữ riêng để xác minh hành vi/ngoại biên, không tự động gộp)

## Đầu ra bàn giao (hạn: cuối tuần 4)

| File | Mô tả |
|---|---|
| `evaluation/compare_models.ipynb` | Bảng so sánh, PCA plot, ARI, chọn mô hình |
| `evaluation/cluster_profiling.ipynb` | Profile cụm, radar chart, boxplot, ngoại biên |
| `evaluation/personas.py` | Hàm gán persona từ nhãn cụm |
| `evaluation/results/personas.json` | Mapping cụm -> tên persona + mô tả |
| `evaluation/figures/` | Hình radar, PCA, boxplot dùng cho báo cáo và app |
| `evaluation/report_chapter_5.md` | Bản thảo Chương 5 và gợi ý vấn đáp |
| `evaluation/results/` | Bảng profile, tỷ trọng doanh thu, độ ổn định và phân tích nhiễu |

### Ghi chú kết quả

- Mô hình phân khúc chính: K-Means, `k=4`. Chỉ số Silhouette là 0.218; cấu hình Hierarchical Ward `k=5` đạt 0.176. ARI giữa hai nhãn là 0.453.
- DBSCAN cấu hình TV2 chọn để phát hiện ngoại biên có 345 điểm nhiễu (5.90%) và chỉ một cụm còn lại; các chỉ số nội bộ không xác định sau khi loại nhiễu. Notebook so sánh dùng cấu hình có ít nhất hai cụm trong bảng tuning để tính chỉ số DBSCAN, đồng thời ghi rõ cụm nhỏ nhất chỉ có 5 khách nên không dùng cấu hình đó làm phân khúc kinh doanh.
- Persona gắn với số nhãn của mô hình K-Means hiện tại. Nếu huấn luyện lại model và số cụm thay đổi, cần cập nhật mapping trong `results/personas.json`.

## Phần báo cáo phụ trách
- [X] Chương 5: Đánh giá mô hình, so sánh, phân tích từng phân khúc và persona (`evaluation/report_chapter_5.md`)

## Câu hỏi nên chuẩn bị khi vấn đáp
- Silhouette tính thế nào? Giá trị bao nhiêu thì được coi là tốt?
- Khi các chỉ số mâu thuẫn nhau (Silhouette chọn k=2, Elbow gợi ý k=4) thì quyết định thế nào?
- Vì sao radar chart phải chuẩn hóa trước khi vẽ?
- Không có nhãn thật thì làm sao biết phân cụm có "đúng" không?
