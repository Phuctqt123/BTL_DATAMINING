# TV3 - Đánh giá & Phân tích cụm

**Người phụ trách:** ...
**Nhánh làm việc:** `feature/evaluation`
**Tương ứng:** Bước 3 của đề, Giai đoạn 3 (Evaluation), rubric mục 2 (30%, làm cùng TV2)

## Mục tiêu

So sánh khách quan 3 thuật toán của TV2, chọn ra mô hình cuối cùng, rồi diễn giải từng cụm thành persona khách hàng có ý nghĩa kinh doanh.

## Công việc

### 1. So sánh mô hình
- [ ] Tính cho cả 3 thuật toán: Silhouette (càng cao càng tốt), Davies-Bouldin (càng thấp càng tốt), Calinski-Harabasz (càng cao càng tốt)
- [ ] Với DBSCAN, tính chỉ số sau khi bỏ điểm nhiễu và ghi rõ tỷ lệ nhiễu
- [ ] Kiểm tra độ ổn định: chạy K-Means với nhiều `random_state`, so sánh nhãn bằng Adjusted Rand Index
- [ ] So sánh mức độ giống nhau giữa các thuật toán (ARI giữa K-Means và Hierarchical)
- [ ] Trực quan hóa cụm bằng PCA 2D (hoặc t-SNE) cho từng thuật toán
- [ ] Lập bảng tổng hợp và chọn mô hình cuối, có lý do

### 2. Phân tích đặc trưng cụm
- [ ] Bảng giá trị trung bình / trung vị của từng đặc trưng theo cụm
- [ ] Kích thước mỗi cụm, tỷ trọng số khách và tỷ trọng doanh thu
- [ ] **Radar Chart** so sánh các cụm (chuẩn hóa min-max các trung bình trước khi vẽ)
- [ ] Boxplot từng đặc trưng theo cụm

### 3. Gán persona
- [ ] Đặt tên cho từng cụm dựa trên đặc trưng, tối thiểu các nhóm trong đề: VIP, Tiềm năng, Ngủ đông, Cần níu kéo
- [ ] Viết mô tả chi tiết cho mỗi persona: hành vi mua, mức chi tiêu, độ gắn bó, rủi ro rời bỏ
- [ ] Xác định quy tắc gán nhãn rõ ràng (ví dụ so trung vị R/F/M của cụm với trung vị toàn tập) để app dùng lại

### 4. Phân tích ngoại biên
- [ ] Xem các điểm DBSCAN đánh dấu nhiễu là khách thế nào (khách sỉ chi tiêu cực lớn? khách mua 1 lần?)
- [ ] Đề xuất cách xử lý nhóm này (tách riêng thành nhóm đặc biệt hay gộp vào cụm gần nhất)

## Đầu ra bàn giao (hạn: cuối tuần 4)

| File | Mô tả |
|---|---|
| `evaluation/compare_models.ipynb` | Bảng so sánh, PCA plot, ARI, chọn mô hình |
| `evaluation/cluster_profiling.ipynb` | Profile cụm, radar chart, boxplot, ngoại biên |
| `evaluation/personas.py` | Hàm gán persona từ nhãn cụm |
| `evaluation/results/personas.json` | Mapping cụm -> tên persona + mô tả |
| `evaluation/figures/` | Hình radar, PCA, boxplot dùng cho báo cáo và app |

## Phần báo cáo phụ trách
- Chương 5: Đánh giá mô hình, so sánh, phân tích từng phân khúc và persona

## Câu hỏi nên chuẩn bị khi vấn đáp
- Silhouette tính thế nào? Giá trị bao nhiêu thì được coi là tốt?
- Khi các chỉ số mâu thuẫn nhau (Silhouette chọn k=2, Elbow gợi ý k=4) thì quyết định thế nào?
- Vì sao radar chart phải chuẩn hóa trước khi vẽ?
- Không có nhãn thật thì làm sao biết phân cụm có "đúng" không?
