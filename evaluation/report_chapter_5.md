# Chương 5. Đánh giá mô hình và phân tích phân khúc

## 5.1. Phương pháp đánh giá

Do dữ liệu không có nhãn phân khúc thật, các mô hình được so sánh bằng chỉ số nội bộ trên cùng bộ đặc trưng đã chuẩn hóa. Silhouette càng cao càng cho thấy các cụm tách biệt và gọn; Davies–Bouldin càng thấp càng tốt; Calinski–Harabasz càng cao càng tốt. Không có một ngưỡng duy nhất đủ để kết luận mô hình phân cụm là đúng, vì vậy kết quả chỉ số cần được xem cùng kích thước cụm, tính ổn định và khả năng diễn giải kinh doanh.

Với DBSCAN, các chỉ số được tính sau khi bỏ nhãn nhiễu `-1`. Cấu hình TV2 chọn để phát hiện ngoại biên (`eps=0.893989`, `min_samples=16`) gán 345 khách vào nhóm nhiễu, tương đương 5,90%; 5.507 khách còn lại nằm trong một cụm. Vì chỉ còn một cụm sau khi bỏ nhiễu, Silhouette, Davies–Bouldin và Calinski–Harabasz không xác định cho cấu hình này. Để có kết quả DBSCAN hợp lệ trong bảng chỉ số, notebook đánh giá thêm cấu hình tốt nhất trong lưới có ít nhất hai cụm (`eps=0.735187`, `min_samples=5`): Silhouette 0,323, Davies–Bouldin 0,648, Calinski–Harabasz 16,664, tỷ lệ nhiễu 7,76%. Tuy nhiên cụm nhỏ nhất chỉ gồm 5 khách, nên cấu hình này không phù hợp làm phân khúc kinh doanh.

## 5.2. So sánh và lựa chọn mô hình

| Thuật toán | Cấu hình | Số cụm | Tỷ lệ nhiễu | Silhouette | Davies–Bouldin | Calinski–Harabasz |
|---|---|---:|---:|---:|---:|---:|
| K-Means | `k=4` | 4 | 0% | 0,218 | 1,401 | 1.778,083 |
| Hierarchical (Ward) | `k=5` | 5 | 0% | 0,176 | 1,533 | 1.388,191 |
| DBSCAN (cấu hình đánh giá) | `eps=0.735187`, `min_samples=5` | 2 | 7,76% | 0,323 | 0,648 | 16,664 |

DBSCAN có Silhouette cao nhất trong bảng nhưng tạo cụm rất mất cân bằng: cụm nhỏ nhất chỉ có 5 khách. Vì mục tiêu là xây dựng các phân khúc khách hàng có thể diễn giải và dùng lại trong ứng dụng, lựa chọn mô hình chính là **K-Means với `k=4`**. Mô hình này đáp ứng mục tiêu đề tài về bốn nhóm, các cụm có quy mô đủ lớn và có thể gán cụm cho khách hàng mới. Hierarchical được giữ làm kết quả đối chiếu; DBSCAN được sử dụng chủ yếu để phát hiện ngoại biên.

K-Means đạt Silhouette 0,218, nhỉnh hơn cấu hình Hierarchical Ward đã bàn giao (0,176). Adjusted Rand Index giữa hai cách phân cụm là 0,453, cho thấy hai mô hình có mức đồng thuận vừa phải nhưng không tạo cùng một cách chia khách hàng. Khi chạy K-Means với `random_state` 0, 1, 7, 42 và 100, ARI so với nhãn đã bàn giao lần lượt xấp xỉ 0,9993; 0,9952; 0,9988; 1,0000; 0,9997. Kết quả cho thấy nghiệm khá ổn định với các lần khởi tạo được thử.

Các chỉ số không phải nhãn đúng/sai tuyệt đối. Silhouette có thể ưu tiên ít cụm hơn hoặc bỏ qua tính hữu ích của nhóm nhỏ; Elbow, chỉ số nội bộ, độ ổn định và mục tiêu kinh doanh cần được cân nhắc đồng thời. PCA 2D là phép chiếu để quan sát cấu trúc, có thể làm mất thông tin ở các chiều bị lược bỏ; vì vậy hình PCA không thay thế đánh giá trên không gian tám đặc trưng.

## 5.3. Hồ sơ các cụm K-Means

Toàn bộ 5.852 khách được gán vào bốn cụm. Monetary là tổng doanh thu gắn với khách hàng; tỷ trọng doanh thu được tính trên tổng Monetary của toàn bộ khách.

| Cụm | Số khách | Tỷ trọng khách | Tỷ trọng doanh thu | Median Recency (ngày) | Median Frequency | Median Monetary | Median CancelRate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.613 | 27,56% | 81,63% | 25 | 11 | 3.987,07 | 0,143 |
| 1 | 1.982 | 33,87% | 9,67% | 149 | 2 | 681,17 | 0 |
| 2 | 938 | 16,03% | 6,56% | 184,5 | 3 | 942,57 | 0,333 |
| 3 | 1.319 | 22,54% | 2,14% | 307 | 1 | 186,05 | 0 |

Radar chart sử dụng trung bình từng đặc trưng đã chuẩn hóa Min–Max riêng theo từng chiều trên bốn cụm. Boxplot biểu diễn từng đặc trưng theo cụm; trục log được dùng cho các đặc trưng dương để giảm ảnh hưởng của lệch phải. Giá trị dùng để diễn giải và persona vẫn là giá trị gốc, không phải giá trị trên radar hoặc trục log.

## 5.4. Persona và hàm ý kinh doanh

- **Cụm 0 — VIP:** trung vị Recency thấp nhất và Frequency/Monetary cao nhất. Cụm chỉ chiếm 27,56% khách nhưng tạo 81,63% doanh thu. Nên duy trì chăm sóc cá nhân hóa, quyền lợi thành viên và ưu tiên sản phẩm mới.
- **Cụm 1 — Tiềm năng:** cụm đông nhất (33,87%), trung vị hai hóa đơn và Monetary 681,17. Đây là nhóm đã có mua lại nhưng tần suất và giá trị còn cách xa nhóm VIP; có thể nuôi dưỡng bằng gợi ý sản phẩm liên quan, combo và ưu đãi khuyến khích lần mua kế tiếp.
- **Cụm 2 — Cần níu kéo:** trung vị Recency 184,5 ngày, Frequency 3 và CancelRate 0,333, cao nhất trong các cụm. Nên tìm hiểu nguyên nhân hủy và thử chiến dịch quay lại có thời hạn thay vì chỉ tăng mức giảm giá.
- **Cụm 3 — Ngủ đông:** trung vị Recency 307 ngày, Frequency 1 và Monetary 186,05, thấp nhất; cụm chiếm 22,54% khách nhưng chỉ tạo 2,14% doanh thu. Có thể thử chiến dịch tái kích hoạt chi phí thấp và hạn chế tiếp tục liên hệ nếu không có phản hồi.

Ánh xạ nhãn cụm sang persona được lưu trong `results/personas.json` và cung cấp qua `get_persona`/`assign_personas` ở `personas.py`. Mapping này gắn với đúng mô hình K-Means bốn cụm hiện tại; cần rà soát lại nếu huấn luyện mô hình khác vì số thứ tự nhãn cụm không có ý nghĩa thứ bậc.

## 5.5. Phân tích ngoại biên DBSCAN

Nhóm nhiễu hiện có 345 khách (5,90%), với tổng Monetary khoảng 5.975.825, tương đương xấp xỉ 35,01% tổng doanh thu của tập khách. Trung vị Monetary của nhóm nhiễu là 1.927,30, cao hơn nhóm không nhiễu (834,44); trung vị AvgOrderValue lần lượt là 441,18 và 275,44. 25,51% khách nhiễu nằm ở hoặc trên ngưỡng Monetary phân vị 95 của toàn tập, so với 3,72% ở nhóm không nhiễu. Với AvgBasketSize, các tỷ lệ tương ứng là 39,71% và 2,85%.

Nhóm nhiễu có trung vị Frequency bằng 3, giống nhóm không nhiễu; vì vậy không có bằng chứng cho thấy nhóm này chủ yếu là khách mua một lần. Kết quả nghiêng về nhóm chứa khách có chi tiêu, giá trị đơn trung bình hoặc giỏ hàng lớn, nhưng không đủ để kết luận tất cả là khách sỉ. Khuyến nghị giữ nhóm nhiễu riêng để xác minh giao dịch và hành vi khách hàng; không tự động gộp về cụm gần nhất. Có thể xây dựng chính sách riêng sau khi xác nhận các trường hợp giá trị lớn là hợp lệ.


