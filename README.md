# BTL Khai phá Dữ liệu (CO3029) - Đề tài 03

**Phân đoạn khách hàng đa chiều theo mô hình RFM kết hợp hành vi**
(Customer Segmentation via RFM & Clustering)

- Tác vụ chính: Gom cụm (Clustering / Unsupervised Learning)
- Chuẩn đầu ra: L.O.3.1, L.O.3.3, L.O.4.1, L.O.4.2
- Mục tiêu: chia khách hàng thành các nhóm đặc trưng (VIP, Tiềm năng, Ngủ đông, Cần níu kéo) để đề xuất chiến dịch marketing phù hợp cho từng nhóm.

## Dữ liệu

Online Retail II - UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/502/online+retail+ii

Giao dịch của một cửa hàng bán lẻ online ở Anh từ 01/12/2009 đến 09/12/2011, khoảng 1,07 triệu dòng, gồm 8 cột:
`Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country`.

File dữ liệu không đưa lên repo (nặng ~45MB). Mỗi người tự tải file zip ở link trên, giải nén vào `data/raw/`.

## Cấu trúc repo

Mỗi thư mục là một phần việc, có file `README.md` mô tả chi tiết nhiệm vụ của người phụ trách.

```
BTL_DATAMINING/
├── preprocessing/   TV1 - Dữ liệu, EDA, tiền xử lý, trích xuất RFM
├── clustering/      TV2 - Cài đặt và tinh chỉnh K-Means, Hierarchical, DBSCAN
├── evaluation/      TV3 - Đánh giá, so sánh mô hình, radar chart, gán persona
├── app/             TV4 (trưởng nhóm) - Giao diện Streamlit, tích hợp, tổng hợp báo cáo
└── data/            dữ liệu (không commit)
```

## Phân công

| | Thư mục | Nội dung chính | Phần báo cáo | Trọng số rubric liên quan |
|---|---|---|---|---|
| TV1 | `preprocessing/` | EDA, làm sạch, RFM + đặc trưng hành vi, log-transform, RobustScaler | Ch.2 Dữ liệu & EDA, Ch.3 Tiền xử lý | 20% |
| TV2 | `clustering/` | K-Means (Elbow, Silhouette), Hierarchical (Dendrogram, linkage), DBSCAN (k-distance, eps, min_samples), GridSearch tham số | Ch.4 Mô hình | 30% (cùng TV3) |
| TV3 | `evaluation/` | So sánh 3 thuật toán theo nhiều chỉ số, chọn mô hình cuối, Radar Chart, gán persona, phân tích ngoại biên | Ch.5 Đánh giá & phân tích | 30% (cùng TV2) |
| TV4 | `app/` | App Streamlit, chiến lược marketing theo cụm, ghép code, review PR, tổng hợp báo cáo + slide | Ch.1 Giới thiệu, Ch.6 Ứng dụng, Ch.7 Kết luận | 20% + 15% báo cáo |

Thuyết trình (15%): mỗi người trình bày phần mình làm và phải trả lời được câu hỏi về thuật toán của phần đó.

## Luồng bàn giao giữa các phần

```
preprocessing  ->  customers_features.csv, X_scaled + scaler
      |
clustering     ->  nhãn cụm của 3 thuật toán + bảng tinh chỉnh tham số
      |
evaluation     ->  mô hình được chọn, bảng so sánh, persona cho từng cụm, hình radar
      |
app            ->  giao diện demo + báo cáo tổng
```

TV1 cần bàn giao sớm nhất vì 3 phần sau đều phụ thuộc vào dữ liệu đã xử lý. Trong lúc chờ, TV2 và TV3 có thể đọc trước lý thuyết và viết sẵn code chạy thử trên một mẫu nhỏ.

## Tiến độ dự kiến

| Tuần | TV1 | TV2 | TV3 | TV4 |
|---|---|---|---|---|
| 1 | Tải data, EDA | Đọc lý thuyết, code khung 3 thuật toán | Đọc lý thuyết các chỉ số đánh giá | Dựng repo, khung app, mục lục báo cáo |
| 2 | Làm sạch, RFM, scaling, **bàn giao** | Chạy trên data thật | Viết code đánh giá | Khung app đọc kết quả mẫu |
| 3 | Viết Ch.2-3, hỗ trợ TV2 | Tinh chỉnh tham số, **bàn giao** | So sánh, chọn mô hình | Trang tổng quan, tra cứu |
| 4 | Rà soát lại data | Viết Ch.4 | Radar, persona, **bàn giao**, viết Ch.5 | Gắn persona + chiến lược vào app |
| 5 | Slide phần mình | Slide phần mình | Slide phần mình | Ghép báo cáo, slide, chạy thử demo |
| 6 | Tập thuyết trình cả nhóm |||

Ngày cụ thể trưởng nhóm sẽ chốt theo lịch demo của lớp.

## Quy ước làm việc

- Mỗi người làm trên nhánh riêng: `feature/preprocessing`, `feature/clustering`, `feature/evaluation`, `feature/app`.
- Xong việc thì tạo Pull Request vào `main`, trưởng nhóm review rồi merge. Không push thẳng lên `main`.
- Không commit dữ liệu (`data/raw`, `data/processed`) và file model lớn.
- Thư viện mới dùng thì thêm vào `requirements.txt`.
- Notebook trước khi commit nên restart & run all để kết quả khớp với code.
- Đặt `random_state=42` ở mọi chỗ có ngẫu nhiên để các phần chạy ra kết quả giống nhau.

## Tài liệu tham khảo

1. Jiawei Han, Jian Pei, Hanghang Tong. *Data Mining: Concepts and Techniques*, 4th Ed., Morgan Kaufmann, 2022.
2. Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
