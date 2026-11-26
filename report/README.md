# Báo cáo LaTeX

Mã nguồn báo cáo (mẫu `hcmut-report`, biên dịch trên Prism).

- `report.tex`: file chính, thông tin trang bìa, tài liệu tham khảo
- `chapters/01..07_*.tex`: 7 chương, mỗi chương là một `subfile`
- `graphics/`: hình dùng trong báo cáo

Cần thêm `hcmut-report.cls`, `codespace.sty` và `graphics/hcmut.png` từ project Prism. Bản `hcmut-report.cls` trên Prism đã được sửa:

- `babel` dùng `[english,vietnamese]`
- thêm lệnh `\studentthree`, `\studentfour`
- nhãn trang bìa đổi sang tiếng Việt (Ngành, Giảng viên hướng dẫn, Sinh viên thực hiện)

Trước khi nộp cần điền họ tên, MSSV của TV1–TV3 và tên giảng viên trong `report.tex`.
