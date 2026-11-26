"""Chiến lược marketing cho từng persona (Bước 4 của đề).

Tên persona khớp với evaluation/results/personas.json của TV3.
"""
from __future__ import annotations

STRATEGIES: dict[str, dict] = {
    "VIP": {
        "muc_tieu": "Giữ chân và tăng giá trị vòng đời; biến khách thành người giới thiệu.",
        "khuyen_mai": [
            "Hạng thành viên cao nhất: tích điểm x2, miễn phí vận chuyển mọi đơn",
            "Quyền đặt trước sản phẩm mới 1 tuần trước khi mở bán",
            "Quà tặng dịp đặc biệt thay vì giảm giá trực tiếp, tránh bào mòn biên lợi nhuận",
            "Chương trình giới thiệu: khách cũ và khách mới cùng nhận voucher",
        ],
        "kenh": "Email cá nhân hóa, quản lý khách hàng riêng cho nhóm chi tiêu cao nhất",
        "tan_suat": "2 email/tháng",
        "kpi": "Tỷ lệ giữ chân 90 ngày, giá trị đơn trung bình, số khách được giới thiệu",
        "email_tieu_de": "Cảm ơn {ten_khach} - quyền lợi thành viên dành riêng cho bạn",
        "email_noi_dung": (
            "Chào {ten_khach},\n\n"
            "Cảm ơn bạn đã đồng hành cùng cửa hàng với {so_don} đơn hàng. "
            "Là thành viên hạng cao nhất, bạn được đặt trước bộ sưu tập mới từ tuần này "
            "và miễn phí vận chuyển cho mọi đơn hàng.\n\n"
            "Giới thiệu bạn bè để cả hai cùng nhận voucher 10%.\n"
        ),
    },
    "Tiềm năng": {
        "muc_tieu": "Tăng tần suất mua: đưa khách từ 2 đơn lên mua đều đặn.",
        "khuyen_mai": [
            "Voucher có ngưỡng chi tiêu cho đơn kế tiếp (ví dụ giảm 10% cho đơn từ £300)",
            "Gợi ý sản phẩm mua kèm dựa trên mặt hàng đã mua",
            "Combo tiết kiệm cho các mặt hàng thường được mua cùng nhau",
        ],
        "kenh": "Email tự động theo chuỗi, banner gợi ý trên website",
        "tan_suat": "Chuỗi 3 email trong 3 tuần, sau đó 1 email/tháng",
        "kpi": "Tỷ lệ mua đơn tiếp theo trong 60 ngày, tần suất mua",
        "email_tieu_de": "{ten_khach}, sản phẩm hợp với đơn hàng trước của bạn",
        "email_noi_dung": (
            "Chào {ten_khach},\n\n"
            "Dựa trên các sản phẩm bạn đã chọn, chúng tôi gợi ý một số mặt hàng đi kèm. "
            "Đơn hàng tiếp theo từ £300 được giảm 10% trong 14 ngày tới.\n"
        ),
    },
    "Cần níu kéo": {
        "muc_tieu": "Kéo khách quay lại trước khi rời bỏ hẳn; tìm hiểu lý do hủy đơn.",
        "khuyen_mai": [
            "Ưu đãi quay lại có thời hạn ngắn (7-14 ngày)",
            "Khảo sát ngắn về lý do hủy/trả hàng, tặng voucher khi hoàn thành",
            "Ưu tiên xử lý đổi trả nhanh, cam kết giao đúng hẹn",
        ],
        "kenh": "Email + khảo sát, có thể gọi điện cho khách giá trị cao trong nhóm",
        "tan_suat": "1 email ngay, nhắc lại sau 7 ngày nếu chưa mở",
        "kpi": "Tỷ lệ quay lại mua trong 30 ngày, tỷ lệ hủy đơn",
        "email_tieu_de": "Lâu rồi không gặp {ten_khach} - ưu đãi dành riêng trong 14 ngày",
        "email_noi_dung": (
            "Chào {ten_khach},\n\n"
            "Đã {so_ngay} ngày kể từ đơn hàng gần nhất của bạn. Chúng tôi muốn biết "
            "trải nghiệm vừa rồi của bạn thế nào: trả lời khảo sát 1 phút để nhận voucher 15% "
            "dùng trong 14 ngày.\n"
        ),
    },
    "Ngủ đông": {
        "muc_tieu": "Tái kích hoạt với chi phí thấp; dừng liên hệ nếu không phản hồi.",
        "khuyen_mai": [
            "Giảm giá sâu cho một nhóm sản phẩm bán chạy",
            "Thông báo mùa lễ/khuyến mãi lớn",
        ],
        "kenh": "Email hàng loạt, chi phí thấp",
        "tan_suat": "1 email/tháng, ngừng sau 3 lần không mở",
        "kpi": "Tỷ lệ mở email, tỷ lệ kích hoạt lại, chi phí trên mỗi khách quay lại",
        "email_tieu_de": "{ten_khach} ơi, giảm đến 30% các mặt hàng bán chạy",
        "email_noi_dung": (
            "Chào {ten_khach},\n\n"
            "Mùa mua sắm đã đến, các mặt hàng bán chạy nhất đang giảm đến 30%. "
            "Ghé cửa hàng để xem những gì mới.\n"
        ),
    },
    "Ngoại biên DBSCAN": {
        "muc_tieu": "Xác minh trước khi đưa vào chiến dịch: khách sỉ, đơn đặc biệt hay dữ liệu bất thường.",
        "khuyen_mai": [
            "Không đưa vào chiến dịch tự động",
            "Nếu là khách sỉ hợp lệ: chính sách giá sỉ, chăm sóc riêng",
        ],
        "kenh": "Bộ phận kinh doanh liên hệ trực tiếp",
        "tan_suat": "Theo từng trường hợp",
        "kpi": "Doanh thu nhóm khách sỉ, tỷ lệ giao dịch bất thường được xác minh",
        "email_tieu_de": "",
        "email_noi_dung": "",
    },
}


def strategy_for(persona: str) -> dict:
    return STRATEGIES.get(persona, STRATEGIES["Ngoại biên DBSCAN"])


def render_email(persona: str, ten_khach: str, so_don: int, so_ngay: int) -> tuple[str, str]:
    s = strategy_for(persona)
    values = {"ten_khach": ten_khach, "so_don": so_don, "so_ngay": so_ngay}
    return s["email_tieu_de"].format(**values), s["email_noi_dung"].format(**values)
