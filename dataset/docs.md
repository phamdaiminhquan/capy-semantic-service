cần phải trích xuất số tiền trước khi đưa vào train
chuyển hết số tiền thành dạng token: <MONEY>
Ví dụ:
"Tôi đã chi tiêu 500000 đồng cho bữa ăn tối." -> "Tôi đã chi tiêu <MONEY> cho bữa ăn tối."

Cần phải word_segment trước khi đưa vào train
Ví dụ: "Cơm gà giảm giá 50k hôm nay." -> "Cơm_gà giảm_giá <MONEY> hôm_nay."

train thế nào thì inference thế đó

Lưu ý:
- Không được bỏ qua bước word_segment
- Không được bỏ qua bước trích xuất số tiền