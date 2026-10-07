# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**  
Điều làm tôi ngạc nhiên nhất là hiện tượng Lỗi #3 khi so sánh giữa `attn_only` và `correct`. Trên tập huấn luyện, `attn_only` (với $r=283$) có loss giảm sâu hơn và nhanh hơn `correct` (0.5367 so với 0.6254), nhưng khi đánh giá thực tế trên tập downstream target thì `correct` lại chiến thắng (0.975 vs 0.970). Điều này cho thấy train loss có thể đánh lừa người làm mô hình đến mức nào nếu không có tập đánh giá tác vụ khách quan.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**  
Tôi mất nhiều thời gian nhất ở công đoạn sinh văn bản (text generation) ở NB2 và NB5 trên GPU T4, do tập đánh giá phải sinh lặp lại ba lần (cho naive prompt, optimized prompt và fine-tune adapter). Ban đầu tôi dự đoán quá trình train LoRA ở NB3 và NB4 sẽ lâu nhất, nhưng thực tế thời gian decode tuần tự không có vLLM/flash-attention mới là nút thắt cổ chai lớn nhất.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**  
Trước đây tôi từng tin rằng "rank càng cao thì LoRA học càng giỏi" và "chỉ cần gắn LoRA vào các ma trận Attention $W_q, W_v$ là đủ chuẩn". Sau lab này, tôi nhận ra việc mở rộng vùng can thiệp ra toàn bộ text decoder (`all-linear`) với rank nhỏ ($r=16$) mang lại hiệu quả vượt trội hơn hẳn việc nhồi nhét một rank khổng lồ ($r=283$) vào riêng Attention. Vị trí đặt adapter chính là đòn bẩy quan trọng nhất.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**  
Tôi dùng AI assistant để phân tích log đào tạo, đối chiếu các con số giữa `runs.csv` với các file JSON và hỗ trợ viết code kiểm thử dữ liệu. Chỗ AI dễ nhầm lẫn nhất là tự động kết luận "fine-tune đã thành công rực rỡ" khi chỉ nhìn vào độ chính xác target đạt 97.5%, mà bỏ qua việc mô hình bị rớt Cổng Hồi quy ở NB5 do regression tụt -0.447 (catastrophic forgetting).

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**  
Bước đầu tiên tôi làm sẽ không phải là tải model về train ngay, mà là:
1. Đóng băng một bộ test đánh giá khách quan và đo baseline với một prompt được tối ưu kỹ càng (Baseline b). Nếu prompt tối ưu đã giải quyết được 80–90% bài toán, tôi sẽ tư vấn khách hàng dùng prompt engineering thay vì tốn chi phí fine-tuning và bảo trì adapter.
2. Nếu bắt buộc fine-tune, tôi sẽ kiểm tra và chứng minh Loss Mask (NB1) đầu tiên bằng cách giải mã ngược để đảm bảo không tính loss lên prompt.
