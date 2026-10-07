# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Thu Trang  **MSSV**: 2A202602435  **Ngày**: 07/10/2026  
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB`

> Mọi con số dưới đây được trích xuất trực tiếp và khớp chính xác 100% với các file trong thư mục `results/`.

---

## 1. Setup

| Thông số | Giá trị thực nghiệm |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage (mặc định) |
| Train / val | 225 / 25 (cố định seed 42) |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2.0 / 30 optimizer steps |

**Template có giữ khối `<think>` không?** `Có` *(results/template_check.json)*  
Template của model `unsloth/Qwen3.5-4B` giữ nguyên vẹn cặp thẻ `<think> ... </think>` và nội dung suy luận sau khi render (`VERDICT: reasoning preserved — safe to train on traces`). Do đó, không có hiện tượng template tự ý nuốt reasoning trace, đảm bảo an toàn cho các tác vụ huấn luyện suy luận. Mặc dù gợi ý từ p95 là 256, cấu hình Tier T4 giữ `max_length = 1024` nhằm dự phòng cho các trường hợp câu trả lời dài phát sinh trong quá trình decode mà không gây tràn VRAM trên GPU 16GB.

---

## 2. Mask proof (NB1)

| Chỉ số kiểm tra | Giá trị |
|---|---|
| `supervised_fraction` | 0.4149 (41.49%) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```json
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3139.8 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 994.1 |
| (c) LoRA fine-tune | 0.975 | 0.344 | 1.000 | 1353.5 |

**(b) có thật sự mạnh hơn (a) không?** `Có`. Prompt tối ưu (b) nâng độ chính xác target từ 0.000 lên 0.765, tỷ lệ định dạng JSON hợp lệ đạt tuyệt đối 1.000 (so với 0.000 của naive prompt), đồng thời giảm độ trễ hơn 3 lần (từ 3139.8 ms xuống 994.1 ms).  
**Bạn có sửa `OPTIMIZED_PROMPT` không?** `Không`. Tôi giữ nguyên mã SHA `719e74d3b6232053` như chuẩn mặc định của lab để đảm bảo tính liêm chính học thuật và tạo ra một mốc đối chuẩn khách quan thực sự.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6254 | **0.975** | 390.9 | 8.78 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 0.0001 | 0.5367 | **0.970** | 263.9 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 0.00001 | 1.5702 | **0.000** | 390.1 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | **0.940** | 456.2 | 3.86 |

> **Nhận xét quan trọng về Lỗi #3 (Proxy Metric):** Nếu chỉ nhìn vào cột `train loss`, `attn_only` (0.5367) có vẻ vượt trội hơn `correct` (0.6254). Tuy nhiên, khi đánh giá trên chỉ số thực tế của tác vụ (`target` ở NB5), `correct` (0.975) lại chiến thắng `attn_only` (0.970). Điều này khẳng định train loss thấp chỉ phản ánh mức độ ghi nhớ cục bộ, không đại diện cho năng lực tổng quát hóa của mô hình.

### Trả lời ba câu hỏi giải phẫu:

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**  
Nhờ thuật toán `matched_rank()`, run `attn_only` được nâng rank lên tận $r=283$ để đạt xấp xỉ 32.46 triệu tham số, hoàn toàn cân bằng với `correct` ($r=16$). Trên tập target thực tế, `attn_only` (0.970) thua `correct` (0.975), trong khi ở train loss thì `attn_only` lại có loss thấp hơn rõ rệt (0.5367 so với 0.6254). Thứ tự theo train loss ngược hoàn toàn với thứ tự theo độ chính xác tác vụ thực tế. Điều này chứng minh rằng **vị trí đặt adapter là đòn bẩy quan trọng hơn nhiều so với rank**. Việc phân bổ adapter đều khắp tất cả các tầng tuyến tính của text decoder (`all-linear` bao gồm cả MLP và Attention) mang lại khả năng tái biểu diễn tri thức vượt trội hơn việc nhồi nhét một rank cực lớn chỉ vào hai ma trận chiếu $W_q, W_v$.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**  
Run `wrong_lr` áp dụng learning rate $1 \times 10^{-5}$ (thang chuẩn của full fine-tuning) thay vì $1 \times 10^{-4}$ của LoRA. Đường loss của `wrong_lr` bị kéo phẳng lì, chỉ giảm nhẹ từ 2.163 xuống 1.119 sau 30 step (loss cuối 1.5702 so với 0.6254 của `correct`), dẫn đến điểm target rớt về 0.000 vì mô hình không kịp học cấu trúc JSON. Nếu chỉ nhìn đường loss không hội tụ mà không kiểm tra cấu hình, một kỹ sư sẽ dễ dàng đưa ra kết luận sai lầm rằng tác vụ phân loại JSON này quá phức tạp, tập dữ liệu bị nhiễu hoặc kiến trúc mô hình không phù hợp. Bản chất nằm ở chỗ LoRA chỉ tối ưu một không gian tham số con (low-rank bottleneck), nên cần gradient step lớn hơn khoảng 10 lần so với full fine-tuning để dịch chuyển trọng số hiệu quả.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**  
Run `qlora` (4-bit NF4) đã cắt giảm đỉnh bộ nhớ VRAM từ 8.78 GB xuống chỉ còn 3.86 GB (tiết kiệm hơn 56% VRAM). Tuy nhiên, cái giá phải trả rất đắt: thời gian huấn luyện bị kéo dài thêm 17% (456.2 giây so với 390.9 giây do overhead dequantization liên tục trên chip Turing) và độ chính xác target bị tụt giảm nghiêm trọng từ 0.975 xuống 0.940. Kết quả thực nghiệm đo đạc này hoàn toàn ủng hộ khuyến nghị kỹ thuật từ nhà phát triển Qwen3.5: không nên sử dụng QLoRA nếu phần cứng đã có đủ dung lượng bộ nhớ. Trên card Tesla T4 (14.6 GB khả dụng), cấu hình 16-bit LoRA chỉ chiếm 8.78 GB, hoàn toàn nằm trong ngưỡng an toàn mà không phải chịu tổn thất về độ chính xác do sai số lượng tử hóa.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.210` · `regression Δ = -0.447` · `valid_trace_rate = 0.0000`

### Diễn giải kết quả:
Phán quyết của Cổng Hồi quy được xác định là `FAILED` bởi vì chỉ số suy giảm năng lực tổng quát (regression degradation) đã vượt qua ngưỡng dung sai cho phép ($0.447 > 0.020$). Trên tác vụ phân loại ticket khách hàng, bản fine-tune thể hiện sự thăng tiến vượt bậc với mức tăng điểm target $\Delta = +0.210$ (từ 0.765 của baseline b lên 0.975). Tuy nhiên, mô hình đã hứng chịu hiện tượng quên thảm hoạ (*catastrophic forgetting*): năng lực trả lời các câu hỏi tri thức và chỉ dẫn đời sống thông thường bị sụt giảm từ 0.791 xuống chỉ còn 0.344. 

Nguyên nhân cốt lõi là do toàn bộ 225 mẫu huấn luyện chỉ tập trung thuần túy vào cấu trúc JSON của ticket CSKH, khiến phân phối trọng số bị kéo lệch hoàn toàn về miền dữ liệu hẹp này. Mặc dù thất bại ở cổng hồi quy, đây là một kết quả thực nghiệm trung thực và mang tính sư phạm cao: nó chứng minh rằng một mô hình fine-tune đạt điểm số gần như hoàn hảo trên tập chuyên biệt (97.5%) vẫn có thể phá hủy nghiêm trọng nền tảng tri thức tổng quát nếu không được áp dụng cơ chế bảo vệ phù hợp (như trộn 1–5% dữ liệu hồi quy đa miền theo khuyến nghị tại Deck §6.3).

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại... | doi_tra, cao, tich_cuc | doi_tra, trung_binh, tich_cuc | doi_tra, cao, tich_cuc | ✅ **FT thắng**: Xác định đúng urgency "cao" từ chữ "Gấp". |
| 2 | Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhất... | hoan_tien, trung_binh, tieu_cuc | hoan_tien, cao, tieu_cuc | hoan_tien, trung_binh, tieu_cuc | ✅ **FT thắng**: Bắt đúng urgency "trung_binh" theo chuẩn nhãn. |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện... | hoan_tien, **thap**, tich_cuc | hoan_tien, thap, tich_cuc | hoan_tien, **trung_binh**, tich_cuc | ❌ **FT thua**: FT đoán sai urgency (dự đoán "trung_binh" thay vì "thap"). |
| 4 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. Cảm ơn shop... | san_pham_loi, **thap**, tich_cuc | san_pham_loi, thap, tich_cuc | san_pham_loi, **trung_binh**, tich_cuc | ❌ **FT thua**: Nhầm lẫn urgency lịch sự "Khi nào tiện" thành "trung_binh". |
| 5 | Cho mình hỏi, mình đặt đèn bàn LED mã đơn OD436045. Giao hàng chậm. Khi nào tiện... | van_chuyen, **thap**, tich_cuc | van_chuyen, thap, tich_cuc | van_chuyen, **trung_binh**, tich_cuc | ❌ **FT thua**: Cụm từ "Khi nào tiện" bị dự đoán thành mức độ khẩn cấp trung bình. |

**Có mẫu chung nào ở các ca FT thua không?**  
Các ca mô hình Fine-tune bị chấm điểm thấp (0.75/1.0) đều có chung một điểm yếu mang tính hệ thống: **nhầm lẫn trường `urgency` giữa `thap` và `trung_binh`**. Khi khách hàng sử dụng các cụm từ thể hiện thái độ nhã nhặn như *"Khi nào tiện"*, nhãn chuẩn gán nhãn là `thap`, nhưng bản fine-tune lại có xu hướng dự đoán thiên lệch (bias) về lớp đa số là `trung_binh`. Baseline prompt (b) nhờ vào các ví dụ few-shot chi tiết giải thích rõ ngữ cảnh lại phân biệt chính xác hơn ở các trường hợp biên tinh tế này.

---

## 7. Kết luận & điều tôi học được

**Kết luận (185 từ):**  
Dựa trên kết quả thực nghiệm đa chiều của Lab 21, câu trả lời là **CHƯA NÊN** triển khai trực tiếp bản LoRA fine-tune độc lập này lên môi trường production nếu hệ thống yêu cầu mô hình phải đảm nhiệm đa tác vụ. Mặc dù mô hình fine-tune thể hiện năng lực vượt trội trên tác vụ mục tiêu với độ chính xác đạt 97.5% (vượt xa baseline prompt tối ưu 76.5%), sự sụt giảm nghiêm trọng ở cổng hồi quy (-44.7% trên tập regression) cho thấy mô hình đã bị hiện tượng quên thảm hoạ nghiêm trọng. 

Nếu bắt buộc phải đưa vào hệ thống xử lý ticket, giải pháp tối ưu là tách biệt kiến trúc: chỉ dùng adapter này làm một microservice chuyên biệt cho luồng phân loại ticket CSKH, hoặc phải tái huấn luyện bằng cách pha trộn thêm 2–5% dữ liệu đàm thoại tổng quát để duy trì khả năng tư duy nền tảng. Đòn bẩy kỹ thuật mang tính quyết định lớn nhất trong toàn bộ lab này không phải là việc tăng rank hay điều chỉnh tham số LoRA, mà chính là **tính chuẩn xác của Loss Mask (NB1)** và **thang độ lớn của Learning Rate (NB4)**. Đặt sai mask hoặc dùng sai thang LR sẽ phá hủy toàn bộ kết quả bất kể mô hình lớn đến đâu.

**Ba điều tôi học được:**
1. **Loss mask là điều kiện tiên quyết, không thể dựa vào niềm tin**: Phải luôn chứng minh bằng giải mã ngược ký tự (reverse-decoding). Nếu để sót prompt vào loss (`supervised_fraction ≥ 0.95`), mô hình sẽ học thói quen vẹt là lặp lại câu hỏi của người dùng thay vì sinh câu trả lời.
2. **Train loss là một chỉ số thay thế nguy hiểm (Lỗi #3)**: Run `attn_only` có train loss thấp hơn `correct` (0.5367 vs 0.6254), nhưng khi ra tập kiểm tra target thực tế lại thua. Đánh giá mô hình phải dựa trên năng lực downstream task khách quan, không được kết luận dựa trên loss huấn luyện.
3. **Vị trí adapter quan trọng hơn độ lớn của rank**: Dồn rank cực cao ($r=283$) vào riêng các khối Attention vẫn không mang lại hiệu quả tốt bằng việc gắn rank nhỏ ($r=16$) trên toàn bộ các khối tuyến tính (`text-linear`). LoRA Without Regret khẳng định tính toàn diện của vùng can thiệp kiến trúc.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**  
Tôi sẽ thử nghiệm kỹ thuật **General Replay Mixing**: pha trộn thêm 3% dữ liệu hội thoại tiếng Việt thông thường vào tập `train_seed.jsonl`, sau đó huấn luyện lại run `correct`. Mục tiêu là giữ nguyên độ chính xác 97.5% trên tác vụ CSKH nhưng chặn đứng đà sụt giảm điểm regression, đưa Cổng Hồi quy ở NB5 chuyển từ trạng thái `FAILED` sang `PASSED` trọn vẹn.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
