# Bài phản tư — Lab 22

**Tên:** Nguyễn Văn A  
**Khoá:** A20-K4  
**Tier đã chạy:** T4  
**Ngày:** 2026-10-09

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Google Colab T4, 16 GB |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned`, 1.000 mẫu theo cấu hình T4 |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy`, 800 mẫu train và 100 mẫu held-out theo cấu hình T4 |
| Chosen dài hơn rejected (NB2) | Đã đo trong `02b-pref-length.png`; dữ liệu và tỷ lệ cần đối chiếu với output NB2 đã tải từ Colab |
| DPO: β / tốc độ học / số epoch | 0,1 / 5e-6 / 1 |
| Giám khảo | Hội đồng reward model Skywork Qwen3-4B và Llama-3.2-3B |
| Chi phí | Colab T4 |

## 2. Kết quả DPO

Các số liệu dưới đây lấy từ `adapters/dpo/dpo_metrics.json`.

| Chỉ số | Giá trị |
|---|---:|
| Loss đầu tiên được log | 0,6948245 |
| Loss cuối | 0,6753245 |
| Reward chosen cuối trên train | 0,3618399 |
| Reward rejected cuối trên train | 0,2745395 |
| Reward gap cuối trên train | 0,0873004 |
| Reward chosen trên held-out | 0,3775171 |
| Reward rejected trên held-out | 0,2968261 |
| Margin trên held-out | 0,0806910 |
| Độ chính xác reward trên held-out | 0,65 |
| Chẩn đoán tự động | `INTENDED` |

## 3. Đọc đường reward

Đường reward cho thấy DPO đã tạo ra phân biệt dương giữa câu `chosen` và
`rejected`. Ở cuối quá trình, reward của `chosen` trên train là 0,3618 còn
reward của `rejected` là 0,2745, tạo gap 0,0873. Trên held-out, hai giá trị
tương ứng là 0,3775 và 0,2968, tạo margin 0,0807. Như vậy xu hướng held-out
cùng chiều với train và không cho thấy margin bị đảo dấu. Chẩn đoán tự động
`INTENDED` phù hợp với việc cả hai tập đều có reward chosen cao hơn reward
rejected. Tuy nhiên, margin held-out nhỏ hơn train một chút, vì vậy mức cải
thiện tổng quát hóa là khiêm tốn chứ không nên diễn giải là DPO đã tạo ra
thắng lợi lớn. Reward accuracy held-out là 0,65, cho thấy mô hình phân biệt
được một phần các cặp preference nhưng vẫn còn nhiều cặp khó. Ngoài ra, margin
tăng không nhất thiết có nghĩa xác suất tuyệt đối của chosen tăng: nếu
rejected giảm nhanh hơn thì margin vẫn tăng. Trong run này, các số cuối phù
hợp hơn với chẩn đoán intended, nhưng hiện tượng likelihood displacement vẫn
là điều cần kiểm tra khi đọc riêng từng đường trên biểu đồ.

## 4. So sánh SFT và SFT+DPO

Kết quả dưới đây lấy từ `data/eval/judge_summary.json`. Hội đồng đã chấm 50
cặp held-out và 8 câu cố định theo hai nhóm helpfulness/safety.

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (95% CI) | Win rate cặp dài gần bằng | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 7 | 7 | 36 | 0,500 (0,430–0,570) | 0,521 (n=48) | 0,429 |
| helpfulness | 4 | 0 | 0 | 4 | 0,500 (0,500–0,500) | 0,500 (n=4) | chưa xác định |
| safety | 4 | 0 | 3 | 1 | 0,125 (0,000–0,375) | 0,167 (n=3) | 0,667 |

Trên held-out, DPO không thắng SFT một cách có ý nghĩa thống kê: win rate là
0,5 và khoảng tin cậy chứa 0,5. Tỷ lệ câu dài hơn thắng là 0,429, còn
mean length của DPO là 582,14 ký tự so với 588,82 của SFT, nên kết quả này
không giống một trường hợp DPO chỉ hack bằng cách viết dài hơn. Ở tám câu
kiểm tra cố định, helpfulness đều hòa; safety nghiêng về SFT với 3 thắng và
1 hòa, nhưng số mẫu quá nhỏ để kết luận rộng. Reward-model sanity accuracy
chung là 1,0; riêng judge Qwen3 là 0,5 còn judge Llama là 1,0. Vì vậy cần
thận trọng: kết quả hội đồng có sự khác biệt giữa hai judge, và judge Qwen3
không đạt ngưỡng 0,8 trên bộ sanity. Hai judge có agreement 0,8276 trên các
cặp hợp lệ. Kết luận phù hợp nhất là run DPO này chưa chứng minh được ưu thế
so với SFT, đặc biệt trên safety.

## 5. Đánh đổi theo β

Không chạy beta-sweep. Với β nhỏ hơn, mô hình có thể rời reference mạnh hơn
và margin tăng nhanh hơn nhưng cũng dễ overfit; với β lớn hơn, cập nhật thận
trọng hơn và kết quả có thể ổn định hơn nhưng margin có thể nhỏ hơn.

## 6. Một quyết định quan trọng nhất

Quyết định quan trọng nhất là dùng β = 0,1 trên GPU T4 thay vì dùng β nhỏ hơn
để ép mô hình ưu tiên chosen mạnh hơn. Phương án thay thế hợp lý là β = 0,05:
nó có thể làm reward gap tăng nhanh hơn, nhưng cũng làm policy rời mô hình
tham chiếu SFT nhiều hơn và tăng nguy cơ overfit hoặc suy giảm safety. β = 0,1
là lựa chọn cân bằng vì mô hình vẫn được regularize quanh reference, phù hợp
với giới hạn thời gian và VRAM của T4, đồng thời cho phép precompute reference
log probabilities. Kết quả thực tế xác nhận DPO đã học được tín hiệu preference
ở mức nhỏ: reward gap train là 0,0873 và margin held-out là 0,0807, đều dương.
Tuy nhiên, phần đánh giá không cho thấy ưu thế rõ ràng: held-out có 7 DPO
thắng, 7 SFT thắng và 36 hòa; khoảng tin cậy win rate 0,430–0,570 bao gồm
0,5. Điều này làm mình bất ngờ vì loss giảm từ 0,6948 xuống 0,6753 nhưng
điểm judge không tăng tương ứng. Nếu làm lại, mình sẽ chạy beta-sweep với
0,05, 0,1 và 0,5, đồng thời kiểm tra riêng các cặp safety và chất lượng
chosen trước khi huấn luyện. Mình cũng sẽ xem xét RPO hoặc DPO-normalized để
giảm ảnh hưởng của độ dài và dùng một judge khác họ mô hình để giảm bias.

## 7. Bộ đo chuẩn

Không chạy NB6.

## 8. Biến thể loss

Không chạy NB3b.

## 9. GRPO

Không chạy NB7.

## Điều bất ngờ nhất

Loss DPO giảm và reward margin held-out dương, nhưng win rate held-out vẫn
đúng bằng 0,5 với rất nhiều lượt hòa. Điều này cho thấy cải thiện của objective
không tự động chuyển thành cải thiện rõ ràng trong đánh giá đầu ra.
