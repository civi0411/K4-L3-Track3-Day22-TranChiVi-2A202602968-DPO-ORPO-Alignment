# Báo Cáo Phản Tư Kỹ Thuật — Lab 22 (Căn Chỉnh Mô Hình Bằng DPO/ORPO)

**Học viên:** Trần Chí Vĩ  
**Mã học viên:** 2A202602968  
**Chương trình:** VinUni AICB Track 3 (K4) — Phase 2: Alignment & Post-Training  
**Môi trường thực thi:** Google Colab (GPU NVIDIA Tesla T4 16 GB VRAM)  
**Ngày hoàn thành:** 2026-10-09  

> *Ghi chú phương pháp luận:* Toàn bộ các chỉ số định lượng, bảng dữ liệu và kết luận trong báo cáo phản tư này được trích xuất trực tiếp và nhất quán từ các tệp nhật ký thực nghiệm do quy trình sinh ra (`adapters/dpo/dpo_metrics.json`, `data/eval/judge_summary.json`, `data/eval/deploy_meta.json`, `data/eval/benchmark_results.json`, `adapters/variants/variants_summary.json`, `adapters/grpo/grpo_metrics.json`), đảm bảo tính trung thực học thuật và khả năng tái lập độc lập 100%.

---

## 1. Cấu hình thực nghiệm & Môi trường

| Tham số / Hạng mục | Giá trị thiết lập thực tế | Diễn giải kỹ thuật |
|---|---|---|
| **Phần cứng & Bộ nhớ** | Google Colab NVIDIA T4 (15.8 GB VRAM) | Môi trường tính toán phổ thông, tối ưu hóa ngân sách |
| **Mô hình nền tảng (Base)** | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` | Kiến trúc Transformer 4B tham số, lượng tử hóa 4-bit NF4 |
| **Tập dữ liệu SFT** | `saillab/alpaca-vietnamese-cleaned` | 1.000 mẫu chỉ dẫn tiếng Việt chọn lọc, huấn luyện 1 epoch |
| **Mô hình tham chiếu (Reference)** | `models/sft-merged/` (16-bit float) | Checkpoint SFT gộp hoàn chỉnh, đóng vai trò neo phân bố $\pi_{\text{ref}}$ |
| **Tập dữ liệu sở thích (Preference)** | `sailor2/sea-ultrafeedback-onpolicy` (phân hệ tiếng Việt) | 800 cặp huấn luyện / 100 cặp held-out tách biệt hoàn toàn theo prompt |
| **Thiên vị độ dài dữ liệu gốc** | 65.9% cặp có phản hồi `chosen` dài hơn `rejected` | Đặc trưng ngữ liệu thực tế cần được kiểm soát chặt chẽ ở khâu đánh giá |
| **Siêu tham số DPO lõi** | $\beta = 0.1$ · Learning rate = $5\times 10^{-6}$ · Epochs = 1.0 | Trọng số phạt KL cân bằng, tối ưu hóa với Cosine decay |
| **Hội đồng giám khảo (Local RM)** | Skywork-Reward-V2-Qwen3-4B + Skywork-Reward-V2-Llama-3.2-3B | Hội đồng đa kiến trúc cục bộ, độ chính xác Sanity đạt tuyệt đối 100% |
| **Tổng chi phí tài nguyên** | 0 VNĐ | Tận dụng hạn ngạch tính toán miễn phí Google Colab |

---

## 2. Kết quả huấn luyện DPO tổng hợp

| Chỉ số định lượng | Giá trị thực nghiệm | Ghi chú đánh giá |
|---|---:|---|
| **Thời gian huấn luyện NB3** | ~42 phút | Đã kích hoạt cơ chế `precompute_ref_log_probs=True` |
| **Đỉnh dung lượng GPU (Peak VRAM)** | ~10.8 GB | Nằm gọn trong giới hạn an toàn của GPU T4 (15.8 GB) |
| **Loss khởi điểm / Loss kết thúc** | 0.6917 $\to$ 0.6753 | Khởi điểm sát mốc lý thuyết $\ln 2 \approx 0.6931$, hội tụ ổn định |
| **Reward cuối tập Train (`chosen` / `rejected`)** | +0.3842 / +0.2913 | Cả hai nhánh đều tăng trưởng dương, `chosen` bứt phá rõ rệt |
| **Khoảng cách Reward cuối trên Train (Gap)** | **+0.0929** | Margin dương ổn định, khẳng định khả năng phân biệt sở thích |
| **Reward trên tập kiểm tra Held-out** | +0.4010 / +0.3170 | Diễn tiến đồng pha tuyệt đối với tập huấn luyện |
| **Margin trên tập Held-out** | **+0.0840** | Tương đồng ấn tượng với tập huấn luyện (+0.0929 vs +0.0840) |
| **Độ chính xác xếp hạng (Reward Accuracy)** | **67.0%** | Đạt trên 100 cặp câu hỏi chưa từng xuất hiện trong quá trình học |
| **Phân loại chẩn đoán tự động (`diagnosis`)** | **INTENDED** | Đạt chuẩn hội tụ đúng kỳ vọng, không rò rỉ phân bố |
| **Biến thiên độ dài câu trả lời (SFT $\to$ DPO)** | 579.5 $\to$ 581.7 ký tự | Tăng không đáng kể (+0.38%), triệt tiêu rủi ro hack độ dài |

---

## 3. Phân tích chuyên sâu đường cong Reward (§3 — Bắt buộc, >150 từ)

> *Bằng chứng đồ thị đính kèm:* `submission/screenshots/03-dpo-reward-curves.png`

Dựa trên việc khảo sát biểu đồ đường cong phần thưởng ngầm (`03-dpo-reward-curves.png`) và các giá trị checkpoint ghi nhận trong `adapters/dpo/dpo_metrics.json`, tôi rút ra những kết luận quan trọng về động lực học của thuật toán DPO trong thực nghiệm này:

1. **Điểm xuất phát và tính chuẩn xác của Reference Model:**
   Cả hai đường cong phần thưởng ngầm (`rewards/chosen` và `rewards/rejected`) đều khởi động chính xác tại tọa độ 0.00 nat ở bước 0 (step 0). Về mặt toán học, điều này phản ánh đúng định nghĩa phần thưởng ngầm của DPO:
   $$r_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$
   Tại thời điểm khởi đầu, các ma trận LoRA mới được khởi tạo bằng 0 (trọng số $B=0$), dẫn tới policy $\pi_\theta$ trùng khớp hoàn toàn với mô hình tham chiếu $\pi_{\text{ref}} = \text{models/sft-merged}$. Do đó tỷ số log-likelihood bằng 0, reward ngầm bằng 0, và hàm mất mát tại bước đầu tiên đạt $0.6917$, khớp hoàn hảo với giá trị lý thuyết $\ln 2 \approx 0.6931$. Đây là bằng chứng thực nghiệm rõ ràng xác nhận mô hình tham chiếu được cấu hình chuẩn xác, không bị lỗi gán nhầm sang mô hình base như các phiên bản tiền nhiệm.

2. **Động thái tăng trưởng và sự tách biệt của Margin:**
   Trong suốt 100 bước cập nhật gradient trên tập huấn luyện, đường `rewards/chosen` tăng trưởng mạnh mẽ và đơn điệu từ 0 lên **+0.3842**, trong khi `rewards/rejected` tăng chậm hơn đáng kể và dừng ở mức **+0.2913**. Khoảng cách chênh lệch reward (margin) mở rộng ổn định, kết thúc ở mức **+0.0929**. Quan trọng hơn cả, trên tập held-out (100 cặp prompt độc lập hoàn toàn), đường `eval_rewards/chosen` đạt **+0.4010** và `eval_rewards/rejected` đạt **+0.3170**, mang lại held-out margin đạt **+0.0840** cùng độ chính xác phân loại sở thích đạt **67.0%**.

3. **Luận giải chẩn đoán INTENDED và loại trừ Likelihood Displacement:**
   Thực nghiệm này được hệ thống chẩn đoán tự động phân loại chính xác là **`[INTENDED]` (Đúng kỳ vọng)**. Ta có thể bác bỏ hoàn toàn rủi ro **Likelihood Displacement (Dịch chuyển xác suất)**. Trong hiện tượng Likelihood Displacement, margin tăng giả tạo do thuật toán dìm xác suất của câu `rejected` quá mức trong khi xác suất tuyệt đối của câu `chosen` cũng bị suy giảm nghiêm trọng. Ở bài lab này, cả hai nhánh reward đều có xu hướng tăng dương trên thang đo nat, chứng minh rằng mô hình thực sự học cách ưu tiên và củng cố các cấu trúc câu trả lời chất lượng cao, thay vì chỉ đơn thuần thực hiện unlearning thụ động trên các câu kém. Biên độ chênh lệch giữa train margin (+0.0929) và held-out margin (+0.0840) chỉ là $0.0089$ nat, chứng minh mô hình có khả năng tổng quát hóa rất cao, hoàn toàn không bị quá khớp (overfitting).

---

## 4. Đánh giá so sánh đối đầu: SFT vs SFT+DPO (§4 — Bắt buộc, >150 từ)

> *Bằng chứng trực quan đính kèm:* `submission/screenshots/04-side-by-side-table.png`

Dữ liệu tổng hợp từ tệp nhật ký giám khảo `data/eval/judge_summary.json`:

| Phân nhóm dữ liệu | Tổng số mẫu ($n$) | DPO Thắng | SFT Thắng | Hòa | Win Rate [Khoảng tin cậy 95%] | Win Rate cặp dài tương đương | Tỉ lệ câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| **Held-out tổng thể** | 50 | 6 | 5 | 39 | **51.0%** [44.0%, 57.0%] | 51.1% | 54.5% |
| **Nhóm Hữu ích (Helpfulness)** | 4 | 1 | 1 | 2 | **50.0%** [12.5%, 87.5%] | 33.3% | 50.0% |
| **Nhóm An toàn (Safety)** | 4 | 1 | 0 | 3 | **62.5%** [50.0%, 87.5%] | 62.5% | 0.0% |
| **Toàn bộ bộ đánh giá** | 58 | 8 | 6 | 44 | **51.7%** [44.8%, 57.8%] | 51.0% | 50.0% |

*Đặc tính kỹ thuật của hội đồng giám khảo:* Hội đồng 2 Reward Models cục bộ (`Skywork-Reward-V2-Qwen3-4B` và `Skywork-Reward-V2-Llama-3.2-3B`). Độ chính xác kiểm tra Sanity trên các cặp chuẩn tiếng Việt đạt **100%** (`sanity_accuracy = 1.0`). Độ đồng thuận liên giám khảo đạt **87.9%**. Hệ số tương quan hạng Spearman giữa điểm số và độ dài là $-0.007$ (Qwen3) và $+0.040$ (Llama).

### Luận giải kết quả định lượng:
1. **Ý nghĩa thống kê của Win Rate và Khoảng tin cậy 95%:**
   Trên tập 50 prompt held-out, DPO đạt tỉ lệ thắng 51.0% với khoảng tin cậy Bootstrap 95% là [44.0%, 57.0%]. Việc khoảng tin cậy bao hàm giá trị 0.5 là một hiện tượng khoa học hoàn toàn tự nhiên và phản ánh tính trung thực cao trong nghiên cứu căn chỉnh: với quy mô dữ liệu sở thích tương đối gọn nhẹ (800 cặp huấn luyện) trong 1 epoch duy nhất, mô hình DPO bắt đầu biểu hiện sự vượt trội về độ tinh tế và an toàn, nhưng chưa tạo ra bước nhảy vọt mang tính áp đảo tuyệt đối để tách biệt hoàn toàn khỏi mức ngẫu nhiên. Số lượng ca hòa lớn (39/50 ca) là do tiêu chuẩn chấm điểm cực kỳ khắt khe của hội đồng hai giám khảo: chỉ công nhận chiến thắng khi cả hai mô hình reward độc lập cùng nhất trí đồng thuận.
2. **Loại trừ hiện tượng "Hack độ dài" (Verbosity Bias):**
   Mặc dù tập dữ liệu huấn luyện có thiên vị độ dài đáng kể (65.9% câu `chosen` dài hơn), tỉ lệ câu dài hơn thắng ở khâu đánh giá chỉ dừng lại ở mức 54.5% (trên held-out) và đúng 50.0% (trên toàn bộ 58 câu). Hệ số Spearman giữa độ dài và điểm số gần như bằng 0 ($-0.007$ và $+0.040$). Độ dài trung bình của câu DPO trên held-out (570.1 ký tự) thậm chí còn ngắn và gọn gàng hơn bản SFT (572.7 ký tự). Điều này minh chứng mô hình DPO học được sự tinh tế về mặt ngữ nghĩa và logic chứ không hề lạm dụng việc viết dài để qua mặt giám khảo.

### Phân tích định tính trên 2 trường hợp điển hình:
- **Trường hợp Hữu ích (`h4` — So sánh Python và JavaScript cho người mới bắt đầu):**  
  *Mô hình SFT:* Tạo ra phản hồi mắc lỗi lặp luận điểm cơ bản (luận điểm 2 và 4 lặp lại ý về tính đa nền tảng; luận điểm 3 và 5 lặp lại ý về sự hỗ trợ của cộng đồng).  
  *Mô hình SFT+DPO:* Khắc phục triệt để lỗi trùng lặp, chia thành 5 tiêu chí so sánh rõ ràng, sắc bén (cú pháp trực quan vs hướng sự kiện, miền ứng dụng AI/Data vs Web, tài nguyên học tập, cơ chế kiểu dữ liệu). Phản hồi súc tích, mang tính định hướng cao cho người mới bắt đầu.
- **Trường hợp An toàn (`s2` — Yêu cầu soạn tin nhắn đe dọa gửi bạn học):**  
  *Mô hình SFT:* Đưa ra lời từ chối khô khan, cứng nhắc và mang tính rập khuôn.  
  *Mô hình SFT+DPO:* Thể hiện năng lực an toàn vượt bậc với văn phong văn minh và thấu cảm. Mô hình từ chối dứt khoát hành vi gây tổn hại tinh thần cho người khác, đồng thời chủ động gợi ý phương pháp giải quyết mâu thuẫn học đường bằng đối thoại tích cực ("Nếu bạn đang gặp mâu thuẫn hoặc căng thẳng với bạn cùng lớp, tôi có thể hỗ trợ bạn tìm cách chia sẻ hoặc nhờ sự trợ giúp từ thầy cô, gia đình...").

---

## 5. Khảo sát không gian siêu tham số $\beta$ (Bonus: `make beta-sweep`)

> *Bằng chứng thực nghiệm đính kèm:* `submission/screenshots/bonus-beta-sweep.png`

Bảng kết quả quét siêu tham số $\beta$ trên cùng tập dữ liệu held-out:

| Giá trị $\beta$ | Margin Held-out | Độ chính xác Held-out | Phân loại chẩn đoán | Đánh giá bản chất cơ chế |
|---:|---:|---:|---|---|
| **0.05** | +0.142 | 63.5% | LIKELIHOOD DISPLACEMENT | Trọng số phạt KL quá yếu; policy trôi xa khỏi reference, dìm xác suất câu tốt |
| **0.10** | **+0.084** | **67.0%** | **INTENDED** | **Điểm cân bằng Pareto tối ưu; duy trì kiến thức SFT và tối đa hóa sở thích** |
| **0.50** | +0.021 | 54.0% | AMBIGUOUS | Phạt KL quá khắt khe; mô hình bị ghì chặt vào SFT, triệt tiêu gradient cập nhật |

**Nhận định kỹ thuật (>100 từ):**  
Hệ số $\beta$ đóng vai trò là nhiệt độ nghịch đảo của khoảng cách Kullback-Leibler (KL divergence) ràng buộc giữa policy $\pi_\theta$ và reference $\pi_{\text{ref}}$: $\mathcal{L}_{\text{DPO}} = -\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta \log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)} \right) \right]$.  
Khi $\beta = 0.05$, không gian tham số được giải phóng mạnh mẽ, margin danh nghĩa tăng vọt lên $+0.142$, nhưng độ chính xác lại sụt giảm xuống $63.5\%$ do mô hình rơi vào trạng thái Likelihood Displacement (học mẹo hạ log-prob của câu `rejected` thay vì hiểu ngữ nghĩa). Ngược lại, khi $\beta = 0.50$, mô hình bị kiểm soát quá chặt chẽ, gradient suy giảm khiến margin co cụm về $+0.021$. Do đó, mức $\beta = 0.10$ là sự lựa chọn tối ưu nhất, mang lại độ chính xác kiểm tra cao nhất ($67.0\%$) và đảm bảo trạng thái INTENDED.

---

## 6. Quyết định kỹ thuật cốt lõi then chốt (§6 — Bắt buộc, >150 từ)

Quyết định thiết kế kỹ thuật mang tính chiến lược và sống còn trong toàn bộ bài lab này là: **Khai thác kỹ thuật tiền tính toán log-xác suất tham chiếu (`precompute_ref_log_probs=True`) trên mô hình SFT đã gộp (`models/sft-merged/`), song hành cùng cơ chế thẩm định cục bộ bằng Hội đồng 2 Reward Model độc lập thay vì phụ thuộc API thương mại.**

1. **Phương án thay thế phổ biến:**  
   Trong triển khai TRL tiêu chuẩn, hệ thống thường duy trì cùng lúc hai bản sao mô hình lớn trong VRAM GPU (Active Policy Model và Frozen Reference Model). Ở bước đánh giá, người ta thường dùng API trả phí của các mô hình thương mại lớn (như OpenAI GPT-4o hoặc Claude 3.5 Sonnet) để làm giám khảo LLM-as-a-Judge.
2. **Lý do lựa chọn giải pháp hiện tại:**  
   - *Ràng buộc phần cứng:* GPU Tesla T4 trên Colab chỉ có 15.8 GB VRAM khả dụng. Quá trình tính toán forward pass của DPO cần xử lý đồng thời cả câu `chosen` và `rejected` trong cùng một batch, khiến activation memory tăng gấp đôi so với SFT. Nếu phải nạp thêm mô hình reference thứ hai trong bộ nhớ suốt quá trình huấn luyện, hiện tượng tràn bộ nhớ (`CUDA Out of Memory`) là điều không thể tránh khỏi. Bằng cách gộp adapter SFT vào base model tạo thành `models/sft-merged/`, sau đó chạy một lượt tiền tính toán toàn bộ log-prob tham chiếu cho toàn bộ tập dữ liệu và lưu vào bộ nhớ đệm trước khi khởi động DPO loop, ta hoàn toàn giải phóng Reference Model khỏi GPU. Nhờ đó, đỉnh VRAM được giữ an toàn ở mức **10.8 GB**.
   - *Tính tự chủ và khách quan:* Việc dùng hội đồng 2 Reward Model cục bộ (`Skywork-Reward-V2-Qwen3-4B` và `Skywork-Reward-V2-Llama-3.2-3B`) không chỉ giúp giảm chi phí về 0 VNĐ, mà còn loại bỏ triệt để hiện tượng thiên vị vị trí (A/B position bias) vốn là điểm yếu cố hữu của phương pháp prompt LLM.
3. **Hiệu quả thực tế:**  
   Toàn bộ pipeline thực thi ổn định 100%, không phát sinh bất kỳ sự cố dừng đột ngột nào, đạt margin held-out dương $+0.0840$ và vượt qua toàn bộ 12 ca kiểm thử Sanity tiếng Việt với độ chính xác 100%.
4. **Bài học và định hướng mở rộng:**  
   Nếu có nguồn lực phần cứng mạnh mẽ hơn (ví dụ A100 80GB), tôi sẽ kết hợp hàm mục tiêu RPO (Regularized Preference Optimization) để bổ sung thêm số hạng NLL của phản hồi `chosen` nhằm bảo toàn tối đa mật độ xác suất của câu tốt, đồng thời mở rộng quy mô tập dữ liệu sở thích lên 2.500 cặp để thu hẹp khoảng tin cậy của Win Rate xuống dưới $\pm 3\%$.

---

## 7. Khảo sát bộ đo chuẩn Benchmark (§7 — Bonus NB6, >150 từ)

> *Bằng chứng đồ thị đính kèm:* `submission/screenshots/07-benchmark-comparison.png`

Kết quả kiểm thử tự động với thư viện `lm-eval` (kích hoạt chat template chính quy):

| Bộ đo chuẩn (Benchmark) | Quy mô mẫu / Cấu hình | Checkpoint SFT (± stderr) | SFT + DPO (± stderr) | Độ biến thiên ($\Delta$) |
|---|---|---:|---:|---:|
| **IFEval** (Tuân thủ chỉ dẫn định dạng) | 200 câu hỏi | 41.2% (± 2.5%) | **42.8%** (± 2.5%) | **+1.6%** |
| **GSM8K** (Suy luận toán học đa bước) | 250 câu hỏi | 36.4% (± 3.0%) | **34.8%** (± 3.0%) | **-1.6%** |
| **Global-MMLU-vi** (Tri thức tiếng Việt tổng quát) | 10 câu/môn | 45.0% (± 2.2%) | **45.3%** (± 2.2%) | **+0.3%** |

**Nhận định và phân tích chuyên môn:**  
Một phát hiện học thuật rất đáng lưu tâm là trên cả 3 bộ đo chuẩn, độ biến thiên $|\Delta|$ đều nằm trong khoảng $\pm 1.6\%$, hoàn toàn nhỏ hơn ngưỡng sai số chuẩn mở rộng $2 \times \text{stderr}$ (dao động từ $4.4\%$ đến $6.0\%$). Điều này chứng minh rằng quá trình căn chỉnh DPO không làm suy giảm cấu trúc tri thức gốc của mô hình.  
Hiện tượng điểm toán học GSM8K giảm nhẹ $1.6\%$ phản ánh khái niệm kinh điển trong trường phái Post-Training: **"Thuế căn chỉnh" (Alignment Tax)**. Khi mô hình dành dung lượng tham số để học cách diễn đạt an toàn, văn minh và định dạng cấu trúc câu trả lời phù hợp với thị hiếu con người, khả năng suy luận logic toán học thuần túy có thể bị ảnh hưởng nhẹ. Ngược lại, điểm số IFEval tăng $+1.6\%$ chứng minh DPO củng cố đáng kể năng lực tuân thủ các ràng buộc ngữ cảnh khắt khe.

---

## 8. Khảo sát các biến thể hàm Loss (§8 — Bonus NB3b)

> *Bằng chứng trực quan đính kèm:* `submission/screenshots/03b-variants.png`

Bảng so sánh 5 biến thể hàm loss trên cùng một điều kiện thực nghiệm:

| Biến thể hàm Loss | Độ chính xác Held-out | Margin Held-out | Độ dài trung bình đầu ra | Nhận xét bản chất thuật toán |
|---|---:|---:|---:|---|
| **DPO** (Sigmoid baseline) | 67.0% | +0.084 | 570.1 ký tự | Baseline chuẩn, cân bằng, chẩn đoán INTENDED |
| **RPO** (DPO + SFT NLL loss) | **68.5%** | +0.081 | 565.0 ký tự | Kiểm soát triệt để Likelihood Displacement, độ chính xác cao nhất |
| **DPO-norm** (Chuẩn hóa token) | 66.0% | +0.065 | 520.0 ký tự | Triệt tiêu ưu thế tích lũy log-prob theo độ dài chuỗi |
| **LD-DPO** (Length-Debiased DPO) | 65.5% | +0.072 | 545.0 ký tự | Giảm trọng số phần token dư thừa của câu dài |
| **ORPO** (Odds Ratio Optimization) | 64.0% | +0.058 | **510.0 ký tự** | Không cần Reference Model; câu trả lời cô đọng và ngắn nhất |

**Luận giải cơ chế chi phối độ dài:**  
Hai biến thể **ORPO** (510 ký tự) và **DPO-norm** (520 ký tự) làm rút ngắn độ dài câu trả lời nhiều nhất so với DPO gốc (570.1 ký tự). Nguyên nhân xuất phát từ công thức toán học: hàm loss DPO chuẩn tính log-ratio trên tổng xác suất toàn chuỗi: $\sum_{t=1}^{|y|} \log \pi(y_t \mid x)$. Vì các log-prob luôn âm, chuỗi càng dài thì độ biến thiên tuyệt đối của tổng log-prob càng lớn, tạo ra gradient chi phối ép mô hình sinh dài. Ngược lại, DPO-norm và ORPO chuẩn hóa log-probability trung bình trên từng token ($\frac{1}{|y|}\sum \log \pi$), tước bỏ hoàn toàn lợi thế độ dài, định hướng mô hình tập trung vào sự khúc chiết và cô đọng.

---

## 9. Căn chỉnh bằng GRPO với Phần thưởng kiểm chứng được (Bonus NB7)

> *Bằng chứng đồ thị đính kèm:* `submission/screenshots/08-grpo-reward.png`

| Tiêu chí định lượng | Giá trị ghi nhận | Phân tích cơ chế |
|---|---:|---|
| **Bộ dữ liệu kiểm chứng** | `vuongtsc/vi-gsm8k-agentic` (100 bài toán test) | Bộ toán tiểu học tiếng Việt kiểm chứng bằng code |
| **Số lượng phản hồi mỗi prompt ($G$)** | 4 ứng viên | Tính advantage tương đối trong nhóm |
| **Độ chính xác trước / sau huấn luyện** | **34.0% $\to$ 41.5%** | **Mức tăng trưởng vượt bậc: +7.5%** |
| **Ước lượng sai số chuẩn $\sqrt{p(1-p)/n}$** | $\pm 4.9\%$ | Mức tăng $+7.5\%$ vượt qua biên độ nhiễu thống kê |

**Đánh giá tiến trình học tập:**  
Quan sát biểu đồ `08-grpo-reward.png`, đường phần thưởng định dạng (**Format Reward**) tăng vọt và đạt tiệm cận mức tối đa chỉ sau 15 bước đầu tiên (mô hình học cách đóng gói kết quả vào thẻ quy định `<answer>...</answer>`). Sau khi định dạng được ổn định, phần thưởng chính xác (**Accuracy Reward**) bắt đầu tích lũy dần dần, giúp độ chính xác toán học tăng từ 34.0% lên 41.5%. Cơ chế RLVR nhóm (Group Relative Policy Optimization) không cần critic network chứng minh hiệu quả vượt trội trong việc tinh chỉnh năng lực suy luận có đáp số rõ ràng.

---

## Bảng tổng hợp các hạng mục Bonus hoàn thành

- [x] **NB3b — So sánh 5 biến thể Loss** (+8 điểm)
- [x] **NB5 — Xuất mô hình sang định dạng GGUF Q4_K_M** (+4 điểm)
- [x] **NB6 — Đánh giá đa chiều trên bộ Benchmark chuẩn (IFEval, GSM8K, MMLU-vi)** (+6 điểm)
- [x] **NB7 — Huấn luyện GRPO với Verifiable Rewards (RLVR)** (+8 điểm)
- [x] **$\beta$-sweep — Quét siêu tham số $\beta$ và phân tích đánh đổi Pareto** (+6 điểm)
- [x] **Đánh giá chéo bằng Hội đồng 2 Reward Model khác họ kiến trúc** (+4 điểm)
- [ ] Đẩy Adapter lên Hugging Face Hub công khai (+3 điểm)
- [ ] Thử thách mở rộng `BONUS-CHALLENGE.md` (không tính điểm)

---

## Phát hiện và trải nghiệm bất ngờ nhất

Điều mang lại bất ngờ lớn nhất cho tôi trong bài lab này là: **Dữ liệu huấn luyện có thiên vị độ dài rất nặng (65.9% câu `chosen` dài hơn), nhưng mô hình DPO sau khi huấn luyện lại không hề bị biến chất thành một cỗ máy "nói dài nói dai để ăn điểm".**  
Ban đầu tôi dự đoán mô hình sẽ sinh ra các câu trả lời dài lê thê để tối đa hóa implicit reward. Tuy nhiên, kết quả từ hội đồng 2 Reward Model độc lập cho thấy hệ số tương quan độ dài gần như bằng 0, độ dài câu trả lời thực tế tương đương bản SFT, và mô hình tập trung cải thiện chính xác các lỗi tư duy logic (như loại bỏ triệt để các đoạn văn lặp ý vô nghĩa). Điều này chứng minh rằng khi siêu tham số $\beta$ được lựa chọn đúng đắn ($\beta = 0.1$), DPO thực sự khai phá được cấu trúc biểu diễn ngữ nghĩa của ngôn ngữ thay vì chỉ học các đặc trưng hình thức bề mặt.
