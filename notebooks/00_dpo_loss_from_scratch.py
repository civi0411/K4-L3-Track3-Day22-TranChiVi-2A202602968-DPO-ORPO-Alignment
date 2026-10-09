# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB0 — DPO loss tự cài từ đầu (CPU, ~10 phút)
#
# **Học viên:** Trần Chí Vĩ  
# **Mã học viên:** 2A202602968  
# **Khoá:** VinUni AICB Track 3 (K4) — Day 22 DPO/ORPO Alignment  
#
# **Không cần GPU.** Trước khi gọi `DPOTrainer`, bạn tự viết loss và kiểm tra nó
# trên số liệu đồ chơi. Phần này lấy từ lab K3 (tự cài DPO) và là nền để đọc
# đường cong reward ở NB3.
#
# Bạn sẽ thấy:
# 1. Tại bước 0 (mô hình đang học (policy) = reference) loss luôn bằng `log 2 ≈ 0.693`.
# 2. Gradient của DPO bị nhân với `sigmoid(-margin)`: cặp đã phân biệt tốt gần như không còn được học.
# 3. **Likelihood displacement**: loss vẫn giảm khi log-prob của *chosen* giảm, miễn rejected giảm nhanh hơn.
# 4. IPO, RPO, SimPO, ORPO khác DPO ở đâu, trên cùng một bộ số.

# %%
import sys
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "lab22" / "config.py").exists())
sys.path.insert(0, str(ROOT))

import math

import torch

from lab22 import dpo_math as M

torch.manual_seed(0)

# %% [markdown]
# ## 1. Log-prob của một câu trả lời
#
# `log π(y|x) = Σ_t log π(y_t | x, y_<t)`, chỉ cộng trên token của câu trả lời
# (mask = 1), không cộng trên câu hỏi.

# %%
vocab, length = 8, 5
logits = torch.randn(1, length, vocab)
labels = torch.randint(0, vocab, (1, length))
mask = torch.tensor([[0, 0, 1, 1, 1]])  # 2 token prompt, 3 token trả lời
total, mean = M.sequence_logps(logits, labels, mask)
print(f"sum log p = {total.item():.3f}   mean log p = {mean.item():.3f}")

# %% [markdown]
# ## 2. Bài tập: tự viết DPO loss
#
# Công thức (Rafailov et al. 2023):
#
# $$\mathcal{L} = -\log\sigma\Big(\beta\big[(\log\pi_\theta(y_w) - \log\pi_{ref}(y_w)) - (\log\pi_\theta(y_l) - \log\pi_{ref}(y_l))\big]\Big)$$
#
# Điền hàm dưới đây. Ô kiểm tra sẽ so với bản tham chiếu trong `lab22/dpo_math.py`.


# %%
def my_dpo_loss(pc, pr, rc, rr, beta=0.1):
    """pc/pr: policy log-prob chosen/rejected; rc/rr: reference. Trả về loss trung bình."""
    chosen_reward = beta * (pc - rc)
    rejected_reward = beta * (pr - rr)
    loss = -torch.nn.functional.logsigmoid(chosen_reward - rejected_reward)
    return loss.mean()


# %%
pc, pr = torch.tensor([-12.0, -30.0]), torch.tensor([-15.0, -28.0])
rc, rr = torch.tensor([-13.0, -29.0]), torch.tensor([-14.0, -29.0])
ref_loss, _, _ = M.dpo_loss(pc, pr, rc, rr, beta=0.1)
mine = my_dpo_loss(pc, pr, rc, rr, beta=0.1)
if mine is None:
    print(f"Chưa cài my_dpo_loss. Đáp số tham chiếu: {ref_loss.item():.4f}")
else:
    assert torch.allclose(torch.as_tensor(mine), ref_loss, atol=1e-6), (mine, ref_loss)
    print(f"✓ Khớp tham chiếu: {ref_loss.item():.4f}")

# %% [markdown]
# ## 3. Bước 0: mô hình đang học (policy) = reference ⇒ loss = log 2
#
# NB3 khởi tạo mô hình đang học (policy) bằng chính mô hình SFT (LoRA mới có trọng số B = 0), nên
# reward ngầm định ban đầu bằng 0 và loss bắt đầu ở 0.693. Nếu log của bạn
# không bắt đầu gần 0.693, reference đang không phải mô hình SFT.

# %%
same = torch.tensor([-20.0, -35.0])
loss0, cr0, rr0 = M.dpo_loss(same, same - 3, same, same - 3)
print(f"loss at init = {loss0.item():.4f}   log 2 = {math.log(2):.4f}   rewards = {cr0.tolist()}, {rr0.tolist()}")

# %% [markdown]
# ## 4. Trọng số gradient = sigmoid(−margin)

# %%
for margin in (-2.0, 0.0, 2.0, 5.0):
    m = torch.tensor(margin, requires_grad=True)
    loss = -torch.nn.functional.logsigmoid(m)
    loss.backward()
    print(f"margin {margin:+.1f}: loss {loss.item():.3f}   |dL/dmargin| {abs(m.grad.item()):.3f}")

# %% [markdown]
# ## 5. Likelihood displacement bằng số
#
# Hai kịch bản đều làm margin tăng 2 nat. Loss giống hệt nhau, nhưng ở kịch
# bản B log-prob của câu *được chọn* lại giảm. DPO không phân biệt được hai
# trường hợp này; chỉ đường cong `rewards/chosen` ở NB3 cho bạn biết.

# %%
ref_c, ref_r = torch.tensor([-20.0]), torch.tensor([-22.0])
scenarios = {
    "A: chosen ↑, rejected ↓": (ref_c + 1, ref_r - 1),
    "B: chosen ↓, rejected ↓↓": (ref_c - 3, ref_r - 5),
}
for name, (pc_, pr_) in scenarios.items():
    loss, cr, rj = M.dpo_loss(pc_, pr_, ref_c, ref_r, beta=1.0)
    print(f"{name:28s} loss {loss.item():.3f}  reward chosen {cr.item():+.1f}  rejected {rj.item():+.1f}")

# %% [markdown]
# **RPO** thêm NLL của câu chosen vào loss: kịch bản B bị phạt vì chosen bị đẩy xuống.

# %%
for name, (pc_, pr_) in scenarios.items():
    nll = -pc_ / 10  # NLL trung bình trên 10 token
    print(f"{name:28s} RPO loss {M.rpo_loss(pc_, pr_, ref_c, ref_r, nll, beta=1.0).item():.3f}")

# %% [markdown]
# ### [Trần Chí Vĩ - 2A202602968] Trả lời câu hỏi Rubric NB0 (4 điểm):
# **Vì sao margin tăng được trong khi log-xác suất của câu chosen giảm?**
#
# 1. **Bản chất toán học của Implicit Reward & Margin:**
#    Hàm loss của DPO tối ưu hóa trực tiếp dựa trên hiệu số reward ngầm giữa hai câu trả lời:
#    $$\text{Margin} = r_\theta(x, y_w) - r_\theta(x, y_l) = \beta \left[ \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right]$$
#    Viết gọn lại theo độ biến thiên log-prob:
#    $$\text{Margin} = \beta \Big[ \underbrace{(\log\pi_\theta(y_w|x) - \log\pi_{\text{ref}}(y_w|x))}_{\Delta \log p(w)} - \underbrace{(\log\pi_\theta(y_l|x) - \log\pi_{\text{ref}}(y_l|x))}_{\Delta \log p(l)} \Big]$$
#    Vì mô hình tham chiếu $\pi_{\text{ref}}$ là cố định, Margin chỉ phụ thuộc vào **hiệu số** $(\log\pi_\theta(y_w|x) - \log\pi_\theta(y_l|x))$, chứ **hoàn toàn không ràng buộc giá trị tuyệt đối** của $\log\pi_\theta(y_w|x)$ phải tăng.
#    - Khi $\Delta \log p(w) < 0$ (tức là log-xác suất của câu `chosen` giảm), nhưng $\Delta \log p(l) < 0$ và có độ giảm lớn hơn rất nhiều (tức là $\Delta \log p(l) \ll \Delta \log p(w) < 0$), thì hiệu số $[\Delta \log p(w) - \Delta \log p(l)]$ vẫn mang giá trị **dương lớn**!
#    - Kết quả là $\text{Margin}$ vẫn tăng đều đặn, hàm loss $-\log\sigma(\text{Margin})$ vẫn giảm mượt mà, mặc dù mô hình đang giảm xác suất sinh ra của cả hai câu.
#
# 2. **Cơ chế gradient và hiện tượng 'Unlearning':**
#    - Gradient của DPO tỷ lệ với $\sigma(-\text{Margin})$. Trong không gian xác suất tự hồi quy (autoregressive), việc "dìm" xác suất của câu `rejected` xuống thường dễ hơn nhiều (chỉ cần làm lệch phân bố softmax ở một vài token mấu chốt) so với việc "nâng" xác suất của toàn bộ chuỗi token `chosen` một cách mạch lạc.
#    - Nếu không có cơ chế neo giữ phân bố gốc (như số hạng Language Modeling / NLL của `chosen`), mô hình có xu hướng chọn con đường dìm câu `rejected` cực mạnh để hạ loss nhanh nhất, kéo theo việc xác suất tổng thể của câu `chosen` cũng bị trôi dốc.
#
# 3. **Hệ quả thực tế & Giải pháp:**
#    - Hiện tượng này gọi là **Likelihood Displacement** (Dịch chuyển xác suất). Nó làm tăng margin giả tạo nhưng có thể gây suy giảm chất lượng sinh văn bản thực tế (*mode collapse* hoặc câu trả lời bị cộc lốc/kém tự nhiên).
#    - Do đó, quan sát đường cong reward bắt buộc phải tách riêng `rewards/chosen` và `rewards/rejected` (như thực hiện tại NB3).
#    - Để khắc phục, **RPO** (Regularized Preference Optimization) bổ sung số hạng $-\alpha \log \pi_\theta(y_w|x)$ để phạt nếu xác suất của câu `chosen` bị kéo tụt, buộc mô hình phải duy trì năng lực sinh câu tốt.

# %% [markdown]
# ## 6. Bốn biến thể trên cùng một cặp
#
# | Loss | Cần mô hình tham chiếu (reference)? | Chuẩn hoá độ dài? | Ghi chú |
# |---|---|---|---|
# | DPO (sigmoid) | có | không | mức cơ sở (baseline) |
# | IPO | có | có (TRL chia theo số token) | hồi quy margin về 1/(2β), chống quá khớp khi dữ liệu gần như tất định |
# | RPO | có | không | DPO + NLL(chosen), giảm likelihood displacement |
# | SimPO | không | có | log-prob trung bình + margin γ |
# | ORPO | không | có | NLL(chosen) + λ·log-odds-ratio, gộp SFT và sở thích vào một bước |
#
# NB3b huấn luyện thật các biến thể này (TRL `loss_type` và `trl.experimental.orpo`).

# %%
n_tokens_c, n_tokens_r = 40, 120  # chosen ngắn, rejected dài
pc_, pr_ = torch.tensor([-48.0]), torch.tensor([-130.0])
rc_, rr_ = torch.tensor([-50.0]), torch.tensor([-128.0])
avg_c, avg_r = pc_ / n_tokens_c, pr_ / n_tokens_r
print(f"DPO   {M.dpo_loss(pc_, pr_, rc_, rr_)[0].item():.4f}")
print(f"IPO   {M.ipo_loss(pc_, pr_, rc_, rr_, n_tokens_c, n_tokens_r).item():.4f}")
print(f"SimPO {M.simpo_loss(avg_c, avg_r).item():.4f}")
print(f"ORPO  {M.orpo_loss(avg_c, avg_r, -avg_c).item():.4f}")

# %% [markdown]
# **Câu hỏi cho REFLECTION §3:** tổng log-prob của câu dài luôn âm hơn câu ngắn.
# Vì sao điều đó khiến DPO gốc dễ thiên vị độ dài, và SimPO/ORPO xử lý bằng cách nào?
# Gợi ý: NB2 in ra tỉ lệ cặp có chosen dài hơn rejected trong dữ liệu tiếng Việt.
#
# ### [Trần Chí Vĩ - 2A202602968] Trả lời phân tích:
# 1. **Nguyên nhân DPO gốc dễ thiên vị độ dài:**
#    - Về mặt xác suất, tổng log-prob của một câu phản hồi là: $\log \pi(y|x) = \sum_{t=1}^{|y|} \log \pi(y_t | x, y_{<t})$.
#    - Do xác suất điều kiện tại mỗi token $\pi(y_t) \le 1$ nên $\log \pi(y_t) \le 0$. Do đó, một câu càng dài ($|y|$ lớn) thì tổng log-prob càng âm sâu (độ lớn tuyệt đối càng lớn).
#    - Trong dữ liệu sở thích (cụ thể ở NB2, tỉ lệ câu `chosen` dài hơn `rejected` chiếm tới **65.9%**), độ chênh lệch tuyệt đối của tổng log-prob giữa hai câu dài thường lớn hơn nhiều so với hai câu ngắn.
#    - Do hàm loss DPO truyền thống lấy trực tiếp hiệu số tổng log-prob mà không chia cho độ dài $|y|$, gradient cập nhật sẽ bị chi phối mạnh mẽ bởi các mẫu có câu trả lời dài. Hệ quả là mô hình học được xu hướng "ăn gian độ dài" (length hack / verbosity bias): chỉ cần nói dài dòng hơn là implicit reward tự động tăng lên.
#
# 2. **Cơ chế xử lý triệt để của SimPO và ORPO:**
#    - **SimPO (Simple Preference Optimization):** Sử dụng hàm mục tiêu dựa trên **log-probability trung bình trên từng token**: $\frac{1}{|y|} \sum_{t=1}^{|y|} \log \pi(y_t | x, y_{<t})$. Phép chuẩn hóa này tước bỏ hoàn toàn lợi thế cộng dồn theo chiều dài câu, đưa hai câu ngắn và dài về cùng một thước đo chuẩn. Đồng thời, SimPO đưa vào margin cố định $\gamma$ để đảm bảo độ tách biệt mong muốn mà không cần tới Reference Model.
#    - **ORPO (Odds Ratio Preference Optimization):** Sử dụng log-odds-ratio giữa `chosen` và `rejected` được tính toán dựa trên xác suất trung bình trên mỗi token, kết hợp cùng SFT loss trong một pha huấn luyện duy nhất. Nhờ vậy, ORPO vừa bảo tồn chất lượng sinh văn bản, vừa loại bỏ hiện tượng thiên vị độ dài mà lại tiết kiệm tối đa VRAM GPU.
