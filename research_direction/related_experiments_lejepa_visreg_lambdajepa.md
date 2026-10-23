# Experiments trong 3 paper: LeJEPA (SIGReg), VISReg, λ-JEPA (SACReg)

Nguồn đã đọc toàn văn:
- LeJEPA: Balestriero & LeCun, arXiv 2511.08544v3 (`research_direction/2511.08544v3.pdf`)
- VISReg: Wu, Balestriero, Levine, arXiv 2606.02572
- λ-JEPA: Demirel, Dominé et al. (ISTA), arXiv 2609.35288v2

"VACReg" không có trong 3 paper. Regularizer của λ-JEPA tên là **SACReg**:
SACReg(R) = ½(‖μ‖² + tr Σ − log det(Σ+εI) − d), áp lên view-center tại cả backbone h và projector z.

"Chỉ figure" = paper không in số, chỉ có hình.

---

## 0. Tổng quan

| | LeJEPA | VISReg | λ-JEPA |
|---|---|---|---|
| Dataset pretrain | IN-1K, IN-100, IN-10, Galaxy10, Flowers102, Food101, CIFAR-10/100 | ImageNette, ImageNet-LT, Galaxy10, IN-1K, IN-22K | IN-1K, IN-100, **K710 20% (video)** |
| Backbone | ~60 kiến trúc timm (ResNet, ViT-T…g, ConvNeXt, Swin, LeViT…), lớn nhất ViT-g 1.8B | ViT-S/8 (bộ nhỏ), ViT-B/16, ViT-L/14 | ViT-S/16, ViT-B/16 (ảnh và video) |
| Eval | linear probe frozen; few-shot; finetune (Galaxy10) | linear probe frozen; finetune; ADE20K linear; OOD; generative (iREPA) | linear probe + kNN (IN-1K); transfer 8 dataset; video attentive probe |
| Video | chỉ hình định tính (Fig. 13) | chỉ PCA 3 frame định tính (Fig. 8) | **có, định lượng (Table 2)** |
| Seed / error bar | 3 seed chỉ ở Galaxy10; còn lại 1 run | không có | 3 seed ở IN-100; IN-1K/video 1 run |

---

## 1. λ-JEPA đánh giá video như thế nào

**Không phải "train IN-1K rồi lấy backbone train SSv2".** Video encoder được pretrain **từ đầu trên video**, sau đó **đóng băng** và đánh giá bằng probe.

**Dữ liệu pretrain.**
- Hợp nhất tập train K400/600/700 (= Kinetics-710).
- Bỏ trùng lặp và bỏ mọi clip có trong tập validation.
- Giữ 20% mỗi lớp, được 120,100 clip.
- LeVJEPA không công bố subset của họ, nên subset này là một lần lấy mẫu khác, cùng kích thước.

**Pipeline.** Giống LeVJEPA (Kuhn et al., 2026), chỉ thay loss.
- Video ViT-S/16 và ViT-B/16.
- 16 frame, stride 2, tubelet 1, drop 95% token.
- AdamW, lr 4e-4, wd 0.04, warmup 12% rồi lr hằng; bf16.
- V = 6 global view 224² mỗi clip.
- Batch 768 (S) và 512 (B).
- Trọng số loss lấy từ run ảnh, không tune lại.
- Đánh giá EMA encoder.
- Train đến 1,085 epoch.

**Probe (frozen encoder, theo V-JEPA / LeVJEPA).**
- **IN-1K và SSv2:** attentive probe gồm 1 khối cross-attention với query học được trên mọi output token, rồi linear; 20 epoch, AdamW lr 1e-3, cosine.
  - IN-1K: mỗi ảnh được lặp lại 16 frame.
  - SSv2: 2 clip × 16 frame × 3 crop.
- **K400:** logistic regression L-BFGS trên CLS đã chuẩn hoá, chọn ridge tốt nhất trong {1e-5, 1e-4, 1e-3}, 1 center crop, full train/val. Lý do: LeVJEPA không công bố cấu hình linear của họ.
- Baseline video (V-JEPA 2, VideoMAEv2, LeVJEPA) **chép số từ paper LeVJEPA**, không chạy lại.

**Table 2 (IN-1K / SSv2 / K400):**

| Cài đặt | Method | IN-1K | SSv2 | K400 |
|---|---|---|---|---|
| ViT-S, 240 ep | LeVJEPA | 39.4 | – | – |
| ViT-S, 240 ep | V-JEPA 2 | 38.7 | – | – |
| ViT-S, 240 ep | **λ-JEPA** | 46.8 | 37.3 | 40.0 |
| ViT-B, 240 ep | VideoMAEv2 | 47.1 | – | – |
| ViT-B, 240 ep | V-JEPA 2 | 51.6 | – | – |
| ViT-B, 240 ep | LeVJEPA | 50.7 | 30.4 | – |
| ViT-B, 240 ep | **λ-JEPA** | 52.9 | 43.9 | 44.2 |
| ViT-B, dài hơn | VideoMAEv2 | 53.4 | 43.6 | 37.4 |
| ViT-B, dài hơn | V-JEPA 2 | 51.6 | 42.5 | 40.7 |
| ViT-B, 1085 ep | LeVJEPA | **61.0** | 40.4 | 44.6 |
| ViT-B, 1085 ep | **λ-JEPA** | 57.1 | **48.3** | **45.7** |

**Lưu ý khi đọc bảng.**
- Lợi thế lớn nhất là SSv2: hơn LeVJEPA +13.5 điểm ở 240 ep và +7.9 ở 1085 ep.
- Ở 1085 ep, IN-1K **thua** LeVJEPA (57.1 so với 61.0).
- Cột K400 không cùng protocol:
  - λ-JEPA dùng linear trên CLS, còn LeVJEPA dùng linear trên mean-pooled.
  - λ-JEPA với mean-pooled linear được 43.4, với attentive probe được 62.0.
- Mỗi ô là 1 run; subset dữ liệu khác LeVJEPA.

---

## 2. LeJEPA (SIGReg)

**Setup chung.**
- Loss: (1−λ)·L_pred + λ·SIGReg (Epps–Pulley).
- Tích phân trên lưới t ∈ [−5, 5] với 17 điểm; slice lấy mẫu lại mỗi step; khuyến nghị 1024 slice.
- AdamW, lr ∈ {5e-3, 5e-4}, wd ∈ {1e-1, 1e-2, 1e-5}; warmup tuyến tính rồi cosine.
- Khuyến nghị λ = 0.05.
- V = 2 global 224 + 6 local 96. Có chỗ khác trong paper ghi V_l = 8.
- Không dùng predictor, teacher hay stop-grad.

### 2.1 Thí nghiệm lý thuyết / synthetic

| Thí nghiệm | Kết quả |
|---|---|
| Fig. 3: variance của linear probe, isotropic vs anisotropic (2D, 100 điểm) | Var(β̂) 0.0056 vs 0.0801 (κ = 20) |
| Fig. 17, 18: bias OLS/Tikhonov, N ∈ {50, 200, 1000}, κ ∈ {1, 5, 10, 50} | chỉ figure |
| Fig. 5, 6, 16: phân phối "X" 2D nhúng trong R^128 và R^1024; so 6 thống kê (VCReg, Ext-JB, CvM, Watson, AD, Epps–Pulley) | chỉ figure; Epps–Pulley khôi phục được Gaussian, các thống kê dựa trên moment thì không |
| Fig. 7: slice ngẫu nhiên vs cố định | slope −2.79 vs −285.91; 16 slice ngẫu nhiên hơn hàng nghìn slice cố định |
| Fig. 20: sai số quadrature | chỉ figure |
| Table 6: thời gian SIGReg (V100) | 0.47 ms (N = 512, M = 512) đến 26 ms (N = 32768) |

### 2.2 Độ ổn định hyperparameter

**Table 7 / Fig. 8: λ × số view, ResNet-50, IN-100, 400 ep.**
- 76.7 đến 84.7 trên toàn lưới λ ∈ [0.001, 0.5].
- Tốt nhất 84.68 (V = 4, λ = 0.02).

**Table 1: ViT-L/14, IN-1K, 100 ep, linear probe.**

| Sweep | Kết quả |
|---|---|
| (a) miền tích phân × số slice × số điểm quadrature | 71.8–75.0 |
| (b) số view | V_g = 1 kém (53–69); V_g = 2 hoặc 4 cho 72–75.1 |
| (c) batch 128 / 256 / 512 / 1024 | 72.2 / 74.2 / 74.7 / 74.1 |
| (d) proj dim 64 → 1024 | 75.65 → 73.65; proj dim nhỏ tốt hơn |
| (e) register token | ảnh hưởng ≤ 0.3 điểm |

**Fig. 9: kiến trúc.** Khoảng 50 kiến trúc timm < 20M tham số, 8 họ, trên IN-10. Tất cả đạt 91.5–95%; số từng model chỉ có trong figure.

**Table 4: bỏ heuristic.** IN-100, 400 ep; predictor × SWA × projector 1/2/3 lớp; RN50, ViT-S/8, ViT-T/8.
- Không ô nào collapse.
- Projector sâu hơn cho +4 đến +9 điểm.
- SWA giúp ViT (tối đa +3); predictor không giúp.
- Tốt nhất: RN50 83.93, ViT-S/8 84.12.

### 2.3 Loss ↔ accuracy (Fig. 10, 11, 19, 21)

- Spearman giữa train loss và accuracy là 0.60–0.90.
- Chia loss cho λ^0.4 thì tăng lên 0.93–0.99.
- Tổ hợp model–dataset được dùng: RN18 / Flowers, RN50 / Galaxy10 / IN-10, ViT-S/8 / Galaxy10 / IN-10, ViT-B / IN-1K.

### 2.4 In-domain vs foundation model

**Table 3 / Fig. 12: Galaxy10 (11k ảnh), 400 ep, 3 seed, k-shot ∈ {1, …, all}.**

| | ConvNeXtV2-N | ResNet-34 | DINOv2-S | DINOv3-S |
|---|---|---|---|---|
| Frozen, all | 76.52 | **78.17** | 67.62 | 71.38 |
| Finetune, all | 82.72 | **83.28** | 78.34 | 81.60 |

**Table 5: in-domain từ đầu (frozen linear), so với I-JEPA ViT-H 630M.**

| Dataset | LeJEPA tốt nhất | I-JEPA ViT-H |
|---|---|---|
| Flowers102 | 82.19 | 85.76 |
| CIFAR-100 | 70.44 | 86.93 |
| Food101 | 78.40 | 81.06 |
| IN-10 | 92.87 | 98.65 |
| CIFAR-10 | 93.16 | 97.77 |
| Galaxy10 | **77.29** | 62.93 |

LeJEPA chỉ thắng trên Galaxy10.

### 2.5 Scale trên IN-1K

- Online linear probe: ViT-L 77.1, ConvNeXtV2-H 78.5.
- "ViT-H/14 79%" chỉ xuất hiện trong abstract.
- ViT-g 1.8B: chỉ có đường cong ổn định (figure).

**Table 2: few-shot transfer từ IN-1K 100 ep, frozen, 8 dataset (DTD, Aircraft, Cars, C10, C100, Flowers, Food, Pets). Average:**

| Model | 1-shot | 10-shot | all |
|---|---|---|---|
| LeJEPA ViT-L | 29.55 | 60.95 | 79.48 |
| LeJEPA ConvNeXtV2-H | 31.58 | 60.70 | 77.56 |
| I-JEPA ViT-H 300 ep | 30.20 | 60.51 | 78.50 |
| I-JEPA ViT-H + STOP | **32.05** | **62.92** | **80.70** |
| I-JEPA ViT-H IN-22K 900 ep | 35.15 | 63.28 | 80.82 |

### 2.6 Định tính

- PCA feature (Fig. 1, Fig. 14).
- Segmentation/tracking trên video bằng ngưỡng hoá attention của CLS (Fig. 13): không nêu dataset, không có metric.
- **Không có thí nghiệm video định lượng**, không segmentation/depth, không kNN, không attentive probe.

**Điểm yếu cần biết khi trích dẫn.**
- Không nêu recipe của linear probe.
- Gần như mọi kết quả là 1 run.
- Số liệu không nhất quán: V = 8 hay 10; ViT-B/8 hay B/14; I-JEPA ghi IN-1K hay IN-22K; 79% chỉ có trong abstract.
- Câu "consistently outperform I-JEPA" không đúng khi so với I-JEPA + STOP.

---

## 3. VISReg

**Loss.** (1−λ)·L_pred + λ·(scale + sliced-W2 shape + center), stop-grad trên σ. Chỉ có ảnh.

### 3.1 Bộ ViT-S/8 128 px (single H100)

**ImageNette (800 ep, batch 256, lr 1e-3 không decay, 4 global view, λ 0.6, online probe):**
- Fig. 4 (proj dim): SIGReg collapse còn 32.25 ở D = 2048.
- Fig. 5 (số slice K): SIGReg hỏng ở K = 32, 64, 256.
- Fig. 6 (1 / 4 / 8 GPU):

  | | 1 GPU | 4 GPU | 8 GPU |
  |---|---|---|---|
  | SIGReg | 75.26 | 80.59 | 88.87 |
  | SWD | 87.85 | 89.38 | 89.72 |
  | VISReg | 88.08 | 89.97 | 90.30 |

**Table 1: ImageNet-LT (400 ep, lr 1e-4).** Overall / many / medium / few:

| Method | Overall | Many | Medium | Few |
|---|---|---|---|---|
| SWD | 31.85 | 51.54 | 22.70 | 8.36 |
| SIGReg | 32.00 | 51.86 | 22.88 | 7.92 |
| VISReg | 32.11 | 51.55 | 23.19 | 8.52 |
| **VISReg\*** | **35.14** | **54.49** | **26.87** | **9.40** |
| VICReg | 33.08 | 52.29 | 24.63 | 8.54 |
| DINO | 5.13 | 12.22 | 0.82 | 0.24 |

**Table 2: Galaxy10 in-domain (400 ep).** SWD 80.60, SIGReg 80.50, VISReg 80.51, VISReg\* 80.76, VICReg 79.93, DINO 73.49.

**Phụ lục:**
- Table 10 (knockout, ImageNette): bỏ scale 20.80, bỏ shape 33.42, bỏ center 91.41, đầy đủ 91.82.
- Table 11 (detach, ImageNette / IN-LT / Galaxy10): có detach 90.89 / 31.94 / 79.44, không detach 90.62 / 31.59 / 78.91.
- Table 12 (tỉ lệ trọng số): shape 4:1 cho IN-LT 35.14 và Galaxy10 80.76.
- Fig. 7: tương quan loss–accuracy r = −0.996.

### 3.2 IN-1K ViT-B/16

**Table 3 (100 ep; riêng cột proj dim là 300 ep):**

| Sweep | Giá trị → accuracy |
|---|---|
| λ | 0.7 → 67.98, 0.8 → 68.64, 0.9 → 69.19, 0.95 → 69.04 |
| lr | 1e-4 → 65.73, 3e-4 → 68.85, 5e-4 → 69.19, 7e-4 → 69.28, 9e-4 → 69.66 |
| batch | 128 → 67.63, 256 → 69.43, 512 → 69.66, 1024 → 69.45, 2048 → 68.79 |
| proj dim (300 ep) | 64 → 73.20, 128 → 73.34, 256 → 73.44, 512 → 73.53 |

**Table 4:** proj dim so với 11 task downstream, trong đó có Galaxy10, ChestXRay AU-ROC và ADE20K.

**Config chính.**
- VISReg-B: ViT-B/16, 400 ep, 4 global + 6 local, lr 9e-4, λ 0.9, d_p 256, K 2048, batch 512, 32 H100, khoảng 1,120 GPU-h.
- VISReg-L: ViT-L/14, λ 0.7, khoảng 2,060 GPU-h.
- VISReg-IN22K: ViT-L/14, 100 ep, khoảng 2,720 GPU-h.

**Protocol.**
- Linear probe trên CLS của 4 layer cuối, 10 ep, lưới 13 lr, báo cáo best test accuracy (theo DINOv2).
- IN-1K: 10 head song song, 100 ep.
- ADE20K: conv 1×1 tuyến tính, 40 ep.
- Finetune theo protocol DINO.

**Kết quả.**

| Bảng | Kết quả |
|---|---|
| Table 5 (transfer) | VISReg-B 400 ep: avg 79.1, IN-1K 75.7; VISReg-L: 78.5 / 77.0 |
| Table 6 (OOD: DTD, Galaxy10, AID, ChestXRay, RetinaMNIST, OrganAMNIST) | VISReg-B 70.19, L 70.63, IN22K 72.94 so với DINOv2 72.93 |
| Table 7 (finetune C10 / C100 / Flowers / IN-1K / Galaxy10) | VISReg 99.2 / 91.8 / 99.0 / 83.0 / 87.0, DINO 99.1 / 91.7 / 98.8 / 82.8 / 86.6 |
| Table 8 (ADE20K linear mIoU) | VISReg 30.16; MoCoV3 31.69, DINO 29.40, MAE 23.60, data2vec 21.99 |
| Table 9 (SiT-B/2 iREPA: IS / gFID / P / R) | VISReg 33.48 / 40.36 / 51.38 / 61.26, DINO 33.47 / 41.15 / 50.51 / 60.70 |

**Video:** không có, chỉ PCA 3 frame định tính.

**Điểm yếu cần biết khi trích dẫn.**
- Không có seed hay error bar.
- Số view và K không nhất quán giữa các phần.
- ImageNette báo cáo cả 91.82 lẫn 90.89 cho cùng config.
- ChestXRay được báo cáo trên hai thang khác nhau.

---

## 4. λ-JEPA (SACReg)

**Loss.** L = L_SSL + β_h·SACReg(H̄) + β_z·SACReg(Z̄).
- Dùng slice trực chuẩn ngẫu nhiên và ring buffer.
- Cơ sở lý thuyết: λ-balance âm giữa các layer thì encoder có full rank.

### 4.1 Lý thuyết / toy

- Mạng tuyến tính 2 lớp (Fig. 3).
- ReLU teacher–student với w_reg ∈ {0, …, 0.1}: effective rank và NTK distance (Fig. 4).
- Table 3: tỉ lệ effective rank theo λ ∈ {−1, …, −0.25}, tăng từ 0.467 lên 0.58–0.64 tại step 20k.
- Fig. 5: CKA và NTK drift của ViT trên IN-100.

### 4.2 Nghiên cứu có kiểm soát trên IN-100

**Setup.**
- ViT-S/16, 100 ep, batch 128, V = 4, **3 seed**.
- Thêm SACReg tại backbone cho 6 phương pháp: LeJEPA, VICReg, VISReg, SimCLR, DINO, BYOL.

**Kết quả.**
- RankMe tăng; kNN và linear đều tăng. **Số chỉ có trong figure** (Fig. 2, Fig. 7).
- Augmentation thickness Θ_h: độ tách tâm lớp cải thiện ở 96/100 lớp.

**Table 5 (LeJEPA → thêm SIGReg tại h):**

| Metric | LeJEPA | + SIGReg tại h |
|---|---|---|
| pos cos | 0.96 | 0.73 |
| neg cos | 0.58 | 0.01 |
| KL diag | 0.96 | 0.07 |
| Linear | 60.4 | 58.9 |
| kNN | 52.4 | 48.1 |

Gaussian hoá trực tiếp backbone làm tệ hơn; SACReg thì không.

### 4.3 IN-1K

**Setup.**
- ViT-S/16 và ViT-B/16 ở 224, 100 và 400 ep.
- Projector 2048 → 256 + BN; batch 128 trên 2 GPU; 6 global view (crop 30–100%), không có local crop.
- AdamW lr 1e-3, wd 0.05, warmup 10 ep rồi cosine.

**Protocol.**
- Linear probe kiểu Lightly/MAE: BN + linear trên CLS, 90 ep.
- kNN với k = 200, T = 0.07.
- Transfer theo protocol VISReg trên 8 dataset.

**Table 1 + Table 4 (IN-1K linear / transfer avg):**

| Backbone, ep | LeJEPA | DINO | iBOT | VISReg | **λ-JEPA** |
|---|---|---|---|---|---|
| ViT-S 100 | 62.5 / 68.7 | 70.0 / 75.7 | 70.9 / 75.9 | – | 69.7 / **77.6** |
| ViT-S 300–400 | 66.4 / 71.9 | 73.8 / 79.4 | **75.2 / 79.5** | – | 72.3 / 79.3 (400 ep) |
| ViT-B 100 | 69.7 / 74.2 | 73.9 / 78.3 | **77.0** / 80.7 | 70.3 / 76.4 | 74.2 / **80.9** |
| ViT-B 300–400 | 72.4 / 75.2 (300) | 77.2 / 83.1‡ | **78.5 / 83.2‡** | 74.6 / 79.1‡ | 76.0 / 82.0 |

‡ = số lấy từ paper VISReg. Các baseline còn lại là checkpoint OK-AI.

**Transfer từng dataset, λ-JEPA ViT-B 100 ep:** DTD 71.5, Aircraft 59.8, Cars 71.1, C10 96.7, C100 84.9, Flowers 92.2, Food 79.4, Pets 91.7.

### 4.4 Video

Xem mục 1.

**Giới hạn tự nêu.**
- Không có SACReg ở patch-level hay local crop (thử thì không ổn định).
- Không có dense task (segmentation, depth).
- IN-1K 400 ep và video không tune lại.

---

## 5. Hàm ý cho paper của mình

- **Bộ nhỏ ViT-S/8 128 px (ImageNette / IN-LT / Galaxy10)** của VISReg là protocol rẻ nhất có số tham chiếu cho cả SIGReg và VISReg. Đó cũng là bộ mình đang chạy.
- λ-JEPA có controlled study trên IN-100 với 3 seed. Đây là mẫu tốt cho dạng "thêm X vào N objective". Riêng câu hỏi residual-parameterization (NeuTRENO × regularizer) cũng có thể đóng khung theo cách đó.
- Video theo protocol LeVJEPA cần pretrain từ đầu trên 120k clip Kinetics, 240–1085 ep. Chi phí vượt xa ngân sách GPU hiện tại.
- Một cách rẻ hơn có thể làm: lấy encoder ảnh, gắn attentive probe trên SSv2 với ảnh lặp 16 frame. Nhưng **không có số tham chiếu nào trong 3 paper cho cài đặt đó**.
