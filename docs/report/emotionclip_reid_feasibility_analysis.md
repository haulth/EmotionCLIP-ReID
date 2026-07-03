# Phân tích khả thi EmotionCLIP-ReID

Ngày lập: 2026-05-01  
Ngày cập nhật bổ sung: 2026-05-02  
Phạm vi: source code CLIP-ReID hiện tại, tài liệu trong `docs/Tài liệu tham khảo`, và graph MCP Grapuco của repository `EmotionCLIP-ReID`.

## 1. Kết luận ngắn

Repo hiện tại là triển khai CLIP-ReID gốc cho bài toán image re-identification, chưa phải EmotionCLIP-ReID cho facial expression recognition (FER). Nền tảng có giá trị nhất là khung huấn luyện 2 giai đoạn của CLIP-ReID: Stage 1 học token văn bản mơ hồ cho từng ID, Stage 2 cố định text side và fine-tune image encoder bằng các ràng buộc phân loại, metric learning và image-to-text.

Phương án EmotionCLIP-ReID trong tài liệu tham khảo là khả thi như một hướng mở rộng nghiên cứu, nhưng không thể triển khai đầy đủ chỉ bằng repo hiện tại. Các phần có thể tận dụng ngay là `PromptLearner`, `TextEncoder`, CLIP visual encoder, train loop hai giai đoạn và cơ chế contrastive/I2T. Các phần còn thiếu lớn là dataset FER, prompt theo lớp cảm xúc/AU, adapter trong ViT, uncertainty/EDL loss, metric FER và protocol đánh giá.

Sơ đồ đi kèm:

- [Mô hình khoa học CLIP-ReID hiện tại](../emotionclip_reid_current_clipreid_model.drawio)
- [Mô hình đề xuất EmotionCLIP-ReID](../emotionclip_reid_proposed_emotionclip_reid_model.drawio)
- [Graph codebase Grapuco](../emotionclip_reid_codebase_graph.drawio)

## 2. Nguồn tham khảo và ý chính

- [CLIP-ReID Exploiting Vision-Language Model for Im.pdf](<../Tài liệu tham khảo/CLIP-ReID Exploiting Vision-Language Model for Im.pdf>): đề xuất chiến lược hai giai đoạn cho ReID khi nhãn chỉ là chỉ số ID, không có nhãn văn bản cụ thể. Stage 1 học text token cho từng ID bằng contrastive loss; Stage 2 dùng text feature cố định làm semantic anchor để fine-tune image encoder.
- [EmotionCLIP-ReID.pdf](<../Tài liệu tham khảo/EmotionCLIP-ReID.pdf>): đề xuất chuyển khung CLIP-ReID sang FER, dùng prompt có cấu trúc theo cảm xúc và AU, adapter nhẹ, và uncertainty calibration để tăng bền vững với occlusion, pose và mẫu mơ hồ.
- [MER-CLIP AU-Guided Vision-Language.pdf](<../Tài liệu tham khảo/MER-CLIP AU-Guided Vision-Language.pdf>): dùng AU/FACS để chuyển nhãn AU thành mô tả văn bản về chuyển động cơ mặt, giúp căn chỉnh visual feature với mô tả vi mô có ý nghĩa giải phẫu.
- [Emotion-aware adaptation of CLIP model for.pdf](<../Tài liệu tham khảo/Emotion-aware adaptation of CLIP model for.pdf>): dùng Expression-aware Adapter để fine-tune CLIP hiệu quả cho FER, đồng thời giữ năng lực tổng quát của CLIP.
- [UA-FER_ Uncertainty-aware representation learning for facial expression recognition.pdf](<../Tài liệu tham khảo/UA-FER_ Uncertainty-aware representation learning for facial expression recognition.pdf>): dùng VLP + Evidential Deep Learning để hiệu chỉnh bất định trong quan hệ ảnh-văn bản và tránh dự đoán quá tự tin trên mẫu khó.

## 3. Hiện trạng codebase CLIP-ReID

Luồng chạy chính nằm ở `train_clipreid.py`. Script này đọc config, tạo dataloader, model, loss, optimizer và chạy tuần tự Stage 1 rồi Stage 2.

Các khối chính:

- `datasets/make_dataloader_clipreid.py`: tạo `train_loader_stage1`, `train_loader_stage2`, `val_loader` từ các dataset ReID như Market1501, DukeMTMC, MSMT17, Occluded-Duke, VeRi, VehicleID.
- `model/make_model_clipreid.py`: định nghĩa `PromptLearner`, `TextEncoder`, `build_transformer`, CLIP image encoder, BNNeck, classifier heads và nhánh forward cho `get_image` hoặc `get_text`.
- `processor/processor_clipreid_stage1.py`: cố định image/text encoder trên thực tế thông qua optimizer chỉ cập nhật `prompt_learner`, trích image feature và học text prompt bằng supervised contrastive loss hai chiều.
- `processor/processor_clipreid_stage2.py`: tạo text feature cố định cho toàn bộ ID, fine-tune image encoder, dùng logits `image_features @ text_features.t()` cùng ID loss, triplet loss và I2T loss.
- `loss/make_loss.py`: kết hợp cross entropy/label smoothing, triplet loss và I2T cross entropy.
- `solver/make_optimizer_prompt.py`: Stage 1 chỉ đưa tham số `prompt_learner` vào optimizer; Stage 2 đóng băng `text_encoder` và `prompt_learner`.

Sơ đồ khoa học hiện tại: [emotionclip_reid_current_clipreid_model.drawio](../emotionclip_reid_current_clipreid_model.drawio)

## 4. Mô hình khoa học hiện tại

CLIP-ReID giải quyết vấn đề ReID không có mô tả text thật bằng cách tạo mô tả mơ hồ có thể học:

```text
Image ReID dataset
  -> CLIP image encoder
  -> image features

ID labels
  -> prompt template "A photo of a [X1][X2]...[XM] person/vehicle"
  -> learnable class tokens per ID
  -> frozen CLIP text encoder
  -> text features
```

Stage 1 tối ưu text tokens để text feature của từng ID tiến gần cụm image feature tương ứng. Stage 2 cố định text tokens và text encoder, rồi fine-tune image encoder để image feature hội tụ quanh semantic anchor đã học. Với inference, model xuất feature ReID bằng cách nối feature trước/sau projection, sau đó evaluator tính CMC/mAP.

Đây là một nền tảng tốt cho EmotionCLIP-ReID vì FER cũng cần ánh xạ ảnh mặt vào không gian ngữ nghĩa. Khác biệt then chốt là nhãn FER có ý nghĩa ngôn ngữ thật như happy, surprise, fear, neutral, trong khi ReID dùng ID index.

## 5. Mô hình đề xuất EmotionCLIP-ReID

Sơ đồ đề xuất: [emotionclip_reid_proposed_emotionclip_reid_model.drawio](../emotionclip_reid_proposed_emotionclip_reid_model.drawio)

Mục tiêu là thay "ID descriptor" bằng "emotion descriptor":

```text
Face image
  -> face-safe preprocessing
  -> CLIP visual encoder + expression-aware adapters
  -> emotion-aware visual feature

Emotion/AU label
  -> structured prompt
  -> frozen CLIP text encoder
  -> fixed emotion descriptors

Training
  -> classification loss
  -> image-text contrastive/I2T loss
  -> optional uncertainty calibration loss
```

Prompt nên chuyển từ dạng ReID:

```text
A photo of a X X X X person.
```

sang dạng FER:

```text
A photo of a person expressing [emotion], characterized by [AU-based learnable tokens].
```

Nếu dataset có AU label, các token có thể được gắn với mô tả FACS như eyebrow raiser, lip corner puller, cheek raiser. Nếu không có AU label, chỉ nên dùng emotion-class prompt learning trước, còn AU-based prompt là hướng nâng cấp.

## 6. Phân loại khả thi triển khai

### Khả dụng trực tiếp

| Thành phần | Lý do | Code nền |
|---|---|---|
| Khung huấn luyện 2 giai đoạn | Đã có Stage 1 prompt learning và Stage 2 image fine-tuning | `train_clipreid.py`, `processor/*stage*.py` |
| `PromptLearner` | Có cơ chế learnable text tokens theo class/ID | `model/make_model_clipreid.py` |
| `TextEncoder` | Bọc CLIP text transformer và projection | `model/make_model_clipreid.py` |
| CLIP visual encoder | Có loader pretrained CLIP và forward ViT/RN50 | `model/clip/*`, `load_clip_to_cpu()` |
| Contrastive/I2T alignment | Đã có `SupConLoss` và logits image-text | `loss/supcontrast.py`, `processor_clipreid_stage2.py` |
| Entry point/config pattern | Có cấu trúc yacs config, optimizer, scheduler | `config/defaults.py`, `configs/*` |

### Khả thi với thay đổi vừa phải

| Thành phần | Việc cần làm | Rủi ro |
|---|---|---|
| Prompt theo lớp cảm xúc | Đổi `num_classes` từ ID sang emotion class, sửa template prompt | Thấp đến vừa |
| FER dataset loader | Thêm loader RAF-DB/AffectNet/FERPlus hoặc dataset được chọn | Vừa, phụ thuộc format data |
| Classification head FER | Thay ReID classifier bằng emotion classifier, output class logits | Thấp |
| FER metrics | Thêm accuracy, balanced accuracy, macro-F1, confusion matrix | Thấp |
| Face-safe preprocessing | Bỏ random erasing/crop mạnh; dùng resize, flip nhẹ, color jitter nhẹ | Thấp |
| Inference FER | Xuất class probability và top-k emotion thay vì CMC/mAP | Vừa |

### Khả thi nhưng công sức cao/rủi ro cao

| Thành phần | Lý do |
|---|---|
| Adapter trong ViT | Cần chèn module trainable vào residual blocks của CLIP ViT mà không phá pretrained weights. |
| EDL/uncertainty calibration | Cần thiết kế head evidence, Dirichlet uncertainty và loss ổn định. |
| AU-based prompt descriptors | Cần AU label thật hoặc pseudo-label đáng tin cậy; nếu không, mô tả AU có thể gây nhiễu. |
| Grad-CAM/attention visualization | Cần chọn layer phù hợp trong CLIP ViT và kiểm tra heatmap có ý nghĩa trên mặt. |
| Benchmark occlusion/pose | Cần dataset hoặc split có nhãn occlusion/góc nhìn, không có sẵn trong repo. |

### Chưa triển khai được từ repo hiện tại

| Thành phần | Lý do |
|---|---|
| Full MER-CLIP video/micro-expression | MER-CLIP dùng video/spatiotemporal encoder và augmentation chuyên biệt, repo hiện tại là image ReID. |
| Full UA-FER MFD/RUC | Cần kiến trúc decoupling global/local affinity và Relation Uncertainty Calibration riêng. |
| Full EA-CLIP IEC/spherical interpolation | Cần module classifier instance-enhanced và thiết kế nội bộ chưa có trong repo. |
| SOTA claim | Chưa có dataset FER, protocol chuẩn, baseline tái lập và evaluation đầy đủ. |

## 7. Phân tích nhánh Action-Units hiện tại

Phần này đọc thêm repository `E:\Source\Action-Units` để hiểu cách nhánh AU đang xử lý dữ liệu đầu vào và cách chuyển CLIP-ReID thành bài toán classifier. Nhìn tổng thể, nhánh này đã đi đúng hướng khi thay nhãn định danh ReID bằng nhãn AU đa nhãn, nhưng đầu ra cảm xúc hiện vẫn là bước suy luận sau bằng luật, chưa phải một dữ kiện trainable tham gia trực tiếp vào dự đoán cảm xúc cuối.

### 7.1. Chuẩn bị dữ liệu DISFA

DISFA là database biểu cảm tự phát, có video stereo của 27 người trưởng thành, nhãn intensity AU theo thang 0-5 cho mọi frame và landmark mặt 66 điểm [DISFA official](https://mohammadmahoor.com/pages/databases/disfa/). Trong repo `Action-Units`, script `prepare_data.py` đang biến dữ liệu thô thành bảng nhãn dễ train:

```text
AUs_DATA/
  Labels/{subject}/{trial}/AU{k}.txt
  Images/{subject}/{trial}/{frame}.jpg
        |
        v
prepare_data.py
  -> đọc từng file AU*.txt
  -> gom 12 AU theo từng frame
  -> intensity >= 2 thành AU active
  -> xuất AUs_DATA/labels.csv
```

Danh sách AU đang dùng là:

```text
[AU1, AU2, AU4, AU5, AU6, AU9, AU12, AU15, AU17, AU20, AU25, AU26]
```

Luật nhị phân hóa:

```text
y_k = 1 nếu intensity_k >= 2
y_k = 0 nếu intensity_k < 2
```

Cách làm này phù hợp với AU occurrence detection, vì nhiều benchmark DISFA cũng dùng ngưỡng intensity để chuyển về bài toán phát hiện AU. Tuy nhiên, với mục tiêu nhận diện cảm xúc chính xác hơn, việc bỏ intensity 0-5 là một mất mát quan trọng. Ví dụ cùng là biểu cảm vui, một người có AU12 mạnh nhưng AU6 yếu, người khác có AU6 và AU12 cùng mạnh; nếu chỉ giữ 0/1 thì mô hình mất thông tin về mức độ co cơ mặt.

### 7.2. Dataset loader và rủi ro chia tập

`datasets/disfa.py` đọc `labels.csv` bằng pandas, tạo mỗi mẫu theo dạng:

```text
(image_path, au_vector, camid=0, viewid=0)
```

Khi `__getitem__`, loader trả về:

```text
image_tensor, au_label_tensor[12], camid, viewid, image_name
```

`make_au_dataloader(cfg)` dùng transform riêng cho mặt, resize về `224 x 224`, normalize bằng CLIP mean/std, rồi chia `train/val` bằng `random_split` 80/20. Điểm mạnh là code giữ được tương thích với pipeline CLIP-ReID cũ bằng `camid/viewid` giả. Điểm yếu là `random_split` theo frame có thể làm train và validation chứa frame rất gần nhau trong cùng subject hoặc cùng trial. Với video DISFA, các frame liên tiếp tương quan cao; do đó metric có thể đẹp hơn khả năng tổng quát thực tế.

Khuyến nghị nghiên cứu là đổi sang subject-independent split:

```text
Train subjects: SN001, SN003, ...
Val subjects:   SN0xx, SN0yy, ...
```

Cách này làm bài toán khó hơn nhưng đáng tin hơn, vì mô hình phải nhận diện AU trên khuôn mặt người chưa thấy trong train.

### 7.3. Chuyển CLIP-ReID thành AU classifier

Nhánh `Action-Units` giữ CLIP visual encoder và BNNeck của CLIP-ReID, nhưng thay classifier ID bằng AU head:

```text
CLIP visual encoder
  -> image feature
  -> BNNeck
  -> Linear(feature_dim, 12)
  -> AU logits
  -> sigmoid probabilities
```

Trong `model/au_head.py`, `AUHead` là một linear layer xuất 12 logits. Trong `model/make_model.py`, nếu `cfg.DATASETS.NAMES == 'disfa'`, model dùng `AUHead` thay cho classifier theo số ID. Vì AU là bài toán multi-label, loss hợp lý là `BCEWithLogitsLoss`, không phải cross entropy một nhãn như ReID/FER truyền thống. `loss/au_loss.py` đã có `WeightedBCELoss`, cho phép thêm `pos_weight` để xử lý mất cân bằng AU.

Hai giai đoạn train được giữ lại theo tinh thần CLIP-ReID:

```text
Stage 1:
  image feature x AU text feature
  -> similarity logits [B, 12]
  -> BCE image-text alignment

Stage 2:
  image encoder + AU head fine-tuning
  -> BCE AU classification
  -> + 0.1 * BCE image-text AU alignment
```

Điểm này rất có giá trị cho EmotionCLIP-ReID vì nó chứng minh có thể biến CLIP-ReID từ "ID classifier" sang "AU multi-label classifier" mà không phá khung CLIP visual/text encoder.

### 7.4. Hạn chế của suy luận cảm xúc hậu kỳ

`au_explainer.py` hiện ánh xạ AU sang mô tả ngôn ngữ và cảm xúc bằng luật, ví dụ:

```text
AU6 + AU12 -> happy
AU4 + AU15 -> sad
AU1 + AU2 + AU5 -> surprised
AU4 + AU9 + AU17 -> angry
```

Cách này dễ hiểu và hữu ích cho giải thích, nhưng chưa đủ cho mục tiêu EmotionCLIP-ReID vì:

| Vấn đề | Ảnh hưởng |
|---|---|
| Threshold cứng `prob > 0.5` | Làm mất độ tin cậy mềm của mô hình; AU 0.49 và 0.01 bị xem như nhau. |
| Một cảm xúc có nhiều tổ hợp AU | Vui có thể chỉ rõ ở miệng, hoặc cả mắt và má; sợ/hơi ngạc nhiên có thể chồng AU. |
| AU cường độ khác nhau | DISFA có intensity 0-5 nhưng pipeline hiện chỉ giữ 0/1. |
| Luật không học từ dữ liệu FER | Không thích nghi được theo dataset, văn hóa, pose, occlusion, hoặc nhãn cảm xúc mơ hồ. |
| Không backprop vào emotion task | AU chỉ giải thích sau, không giúp emotion head học dự đoán cuối tốt hơn. |

Vì vậy, trong nhánh mới, AU không nên chỉ là "bộ luật suy luận sau". AU nên trở thành một nhánh phụ có loss riêng, tạo bằng chứng vi mô để emotion classifier học cách kết hợp với semantic feature của EmotionCLIP-ReID.

## 8. Thiết kế nhánh AU làm dữ kiện cho EmotionCLIP-ReID

Mục tiêu của thiết kế mới là giữ EmotionCLIP-ReID làm nhánh chính, đồng thời thêm AU/micro-description branch như một nguồn bằng chứng. Nói cách khác, model không còn dự đoán cảm xúc chỉ từ ảnh hoặc chỉ từ prompt cảm xúc; model được thấy thêm "dấu hiệu cơ mặt" ở dạng mềm.

### 8.1. Kiến trúc đề xuất

```text
Face image
  -> shared CLIP visual encoder / expression-aware adapter
      -> emotion visual feature
      -> AU branch
          -> au_logits[12]
          -> au_probs[12]
          -> au_embedding

Emotion prompts
  -> CLIP text encoder
  -> emotion text anchors

AU prompts / AU descriptions
  -> CLIP text encoder
  -> AU text anchors

Fusion head
  -> concat/emphasize(emotion visual feature, emotion text similarity, au_probs, au_embedding)
  -> final emotion logits
  -> final emotion prediction
```

Điểm quan trọng là `au_probs` được đưa vào fusion head trước khi threshold. Như vậy mô hình có thể học rằng:

- `AU12=0.95, AU6=0.30` vẫn có thể nghiêng về happy, nhưng độ tự tin khác với `AU12=0.95, AU6=0.90`.
- `AU1/AU2/AU5` có thể là surprise, nhưng nếu đi kèm `AU20` và feature thị giác căng thẳng, model có thể nghiêng về fear.
- Neutral không đơn giản là "không có AU"; có thể có AU rất nhẹ hoặc nhãn emotion không chắc chắn.

### 8.2. Fusion nên là soft learned fusion, không phải rule fusion

Có ba phương án có thể cân nhắc:

| Phương án | Mô tả | Nhận xét |
|---|---|---|
| Rule-based AU-to-emotion | Dùng luật cố định từ AU sang emotion | Dễ giải thích nhưng cứng, không học được biến thiên cá nhân. |
| Logit fusion | Tạo emotion logits từ nhánh chính và AU-to-emotion logits rồi cộng trọng số | Dễ triển khai, nhưng AU vẫn bị ép thành emotion phụ quá sớm. |
| Mid-level soft AU fusion | Đưa AU logits/probs/embedding vào emotion head để học quyết định cuối | Phù hợp nhất vì giữ được bằng chứng vi mô mềm và cho phép học quan hệ phức tạp. |

Khuyến nghị là mid-level soft AU fusion. Đây là lựa chọn tổng quát hơn vì Facial Action Units không ánh xạ một-một sang cảm xúc. FACS/AU mô tả chuyển động cơ mặt, còn emotion label là diễn giải cấp cao; cùng một emotion có thể có nhiều biểu hiện mặt, và cùng một AU có thể xuất hiện trong nhiều emotion. MER-CLIP cũng đi theo tinh thần dùng AU như mô tả chuyển động cơ mặt để căn chỉnh vision-language, thay vì chỉ dùng AU như rule hậu kỳ [MER-CLIP](https://arxiv.org/abs/2505.05937).

### 8.3. Loss và dữ liệu huấn luyện

Loss tổng quát đề xuất:

```text
L = L_emotion_CE
  + lambda_au * L_AU_BCE
  + lambda_align * L_AU_text_alignment
```

Trong đó:

- `L_emotion_CE`: cross entropy hoặc label-distribution loss cho nhãn cảm xúc cuối.
- `L_AU_BCE`: BCE/weighted BCE cho 12 AU, chỉ bật khi dataset có AU label thật hoặc pseudo-label đủ tin cậy.
- `L_AU_text_alignment`: ràng buộc image/AU feature gần AU text anchors, giúp nhánh AU không chỉ học classifier số mà còn giữ ý nghĩa ngôn ngữ.

Nếu batch đến từ dataset emotion-only như RAF-DB hoặc AffectNet, không nên ép `L_AU_BCE` bằng nhãn giả kém tin cậy ngay từ đầu. Có thể dùng AU branch pretrained trên DISFA/BP4D rồi:

```text
emotion-only image
  -> AU branch pretrained
  -> pseudo au_probs
  -> fusion head dùng như feature mềm
```

Ở bước đầu, nên detach pseudo-AU để tránh emotion loss kéo AU branch lệch khỏi ý nghĩa FACS.

### 8.4. Vai trò của AUExplainer trong hệ mới

`AUExplainer` không nên bị bỏ. Nó nên đổi vai trò:

```text
Training/inference decision:
  learned fusion head quyết định cảm xúc cuối

Explanation:
  AUExplainer hoặc template language mô tả vì sao model nghiêng về cảm xúc đó
```

Ví dụ output nghiên cứu hợp lý:

```text
Predicted emotion: happy, confidence 0.82
AU evidence: AU12 lip corner puller 0.91, AU6 cheek raiser 0.63
Explanation: model dự đoán vui vì miệng kéo lên mạnh và má nâng ở mức vừa.
```

Như vậy báo cáo vừa giữ được tính diễn giải của AU, vừa tránh biến rule hậu kỳ thành quyết định chính của mô hình.

## 9. AU-based prompt khi có AU labels

AU-based prompt chỉ nên triển khai đầy đủ khi có AU labels thật, vì prompt AU cần được neo vào quan sát cơ mặt. Nếu không có nhãn AU, prompt AU dễ trở thành mô tả đẹp về mặt ngôn ngữ nhưng không kiểm chứng được bằng dữ liệu.

### 9.1. Prompt theo từng AU

Thay vì tạo prompt cho emotion tổng quát:

```text
A photo of a person expressing happy.
```

ta thêm prompt theo chuyển động vi mô:

```text
A face showing inner brow raiser.
A face showing outer brow raiser.
A face showing brow lowerer.
A face showing cheek raiser.
A face showing lip corner puller.
A face showing lips part.
```

Trong code, có thể xem mỗi AU là một "class text anchor" tương tự cách `PromptLearner` hiện học token theo ID/class. Khác biệt là AU là multi-label, nên một ảnh có thể khớp nhiều AU text anchors cùng lúc.

### 9.2. Không sinh prompt cho mọi tổ hợp AU

Với 12 AU, nếu sinh mọi tổ hợp sẽ có `2^12 = 4096` prompt. Cách này không nên dùng vì:

- nhiều tổ hợp không xuất hiện hoặc rất hiếm;
- tổ hợp AU thật có intensity khác nhau, không chỉ có bật/tắt;
- prompt dài dễ nhiễu và khó học ổn định;
- số tổ hợp tăng nhanh nếu mở rộng AU.

Khuyến nghị là compositional prompt:

```text
AU text anchors:
  T_AU1, T_AU2, ..., T_AU12

Image AU probabilities:
  p_AU1, p_AU2, ..., p_AU12

AU semantic evidence:
  T_micro = sum_k p_AUk * T_AUk
```

Nếu có intensity 0-5, nên chuẩn hóa intensity thành trọng số mềm:

```text
p_AUk = intensity_k / 5
```

hoặc dùng label mềm theo confidence. Cách này giữ được ý tưởng "mô tả vi mô" mà không ép mọi khuôn mặt vào một luật cứng.

### 9.3. Kết hợp prompt cảm xúc và prompt AU

Prompt cảm xúc và prompt AU nên giữ hai vai trò khác nhau:

| Loại prompt | Vai trò |
|---|---|
| Emotion prompt | Neo không gian ngữ nghĩa cấp cao: happy, sad, angry, fear, surprise, disgust, neutral. |
| AU prompt | Neo chuyển động cơ mặt cấp thấp: brow raiser, cheek raiser, lip corner puller, jaw drop. |
| Fusion prompt/evidence | Giúp model học quan hệ mềm giữa biểu hiện vi mô và nhãn cảm xúc cuối. |

Ví dụ một mẫu training có nhãn `happy` và AU6/AU12:

```text
Image feature
  -> gần emotion text anchor "happy"
  -> gần AU text anchors "cheek raiser" và "lip corner puller"
  -> fusion head học rằng tổ hợp này ủng hộ happy
```

Nếu một ảnh `happy` chỉ có AU12 rõ nhưng AU6 yếu, model vẫn học được xác suất thấp hơn cho "Duchenne smile" thay vì loại bỏ mẫu vì thiếu một AU trong luật.

## 10. Dataset phù hợp cho hướng AU/micro-description

Chiến lược dataset nên chia theo ba tầng, vì hiếm dataset vừa lớn, vừa in-the-wild, vừa có emotion label và AU label đáng tin cậy.

### 10.1. Tầng 1 - Có cả emotion và AU

Đây là nhóm ưu tiên nhất để train hoặc đánh giá fusion thật, vì model có thể học trực tiếp quan hệ giữa AU evidence và emotion label.

| Dataset | Nhãn chính | Điểm phù hợp | Lưu ý |
|---|---|---|---|
| Aff-Wild2 | 7 basic expressions, 12 AU, valence-arousal theo frame | Rất hợp với multi-task/fusion vì cùng một database có expression, AU và VA; là in-the-wild video lớn [Aff-Wild2](https://ibug.doc.ic.ac.uk/resources/aff-wild2/). | Cần xin quyền truy cập; video lớn, cần trích frame/clip cẩn thận. |
| RAF-AU | Emotion judgement + AU annotations in-the-wild | Đúng hướng "perceived emotion + objective AU"; paper nêu rõ mục tiêu phân tích quan hệ AUs và facial expressions [RAF-AU](https://openaccess.thecvf.com/content/ACCV2020/html/Yan_RAF-AU_Database_In-the-Wild_Facial_Expressions_with_Subjective_Emotion_Judgement_and_ACCV_2020_paper.html). | Cần kiểm tra quyền tải; quy mô nhỏ hơn Aff-Wild2. |
| EmotioNet | AU/AU intensity và emotion categories quy mô rất lớn | Phù hợp để học weak/pseudo AU và prompt vi mô; bài CVPR 2016 mô tả annotation một triệu ảnh với AU, intensity và emotion category [EmotioNet](https://openaccess.thecvf.com/content_cvpr_2016/html/Benitez-Quiroz_EmotioNet_An_Accurate_CVPR_2016_paper.html). | Nhiều nhãn AU là tự động, phải xem là noisy label. |
| CK+ | FACS AU + emotion-specified expression | Tốt để kiểm thử nhanh vì có FACS và emotion label; paper CK+ mô tả 593 sequences từ 123 subjects, peak expression được FACS coded và emotion labels được validate [CK+](https://www.jeffcohn.net/wp-content/uploads/2020/02/CVPR2010_CK2.pdf.pdf). | Chủ yếu posed/lab-controlled, không đủ cho in-the-wild generalization. |

Với đề tài EmotionCLIP-ReID, tầng 1 nên là mục tiêu chính sau khi baseline chạy được. Nếu chỉ chọn một hướng dữ liệu để chứng minh AU-fusion, RAF-AU hoặc Aff-Wild2 là hợp lý nhất vì có nhãn emotion và AU trong cùng miền dữ liệu.

### 10.2. Tầng 2 - AU-only để pretrain micro branch

Nhóm này không đủ để train emotion classifier cuối, nhưng rất quan trọng để làm nhánh AU hiểu FACS trước khi fusion.

| Dataset | Nhãn chính | Điểm phù hợp | Lưu ý |
|---|---|---|---|
| DISFA | 12 AU intensity 0-5, frame-level | Đang được repo `Action-Units` hỗ trợ; hợp để pretrain AU branch và AU prompt alignment [DISFA official](https://mohammadmahoor.com/pages/databases/disfa/). | Không có emotion label trực tiếp; cần subject-independent split. |
| BP4D/BP4D+ | FACS AU, spontaneous tasks, 2D/3D/video | BP4D có 41 subjects, 8 tasks, 328 sequences, metadata gồm FACS coding, landmarks và pose [BP4D](https://www.sciencedirect.com/science/article/pii/S0262885614001012). | Cần xử lý video/frame và quyền truy cập. |

Quy trình đề xuất:

```text
DISFA/BP4D
  -> train AU branch + AU prompt anchors
  -> lưu AU encoder/head
  -> dùng làm nhánh phụ trong EmotionCLIP-ReID
```

### 10.3. Tầng 3 - Emotion-only để train emotion branch hoặc pseudo-AU

Nhóm này giúp học cảm xúc trên dữ liệu lớn hoặc dễ lấy hơn, nhưng không đủ để xác nhận AU-fusion nếu không có AU label thật.

| Dataset | Nhãn chính | Điểm phù hợp | Lưu ý |
|---|---|---|---|
| RAF-DB | 7 basic emotions, compound emotions, khoảng 30K ảnh in-the-wild | Dễ dùng cho FER image classification; official page mô tả 29,672 ảnh và split train/test [RAF-DB](https://www.whdeng.cn/RAF/model1.html). | Không có AU label trong RAF-DB cơ bản; cần pseudo-AU hoặc ghép với RAF-AU. |
| AffectNet | Discrete emotion + valence/arousal, hơn 1M ảnh web | Rất tốt để học emotion branch và robustness in-the-wild; paper nêu hơn 1M ảnh và khoảng một nửa được annotate thủ công [AffectNet](https://arxiv.org/abs/1708.03985). | Không phải AU dataset; nhãn mất cân bằng và có lớp khó như contempt. |
| FERPlus | Label distribution từ nhiều annotator trên FER2013 | Phù hợp để học uncertainty/soft label vì mỗi ảnh có phân phối nhãn cảm xúc từ 10 taggers [FERPlus](https://github.com/microsoft/FERPlus). | Ảnh 48x48, chất lượng thấp; không có AU label. |

Với tầng 3, AU branch nên được dùng như pseudo-evidence:

```text
Emotion-only image
  -> AU branch pretrained dự đoán au_probs
  -> emotion fusion head học từ au_probs nhưng không xem đó là ground truth
```

Trong báo cáo khoa học, cần ghi rõ kết quả trên tầng 3 chỉ chứng minh tính hữu ích của pseudo-AU, không thay thế benchmark có AU label thật.

### 10.4. Khuyến nghị dataset cho giai đoạn triển khai

| Giai đoạn | Dataset khuyến nghị | Mục tiêu |
|---|---|---|
| Prototype nhanh | DISFA + RAF-DB/FERPlus | Pretrain AU branch rồi kiểm tra fusion/pseudo-AU trên FER đơn giản. |
| Nghiên cứu đúng nhất | RAF-AU hoặc Aff-Wild2 | Train/evaluate quan hệ AU-emotion trong cùng dữ liệu. |
| Mở rộng độ tổng quát | AffectNet + AU pseudo-label + kiểm thử cross-dataset | Tăng robustness in-the-wild, nhưng cần phân tích nhiễu pseudo-label. |

Thứ tự an toàn là:

```text
1. Pretrain AU branch trên DISFA/BP4D.
2. Train EmotionCLIP-ReID baseline trên RAF-DB hoặc AffectNet.
3. Thêm soft AU fusion.
4. Nếu có RAF-AU/Aff-Wild2, fine-tune với cả emotion loss và AU loss thật.
5. Đánh giá macro-F1, balanced accuracy, confusion matrix, AU F1 và calibration/uncertainty.
```

## 11. Graph codebase bằng Grapuco

Repository đã được init và ingest vào Grapuco bằng CLI cục bộ:

```text
grapuco init --name EmotionCLIP-ReID
grapuco ingest
```

Kết quả ingest:

```text
Repository: EmotionCLIP-ReID
Repo ID: cb7f0ca4-0563-46ec-8c0f-bb08ad5e1e5a
Remote status: COMPLETED
Cached/parseable files: 51
Parsed nodes: 286
Parsed edges: 364
Embeddings: ON
Data flows: ON
Source code bytes sent: 0
```

MCP Grapuco đã xác nhận repository `EmotionCLIP-ReID` và `get_architecture` trả về architecture map. File [emotionclip_reid_codebase_graph.drawio](../emotionclip_reid_codebase_graph.drawio) đã được cập nhật thành graph codebase tóm tắt từ Grapuco, tập trung vào các cụm quan trọng cho phân tích EmotionCLIP-ReID.

Graph Grapuco cần có ít nhất các cụm:

- `train_clipreid.py` gọi dataloader, model, loss, optimizer, Stage 1 và Stage 2 processors.
- `model/make_model_clipreid.py` chứa `PromptLearner`, `TextEncoder`, CLIP image encoder và classifier heads.
- `processor/processor_clipreid_stage1.py` biểu diễn prompt learning bằng image-text contrastive loss.
- `processor/processor_clipreid_stage2.py` biểu diễn image fine-tuning với text anchors và I2T logits.
- `loss/make_loss.py`, `loss/supcontrast.py`, `loss/triplet_loss.py` biểu diễn các objective chính.

## 12. Lộ trình triển khai đề xuất

### Bước 1 - FER baseline tối thiểu

Triển khai dataset loader FER, emotion class list, classifier head và metric FER. Giữ CLIP visual encoder, bỏ ReID evaluator, kiểm tra train/inference trên một dataset nhỏ. Đây là bước bắt buộc trước khi thêm prompt phức tạp.

### Bước 2 - Emotion prompt learning

Chuyển `PromptLearner` từ ID-specific token sang emotion-class token. Stage 1 học text descriptor cho từng emotion class bằng ảnh FER, Stage 2 dùng text descriptor làm anchor cho image encoder.

### Bước 3 - Adapter nhẹ

Chèn adapter vào một số block ViT cuối, chỉ train adapter/head và giữ phần lớn CLIP frozen. Bước này giúp giảm overfitting so với fine-tune toàn bộ image encoder.

### Bước 4 - Uncertainty

Thêm evidence head và uncertainty loss sau khi baseline đã ổn định. Không nên thêm EDL ngay từ đầu vì khó debug nếu metric thấp.

### Bước 5 - AU/micro-description

Triển khai AU branch như một nhánh con của EmotionCLIP-ReID, không dùng nó như rule hậu kỳ. Nhánh này xuất `au_logits`, `au_probs` và `au_embedding`; emotion fusion head học kết hợp feature EmotionCLIP với AU soft evidence. AU-based prompt chỉ triển khai đầy đủ khi có AU labels hoặc một pipeline pseudo-AU đã được kiểm định.

### Bước 6 - Đánh giá fusion và diễn giải

So sánh ít nhất ba cấu hình: EmotionCLIP-ReID baseline, AU-only/rule baseline, và EmotionCLIP-ReID + soft AU fusion. Báo cáo thêm ví dụ diễn giải bằng AU để cho thấy mô hình dự đoán cảm xúc dựa trên dấu hiệu vi mô nào, nhưng không xem rule explanation là prediction chính.

## 13. Tiêu chí chấp nhận

- Report phân biệt rõ CLIP-ReID hiện tại, nhánh Action-Units hiện tại, và EmotionCLIP-ReID + AU-fusion đề xuất.
- Có hai sơ đồ Draw.io cho mô hình hiện tại và mô hình đề xuất.
- Có file Draw.io cho graph codebase, nhưng ghi rõ blocker Grapuco thay vì graph tự dựng.
- Bảng khả thi trả lời rõ phần nào triển khai được ngay, phần nào cần thay đổi, phần nào rủi ro cao, phần nào chưa triển khai được từ repo hiện tại.
- Có phân tích rõ vì sao AU rule-based hậu kỳ chưa đủ và vì sao nên dùng learned soft AU fusion.
- Có đề xuất dataset theo ba tầng: AU+emotion, AU-only, emotion-only.
- Không có thay đổi vào code huấn luyện/model trong task tài liệu này.
