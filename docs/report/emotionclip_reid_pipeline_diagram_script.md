# Script trình bày sơ đồ EmotionCLIP-ReID v1 pipeline

Nguồn sơ đồ: `docs/report/fig/emotionclip_reid_publication_pipeline_update.drawio`

## 1. Ý chính của sơ đồ

Sơ đồ mô tả pipeline EmotionCLIP-ReID v1 cho bài toán nhận diện biểu cảm khuôn mặt, viết tắt là FER. Pipeline có hai giai đoạn.

Giai đoạn 1 học các emotion text anchors, tức là các prototype biểu diễn 7 lớp cảm xúc trong không gian text của CLIP. Ở giai đoạn này, cả CLIP image encoder và CLIP text encoder đều được giữ cố định; mô hình chỉ cập nhật các context token `[X1]...[XM]` trong prompt.

Giai đoạn 2 cố định các text anchors đã học được và dùng chúng để hướng dẫn huấn luyện nhánh thị giác. Ảnh đi qua CLIP ViT visual encoder có thêm expression adapters, tạo ra global feature và patch features. Sau đó mô hình kết hợp ba nguồn logit: classifier head, global text alignment và local text alignment. Output cuối được dùng cho dự đoán, uncertainty và loss huấn luyện.

## 2. Phân tích từng khối trong Stage 1

### Stage 1: Prompt / Emotion Anchor Learning

Mục tiêu của Stage 1 là học một bộ text anchor cho từng lớp cảm xúc. Các anchor này không phải là câu mô tả thủ công cuối cùng, mà là embedding text đã được tối ưu để khớp với feature ảnh của từng lớp.

`image input x`

Đây là ảnh đầu vào trong tập huấn luyện FER. Ảnh được dùng để lấy biểu diễn thị giác ban đầu từ CLIP. Trong giai đoạn này, ảnh không đi qua một backbone được fine-tune; nó chỉ dùng để tạo feature tham chiếu.

`CLIP Image Encoder` và nhãn `FROZEN`

Khối này mã hóa ảnh thành đặc trưng ảnh. Vì encoder bị frozen, trọng số của CLIP image encoder không thay đổi trong Stage 1. Vai trò của nó là cung cấp một không gian ảnh ổn định để prompt learner học cách đặt các anchor text vào vị trí phù hợp.

`cached global image features z_v`

Đây là feature ảnh toàn cục sau khi đi qua CLIP Image Encoder. Từ "cached" nghĩa là feature này có thể được tính trước và lưu lại để tăng tốc training. Khi huấn luyện prompt, mô hình không cần chạy lại image encoder nhiều lần.

`Emotion Prompt Learner`

Đây là cụm khối học prompt cho từng lớp cảm xúc. Nó nhận danh sách lớp cảm xúc và sinh prompt tương ứng cho từng lớp. Điểm quan trọng là prompt có phần token học được, nên mô hình không chỉ dùng câu prompt cố định như "a happy face".

`text input: emotion class set C, c = 1..7`

Đây là tập nhãn cảm xúc, ví dụ 7 lớp như anger, disgust, fear, happiness, sadness, surprise và neutral. Mỗi lớp `c` sẽ có một prompt riêng `p_c`.

`generated prompt p_c`

Prompt có dạng: `A photo of a face showing [X1] [X2] ... [XM] expression of [emotion c].` Với mỗi lớp cảm xúc, phần `[emotion c]` là tên lớp, còn `[X1]...[XM]` là các context token học được.

`class-specific learnable context tokens [X1]...[XM]` và nhãn `TRAINABLE`

Đây là phần duy nhất được cập nhật chính trong Stage 1. Các token `[X]` là vector ẩn trong không gian embedding, không phải từ tiếng Anh cụ thể. Vì vậy không nên diễn giải `[X]` là "eye", "mouth" hay "AU"; chúng là latent vectors được học để giúp prompt khớp tốt hơn với ảnh.

`Text Encoder input p_c, c = 1..7`

Sau khi tạo prompt cho từng lớp, các prompt này được đưa vào CLIP text encoder. Tại mỗi vòng huấn luyện, ta có 7 prompt tương ứng 7 cảm xúc.

`CLIP Text Encoder` và nhãn `FROZEN`

Text encoder chuyển prompt thành text feature. Encoder này cũng frozen, nên ý nghĩa là ta không thay đổi kiến thức ngôn ngữ gốc của CLIP. Mô hình chỉ điều chỉnh các context token đầu vào.

`text features T`

Đây là ma trận text features cho 7 cảm xúc. Mỗi hàng hoặc mỗi vector tương ứng một emotion prototype trong không gian CLIP.

`S1 = scale(z_v T^T): emotion similarity`

Khối này tính độ tương đồng giữa feature ảnh `z_v` và từng text feature trong `T`. Phép tính là dot product đã scale: `S1 = scale(z_v T^T)`. Kết quả `S1` là logit phân loại theo 7 lớp cảm xúc.

`L1 = CE(S1, y): update [X] only`

Loss Stage 1 là cross-entropy giữa similarity logits `S1` và nhãn thật `y`. Backprop chỉ cập nhật các context token `[X]`, không cập nhật CLIP image encoder và CLIP text encoder. Vì vậy Stage 1 chủ yếu là học anchor text, không phải huấn luyện lại toàn bộ CLIP.

`Optimized text anchors T* = get_text().detach(): 7 emotion prototypes`

Sau khi học xong, text features tối ưu được lấy ra thành `T*`. Từ `detach()` nhấn mạnh rằng các anchor này được tách khỏi graph huấn luyện và sẽ cố định khi chuyển sang Stage 2. Ta có thể xem `T*` là 7 prototype cảm xúc đã học.

## 3. Luồng xử lý Stage 1 theo mũi tên

1. Ảnh `x` đi vào CLIP Image Encoder.
2. CLIP Image Encoder tạo global image feature `z_v` và cache lại.
3. Tập lớp cảm xúc `C` được đưa vào prompt learner để tạo prompt `p_c`.
4. Các context token `[X1]...[XM]` được chèn vào prompt và là phần trainable.
5. Prompt `p_c` đi qua frozen CLIP Text Encoder để tạo text features `T`.
6. Feature ảnh `z_v` và text features `T` gặp nhau tại khối similarity bằng dot-product.
7. Similarity logits `S1` được so với nhãn thật bằng cross-entropy.
8. Gradient quay ngược về context tokens `[X]` để tối ưu prompt.
9. Sau khi tối ưu, `T` được detach thành `T*` và chuyển sang Stage 2 như fixed emotion anchors.

## 4. Phân tích từng khối trong Stage 2

### Stage 2: Visual Adaptation and FER Training

Mục tiêu của Stage 2 là huấn luyện nhánh ảnh cho bài toán FER, nhưng vẫn được định hướng bởi các anchor cảm xúc `T*` đã học từ Stage 1. Khác với Stage 1, ở đây có các phần trainable trong visual encoder, classifier, alignment và fusion.

`FER image x`

Đây là ảnh FER dùng để huấn luyện hoặc đánh giá ở giai đoạn chính. Ảnh đi vào visual encoder để trích xuất cả đặc trưng toàn cục và đặc trưng patch.

`CLIP ViT visual encoder + expression adapters`

Đây là backbone thị giác dựa trên CLIP ViT, được thêm các expression adapters để thích nghi với bài toán biểu cảm. Nhãn `(train adapters + last blocks)` cho biết không fine-tune toàn bộ mô hình, mà chỉ huấn luyện adapters và một số block cuối. Cách này giảm chi phí và hạn chế phá vỡ không gian CLIP ban đầu.

`global feature z_g = CLS image token`

Feature toàn cục lấy từ CLS token của ViT. Nó đại diện cho toàn bộ khuôn mặt và được dùng cho classifier head cũng như global text alignment.

`patch features P = visual patch tokens`

Patch features là các token cục bộ của ảnh. Mỗi token tương ứng một vùng ảnh hoặc patch trong ViT. Chúng hữu ích vì biểu cảm thường nằm ở các vùng cục bộ như mắt, lông mày, miệng hoặc má, dù sơ đồ không gán trực tiếp patch nào với AU nào.

`fixed text anchors T*` và nhãn `DETACHED`

Đây là 7 emotion prototypes chuyển từ Stage 1 sang. Vì đã detached, chúng không được cập nhật trong Stage 2. Vai trò của chúng là làm chuẩn ngữ nghĩa cố định để kéo feature ảnh về đúng lớp cảm xúc.

`classifier head: S_cls = classifier(z_g)`

Đây là nhánh phân loại trực tiếp từ global feature. Nó học logit cảm xúc bằng một classifier thông thường. Nhánh này cung cấp tín hiệu supervised mạnh và ổn định.

`global text alignment: S_g = scale(z_g T*^T)`

Khối này so sánh global image feature `z_g` với các fixed text anchors `T*`. Nếu ảnh thể hiện cảm xúc "happy", feature `z_g` nên gần anchor "happy" hơn các anchor khác. Đây là nhánh alignment toàn cục giữa ảnh và text.

`local alignment: S_l = mean_topk(scale(P T*^T))`

Khối này so sánh từng patch feature trong `P` với các text anchors `T*`. Sau đó chọn top-k patch có độ tương đồng cao và lấy trung bình. Ý tưởng là không phải mọi vùng ảnh đều chứa tín hiệu biểu cảm; local alignment tập trung vào những patch có bằng chứng cảm xúc mạnh nhất.

`learned fusion: S = w_cls S_cls + w_g S_g + w_l S_l`

Khối fusion học cách kết hợp ba nguồn logit: logit classifier, logit alignment toàn cục và logit alignment cục bộ. Các trọng số `w_cls`, `w_g`, `w_l` cho phép mô hình điều chỉnh mức tin cậy vào từng nhánh thay vì cộng đều một cách thủ công.

`EDL outputs: evidence = softplus(S), alpha = evidence + 1`

Khối này chuyển logits cuối `S` thành evidence bằng `softplus`, sau đó tạo tham số Dirichlet `alpha`. Đây là cơ chế evidential deep learning, dùng để tạo cả dự đoán và tín hiệu uncertainty. Nếu evidence thấp hoặc phân tán, mô hình có thể biểu thị độ không chắc chắn cao hơn.

`prediction + uncertainty`

Output cuối gồm nhãn dự đoán và các metric đánh giá. Sơ đồ liệt kê accuracy, balanced accuracy, macro-F1, ECE và risk-AUC. Các metric này phù hợp với FER vì bài toán có mất cân bằng lớp, nhãn mơ hồ và yêu cầu calibration.

`L2 = CE(S,y) + beta CE(S_g+S_l,y) + lambda L_EDL(S,y)`

Đây là loss chính của Stage 2. Thành phần `CE(S,y)` huấn luyện prediction cuối. Thành phần `beta CE(S_g+S_l,y)` ép hai nhánh alignment với text anchors cũng mang tín hiệu phân loại đúng. Thành phần `lambda L_EDL(S,y)` bổ sung ràng buộc evidential/uncertainty. Loss này cập nhật adapters, một số block cuối, classifier/fusion và các tham số trainable của Stage 2, nhưng không cập nhật `T*`.

## 5. Luồng xử lý Stage 2 theo mũi tên

1. Ảnh FER `x` đi vào CLIP ViT visual encoder có expression adapters.
2. Visual encoder tạo global feature `z_g` từ CLS token.
3. Visual encoder đồng thời tạo patch features `P` từ các visual patch tokens.
4. `z_g` đi vào classifier head để tạo `S_cls`.
5. `z_g` cũng được so sánh với `T*` để tạo global alignment logits `S_g`.
6. `P` được so sánh với `T*`; mô hình chọn top-k patch liên quan nhất để tạo local alignment logits `S_l`.
7. Ba nguồn logit `S_cls`, `S_g` và `S_l` đi vào learned fusion.
8. Fusion tạo logits cuối `S`.
9. `S` được chuyển thành evidence và `alpha` cho EDL.
10. Từ evidence/alpha, mô hình tạo prediction và uncertainty, sau đó báo cáo các metric đánh giá.
11. Trong huấn luyện, `S`, `S_g` và `S_l` cùng đi vào loss `L2` để cập nhật các phần trainable của Stage 2.

## 6. Script nói khi trình bày sơ đồ

Slide này mô tả pipeline EmotionCLIP-ReID v1 theo hướng two-stage FER. Ý tưởng chính là em không huấn luyện mô hình biểu cảm chỉ bằng classifier thông thường, mà trước hết học một bộ text anchor cho các lớp cảm xúc trong không gian CLIP, sau đó dùng các anchor này để hướng dẫn nhánh ảnh.

Ở giai đoạn 1, mục tiêu là Prompt hoặc Emotion Anchor Learning. Ảnh đầu vào `x` đi qua CLIP Image Encoder để tạo global image feature `z_v`. Cả image encoder và text encoder của CLIP đều được giữ frozen, nên Stage 1 không làm thay đổi backbone CLIP. Feature ảnh `z_v` còn được cache lại để quá trình học prompt nhanh hơn.

Ở nhánh text, đầu vào là tập 7 lớp cảm xúc. Với mỗi lớp `c`, prompt learner sinh một prompt `p_c` có dạng "A photo of a face showing [X1] [X2] ... [XM] expression of [emotion c]." Phần quan trọng nhất ở đây là các token `[X1]...[XM]`. Đây là các latent vectors được học theo từng lớp, không phải các từ mô tả mắt, miệng hay AU cụ thể.

Các prompt này đi qua frozen CLIP Text Encoder để tạo text features `T`. Sau đó feature ảnh `z_v` và text features `T` được so sánh bằng dot-product similarity, tạo logits `S1 = scale(z_v T^T)`. Loss của Stage 1 là `L1 = CE(S1, y)`, và gradient chỉ cập nhật các context token `[X]`. Vì vậy kết quả của Stage 1 là một bộ text anchors tối ưu, ký hiệu là `T*`, gồm 7 emotion prototypes. Bộ `T*` này được detach và cố định để chuyển sang giai đoạn 2.

Sang Stage 2, mục tiêu chuyển từ học prompt sang huấn luyện nhánh visual cho FER. Ảnh FER `x` đi qua CLIP ViT visual encoder có thêm expression adapters. Ở đây không fine-tune toàn bộ backbone; sơ đồ nhấn mạnh các phần trainable là adapters và một số block cuối. Encoder tạo ra hai loại đặc trưng: global feature `z_g` từ CLS token và patch features `P` từ các visual patch tokens.

Từ `z_g`, mô hình có một classifier head tạo logit `S_cls`. Đây là nhánh phân loại trực tiếp, giống cách supervised learning thông thường. Nhưng ngoài classifier, pipeline còn dùng fixed text anchors `T*` để tạo hai nhánh alignment.

Nhánh alignment thứ nhất là global text alignment. Nó tính `S_g = scale(z_g T*^T)`, tức là đo độ gần giữa feature toàn cục của ảnh và từng emotion anchor. Nếu ảnh thuộc lớp happy, lý tưởng là `z_g` sẽ gần anchor happy hơn các anchor còn lại.

Nhánh alignment thứ hai là local alignment. Thay vì dùng toàn bộ ảnh, nó so sánh từng patch token trong `P` với `T*`, sau đó lấy trung bình top-k similarity cao nhất để tạo `S_l`. Cách này hợp lý với biểu cảm khuôn mặt vì tín hiệu cảm xúc thường xuất hiện cục bộ, ví dụ ở mắt, lông mày hoặc vùng miệng, và không phải patch nào cũng hữu ích như nhau.

Ba nguồn logit `S_cls`, `S_g` và `S_l` sau đó được đưa vào learned fusion. Công thức là `S = w_cls S_cls + w_g S_g + w_l S_l`. Nghĩa là mô hình học trọng số để quyết định nên tin classifier, global alignment hay local alignment nhiều hơn trong từng cấu hình huấn luyện.

Sau fusion, logits cuối `S` được đưa qua khối evidential output. Cụ thể, `evidence = softplus(S)` và `alpha = evidence + 1`. Phần này cho phép mô hình không chỉ đưa ra nhãn dự đoán, mà còn tạo tín hiệu uncertainty. Khi đánh giá, pipeline báo cáo accuracy, balanced accuracy, macro-F1, ECE và risk-AUC, thay vì chỉ báo cáo accuracy.

Loss Stage 2 gồm ba phần: cross-entropy trên logits cuối `S`, alignment loss trên `S_g + S_l`, và EDL loss trên `S`. Công thức trong sơ đồ là `L2 = CE(S,y) + beta CE(S_g+S_l,y) + lambda L_EDL(S,y)`. Loss này cập nhật các thành phần trainable của Stage 2, còn text anchors `T*` vẫn cố định.

Tóm lại, điểm chính của sơ đồ là pipeline tách rõ hai nhiệm vụ. Stage 1 học semantic anchors cho cảm xúc bằng prompt learning trong không gian CLIP. Stage 2 dùng các anchors đó để hướng dẫn visual adaptation bằng cả global và local alignment, rồi kết hợp với classifier và uncertainty output để tạo dự đoán FER ổn định hơn.

## 7. Câu kết ngắn để chuyển slide

Vì vậy, contribution của sơ đồ này không nằm ở một block đơn lẻ, mà ở cách tổ chức pipeline: học anchor ngữ nghĩa trước, cố định anchor đó, rồi dùng nó để điều hướng huấn luyện visual encoder. Phần cần kiểm chứng tiếp là ablation: bỏ prompt learning, bỏ global alignment, bỏ local alignment hoặc bỏ EDL để xem từng thành phần đóng góp bao nhiêu.

