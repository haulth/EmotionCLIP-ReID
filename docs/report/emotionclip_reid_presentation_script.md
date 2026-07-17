# Script trình bày EmotionCLIP-ReID

Nguồn bám theo `docs/report/emotionclip_reid.pptx`, mã hiện tại trong repo, và kết quả tổng hợp tại `outputs/report_w4/emotionclip_outputs_summary.json`.

## Slide 1 - EmotionCLIP-ReID

Em xin trình bày tiến độ hiện tại của hướng EmotionCLIP-ReID. Ý tưởng chính của đề tài là chuyển khung CLIP-ReID từ bài toán nhận dạng định danh sang bài toán nhận diện biểu cảm khuôn mặt. Trọng tâm không phải chỉ thay nhãn ID bằng nhãn emotion, mà là dùng mô tả ngữ nghĩa của cảm xúc làm neo để huấn luyện nhánh ảnh theo cách dễ diễn giải hơn.

Trong phần trình bày này, em sẽ đi từ vấn đề nghiên cứu, khoảng trống trong tài liệu, cách kế thừa baseline CLIP-ReID, phần đã triển khai, kết quả hiện tại và cuối cùng là các điểm cần kiểm chứng tiếp.

## Slide 2 - Vấn đề cần giải quyết

Bài toán em nhắm tới là facial expression recognition trong điều kiện thực tế. Với ảnh khuôn mặt ngoài môi trường kiểm soát, accuracy tổng không đủ để kết luận mô hình ổn định.

Có bốn khó khăn chính: che khuất có thể làm mất tín hiệu vùng mắt hoặc miệng; thay đổi pose và domain shift làm biểu diễn ảnh kém ổn định; nhãn biểu cảm có tính mơ hồ nên mô hình dễ quá tự tin; và nếu chỉ báo cáo accuracy thì chưa phản ánh được độ cân bằng giữa các lớp.

Vì vậy, câu hỏi nghiên cứu của em là: có thể xây dựng một mô hình FER ổn định hơn và có thể diễn giải dựa trên tín hiệu biểu cảm có nghĩa hay không. Các bằng chứng cần có sẽ không chỉ là accuracy, mà còn gồm macro-F1, balanced accuracy, ablation, calibration và phân tích trên các subset khó.

## Slide 3 - Tổng hợp tài liệu: các nhánh hội tụ

Literature review ở đây được dùng để định vị khoảng trống nghiên cứu, không trình bày như một danh sách paper.

Nhánh thứ nhất là CLIP và VLM adaptation. Các phương pháp prompt hoặc adapter cho thấy có thể thích nghi không gian ngữ nghĩa của CLIP cho downstream task, nhưng thường vẫn ở cấp nhãn lớp.

Nhánh thứ hai là language-guided ReID, đặc biệt là CLIP-ReID. Điểm em kế thừa là cơ chế dùng text làm anchor cho visual encoder. Tuy nhiên trong CLIP-ReID, token học được vẫn phục vụ định danh, không trực tiếp mô tả biểu cảm.

Nhánh thứ ba là CLIP-FER hoặc DFER, cho thấy descriptor và adapter có thể cải thiện nhận diện biểu cảm. Nhưng vai trò của fixed semantic anchor vẫn cần được kiểm chứng rõ hơn.

Nhánh thứ tư là AU/FACS semantics. Đây là cầu nối giữa giải phẫu cơ mặt và ngôn ngữ, nhưng nhãn AU thường thiếu, nhiễu hoặc không đồng nhất giữa dataset.

Nhánh cuối là uncertainty-aware FER, hữu ích cho label ambiguity và overconfidence. Tuy nhiên trong hướng hiện tại, uncertainty nên được xem là phần hiệu chỉnh sau khi baseline descriptor đã ổn.

Luận điểm của em là giữ khung hai giai đoạn của CLIP-ReID, nhưng thay anchor định danh bằng descriptor cảm xúc hoặc AU có thể kiểm chứng.

## Slide 4 - Baseline: giá trị kế thừa và giới hạn

Slide này đặt baseline vào đúng vị trí. CLIP gốc học liên kết ảnh và text bằng contrastive learning với prompt thủ công. CoOp mở rộng bằng learnable context token, tức là thay vì chỉ dùng prompt cố định, mô hình học một phần token ngữ cảnh cho downstream task.

CLIP-ReID kế thừa logic đó cho person re-identification. Vì ReID không có mô tả text tự nhiên cho từng ID, CLIP-ReID học token theo ID ở Stage 1, sau đó cố định text anchor và huấn luyện nhánh ảnh ở Stage 2.

Giá trị em kế thừa là scaffold hai giai đoạn và cách dùng text anchor. Giới hạn khi chuyển sang FER là ID token không mô tả dấu hiệu cơ mặt tạo ra cảm xúc. Vì vậy nếu bê nguyên baseline sang FER, mô hình có thể phân loại được nhưng chưa chắc có ý nghĩa biểu cảm hoặc độ diễn giải tốt.

## Slide 5 - Khoảng trống khi chuyển baseline sang FER

Khoảng trống chính nằm ở mục tiêu học biểu diễn. Với ReID, token theo ID chỉ cần giúp phân biệt người này với người khác. Nhưng với FER, em cần biểu diễn phản ánh dấu hiệu cảm xúc.

Nếu dùng ID token, mô hình thiếu nghĩa biểu cảm. Nếu dùng prompt class-name như happy, sad, neutral thì quá thô, chưa mô tả vùng cơ mặt hoặc biến thể biểu cảm. AU/FACS có ý nghĩa giải phẫu hơn, nhưng nhãn AU không phải dataset nào cũng có và có thể nhiễu. Uncertainty hữu ích cho mẫu mơ hồ, nhưng nếu đưa quá sớm thì khó biết phần cải thiện đến từ descriptor hay từ cơ chế hiệu chỉnh.

Vì vậy research gap của em là cần một descriptor emotion/AU làm fixed semantic anchor, và phải chứng minh bằng ablation. Nói cách khác, contribution không phải là "thêm nhiều module", mà là kiểm chứng từng module có thật sự đóng góp hay không.

## Slide 6 - Đề xuất hiện tại: two-stage semantic anchoring

Đề xuất hiện tại gồm hai giai đoạn.

Ở Stage 1, ảnh và text được mã hóa tách biệt. CLIP image encoder và CLIP text encoder được giữ frozen; phần trainable chính là các token prompt theo từng lớp cảm xúc. Mục tiêu là học bộ text anchor cho 7 cảm xúc sao cho feature ảnh và feature text cùng lớp có độ tương đồng cao hơn. Trong code hiện tại, Stage 1 dùng cached image features và cross-entropy trên similarity logits, chỉ cập nhật prompt learner.

Ở Stage 2, các text anchor đã học được cố định và chuyển sang huấn luyện nhánh visual. Nhánh ảnh dùng CLIP ViT với expression adapters, classifier head, alignment toàn cục giữa global image feature và text anchors, cùng alignment cục bộ dựa trên top-k patch tokens. Logit cuối là fusion giữa classifier, global alignment và local alignment.

Phần uncertainty hiện tại được triển khai theo hướng evidential output: logits được chuyển thành evidence, alpha và xác suất Dirichlet; uncertainty được tính từ tổng evidence. Vì vậy khi trình bày em sẽ nói đây là cơ chế calibration/evidence hiện tại, không khẳng định nó đã giải quyết hoàn toàn robustness.

## Slide 7 - Phần đã triển khai so với baseline

Slide này tóm tắt đóng góp theo khối chức năng.

Nhóm bổ sung mới trong W4 gồm dataset và manifest cho FER2013/HF và RAF-DB, emotion descriptors thông qua prompt learner cho 7 cảm xúc, expression adapters trong CLIP ViT, global-local alignment, và các artifact đánh giá như benchmark table, validation curves và per-class F1.

Nhóm chỉnh sửa từ W3 gồm chuyển objective từ ReID sang FER, giữ protocol hai giai đoạn nhưng tách rõ phần frozen và trainable, thiết kế loss gồm classification, alignment và uncertainty term, mở rộng evaluation protocol với macro-F1, balanced accuracy, per-class F1, ECE và risk AUC, đồng thời sửa cách viết báo cáo để nhận xét chính nằm trong bảng và caption.

Điểm em muốn nhấn mạnh là tiến độ này đã chuyển baseline từ mức ý tưởng sang một pipeline FER có thể chạy và xuất metric, nhưng chưa nên trình bày như một mô hình hoàn tất về mặt học thuật vì vẫn thiếu ablation và phân tích subset khó.

## Slide 8 - Kiến trúc hiện tại

Kiến trúc hiện tại có ba nhánh logic.

Nhánh thứ nhất là visual feature branch. Ảnh khuôn mặt đi qua tiền xử lý an toàn cho FER như resize, align và augmentation nhẹ, sau đó vào CLIP ViT. Mô hình hiện tại thêm expression adapters vào ViT và chủ yếu giữ backbone ở trạng thái đóng băng, chỉ fine-tune adapter và một số block cuối theo cấu hình. Output của nhánh này gồm global feature `z_g` và patch tokens `P`.

Nhánh thứ hai là semantic descriptor branch. Đầu vào là 7 nhãn cảm xúc chuẩn: anger, disgust, fear, happiness, sadness, surprise và neutral. Prompt learner học context token theo từng lớp, sau đó qua frozen CLIP text encoder để tạo emotion descriptors `T`. Trong repo hiện tại, AU được parse và lưu như metadata, nhưng chưa dùng như một nhánh AU fusion đầy đủ; vì vậy phần AU nên nói là hướng mở rộng hoặc ablation tiếp theo.

Nhánh thứ ba là decision, uncertainty và evaluation. Prediction heads kết hợp classifier với alignment global/local. Fusion logits tạo dự đoán cuối; evidential output tạo probability và uncertainty. Evaluation hiện tại báo cáo accuracy, macro-F1, ECE và risk AUC, phù hợp hơn với FER ngoài thực tế so với chỉ dùng accuracy.

## Slide 9 - Kết quả hiện tại

Kết quả hiện tại được chọn theo best macro-F1 checkpoint.

Trên RAF-DB, mô hình đạt accuracy khoảng 87.8%, balanced accuracy 80.5%, macro-F1 81.2% và ECE khoảng 0.39 trên 3,068 mẫu ở tập đánh giá theo kết quả tổng hợp. Điều này cho thấy RAF-DB hiện là dataset ổn định hơn trong thí nghiệm này.

Trên FER2013, mô hình đạt accuracy khoảng 70.6%, balanced accuracy 68.3%, macro-F1 68.6% và ECE khoảng 0.37 trên 7,178 mẫu validation. FER2013 thấp hơn là hợp lý vì ảnh thường low-resolution, label nhiễu hơn, và trong pipeline hiện tại validation gộp public/private test.

Điểm cần nói cẩn thận là các con số này chứng minh pipeline hiện tại chạy được và có baseline thực nghiệm, nhưng chưa chứng minh mô hình vượt SOTA hay robustness. Để kết luận mạnh hơn, cần thêm so sánh baseline và ablation.

## Slide 10 - Lỗi theo lớp

Khi nhìn per-class F1, lỗi không phân bố đều giữa các lớp.

Trên FER2013, lớp fear có F1 khoảng 0.53, là lớp yếu nhất. Điều này phù hợp với trực giác vì fear dễ bị lẫn với sadness hoặc surprise, nhất là khi ảnh nhỏ và nhãn mơ hồ. Lớp sadness trên FER2013 cũng chỉ khoảng 0.60, cho thấy vẫn còn ảnh hưởng của ambiguity.

Trên RAF-DB, lớp disgust có F1 khoảng 0.62, thấp hơn các lớp khác. Đây thường là lớp ít mẫu hơn và biểu hiện tinh vi hơn. Lớp fear trên RAF-DB đạt khoảng 0.71, tốt hơn FER2013 nhưng vẫn là lớp cần phân tích thêm.

Vì vậy bước tiếp theo nên bổ sung confusion matrix, phân tích subset che khuất hoặc lệch pose, và retrieval ảnh-descriptor để kiểm tra text anchor có thật sự kéo visual feature theo nghĩa cảm xúc hay không.

## Slide 11 - Tự phản biện học thuật

Slide cuối là phần em tự đặt câu hỏi trước khi bị hỏi trong buổi bảo vệ.

Câu hỏi thứ nhất: descriptor có thật sự mang nghĩa không, hay chỉ là embedding phân loại? Cách kiểm chứng là so sánh class-name prompt, learned descriptor và AU descriptor, đồng thời xem retrieval giữa ảnh và descriptor.

Câu hỏi thứ hai: text anchor có thật sự kéo visual feature không? Nếu chỉ tăng accuracy thì chưa đủ. Cần ablation như bỏ image-to-text alignment, bỏ fixed descriptor hoặc chỉ dùng classifier head.

Câu hỏi thứ ba: AU hoặc pseudo-AU có đáng tin không? Vì AU có thể nhiễu khi ảnh mờ, che khuất hoặc lệch pose, nên cần thí nghiệm with/without AU, confidence threshold và missing-AU mask. Với code hiện tại, AU chưa phải thành phần fusion chính, nên em sẽ không claim quá mức.

Câu hỏi thứ tư: uncertainty có thật sự hữu ích không? Nó có thể giúp calibration, nhưng cũng làm training phức tạp. Vì vậy cần báo cáo thêm ECE, NLL, risk-coverage hoặc uncertainty-quality curve, thay vì chỉ nói mô hình "biết khi nào không chắc".

Kết luận của em là hướng này đã có pipeline và kết quả ban đầu, nhưng contribution học thuật cần được khóa bằng ablation và phân tích lỗi. Đây sẽ là phần ưu tiên tiếp theo.

## Ghi chú để tránh claim sai

- Không nói mô hình đã đạt SOTA; slide hiện chỉ có kết quả nội bộ, chưa có bảng so sánh chuẩn.
- Không nói đã chứng minh robustness; hiện mới có metric tổng, calibration và per-class F1, còn thiếu occlusion/pose subset.
- Không nói AU đã được dùng đầy đủ trong fusion; repo hiện parse và lưu AU metadata, còn AU descriptor cần ablation riêng.
- Không nói uncertainty đã giải quyết label ambiguity; hiện chỉ nên nói đã có evidential output và ECE/risk AUC để bắt đầu đo calibration.
- Khi bị hỏi về điểm mới, nên trả lời là chuyển CLIP-ReID sang FER bằng semantic emotion anchors, expression adapters, global-local alignment và evaluation protocol phù hợp FER.
