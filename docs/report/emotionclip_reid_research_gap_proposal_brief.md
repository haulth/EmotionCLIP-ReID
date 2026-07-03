# EmotionCLIP-ReID - Research Gap & Proposal Brief

Ngày lập: 2026-05-18

## Luận điểm

EmotionCLIP-ReID nên được bảo vệ như một hướng giải quyết cho FER in-the-wild: dùng khung hai giai đoạn của CLIP-ReID làm scaffold, nhưng định nghĩa anchor bằng emotion/AU semantic descriptors để visual features bám vào tín hiệu biểu cảm có nghĩa.

## Research gaps

| Gap | Bối cảnh / khoảng trống cụ thể | Hệ quả với FER in-the-wild | Cách trả lời và bằng chứng cần có |
|---|---|---|---|
| G1. Descriptor ngữ nghĩa biểu cảm còn yếu | Nhiều hướng CLIP-FER chỉ dùng class-name prompt như happy/sad/angry hoặc mô tả ngắn ở cấp lớp. Các prompt này chưa mô tả rõ AU/vùng cơ mặt; còn ID-token của CLIP-ReID là anchor định danh, không mang nghĩa biểu cảm. | Mô hình có thể tăng accuracy nhưng khó giải thích vì không biết visual feature đang bám vào tín hiệu mắt, miệng, lông mày hay chỉ học shortcut theo dataset. Khi bị che khuất hoặc lệch góc nhìn, tín hiệu mỏng này dễ mất ổn định. | Xây dựng emotion/AU semantic descriptors gồm emotion label, mô tả vùng cơ mặt và AU/pseudo-AU có điều kiện. So sánh class prompt, learned emotion descriptor và AU descriptor; kiểm bằng macro-F1, balanced accuracy, retrieval ảnh-descriptor và visualization. |
| G2. Descriptor chưa trở thành anchor huấn luyện | CLIP-FER/adapters thường dùng text như cue phụ hoặc prompt để cải thiện biểu diễn, nhưng chưa cố định descriptor thành anchor ổn định dẫn hướng image encoder. CLIP-ReID có ý tưởng two-stage anchoring, nhưng anchor gốc phục vụ ID, không phục vụ FER. | Khó chứng minh text side thật sự kéo visual embedding về vùng ngữ nghĩa biểu cảm. Nếu chỉ cải thiện classifier, đóng góp sẽ bị hiểu là thêm adapter/loss chứ chưa giải quyết robustness và interpretability. | Giữ scaffold hai giai đoạn: Stage 1 học descriptor khi đóng băng encoder; Stage 2 cố định descriptor và fine-tune visual branch bằng L_cls + beta L_i2t. Ablation: no-fixed descriptor, no-I2T, class prompt, learned descriptor. |
| G3. AU/FACS hữu ích nhưng không luôn sẵn | AU/FACS giúp diễn giải biểu cảm, nhưng nhiều FER dataset không có AU label; pseudo-AU có thể nhiễu khi mặt bị che, pose lệch, ảnh mờ hoặc ánh sáng kém. Nếu bắt buộc AU thật, proposal dễ bị phụ thuộc detector ngoài. | Phạm vi áp dụng bị hẹp và dễ bị phản biện về nguồn nhãn AU. Pseudo-AU nhiễu có thể làm descriptor sai hướng, khiến kết quả kém hơn dù ý tưởng có vẻ hợp lý. | Trong v1, emotion descriptor là lõi; AU/pseudo-AU là prior có điều kiện và ablation riêng. Dùng confidence threshold/missing-AU mask, báo cáo with/without AU, phân tích noise sensitivity và lỗi theo AU khó. |
| G4. Robustness dễ bị claim chung chung | FER in-the-wild chịu occlusion, pose/viewpoint shift, illumination, low-resolution, domain shift và label ambiguity. Nhiều nghiên cứu vẫn chủ yếu báo accuracy tổng; uncertainty hoặc multimodal nếu đưa quá sớm sẽ làm lệch trọng tâm. | Accuracy tổng không đủ chứng minh mô hình bền vững hoặc có thể diễn giải. Mô hình có thể overconfident trên mẫu mơ hồ/che khuất, trong khi claim robustness thiếu bằng chứng theo subset khó. | Tạo hoặc lọc subset occlusion/pose/low-confidence; báo cáo macro-F1/balanced accuracy theo subset, confusion matrix, ECE/NLL và uncertainty-quality curve. Chỉ giữ uncertainty khi cải thiện calibration sau baseline semantic alignment. |

## Research problem

Trong điều kiện thực tế (in-the-wild), nhận dạng biểu cảm khuôn mặt thường kém ổn định vì tín hiệu biểu cảm cục bộ bị che khuất, biến dạng bởi thay đổi góc nhìn, nhiễu bởi domain shift và chịu ảnh hưởng của nhãn cảm xúc mơ hồ. Vấn đề nghiên cứu là xây dựng một mô hình FER có khả năng nhận diện ổn định hơn và có thể diễn giải dựa trên tín hiệu biểu cảm có nghĩa, thay vì chỉ tối ưu độ chính xác tổng trên ảnh tương đối sạch.

Tự phản biện: câu "chuyển cơ chế CLIP-ReID từ ID-token anchoring sang emotion/AU descriptor anchoring" mô tả lựa chọn thiết kế, không phải research problem. Câu này chỉ nên xuất hiện trong proposal/method, sau khi đã đặt vấn đề từ khó khăn thực tế của FER in-the-wild.

## Proposal hướng hiện tại

- Stage 1: học emotion/AU descriptor bằng CLIP text encoder trong khi giữ cố định encoder.
- Stage 2: cố định descriptor làm semantic anchors, fine-tune visual adapter/head bằng classification + image-text alignment.
- AU và uncertainty là nhánh ablation/nâng cao, không phải điều kiện bắt buộc của v1.
