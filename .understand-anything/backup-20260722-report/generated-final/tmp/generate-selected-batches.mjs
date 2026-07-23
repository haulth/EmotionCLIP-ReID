import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const intermediate = path.join(root, '.understand-anything', 'intermediate');
const temp = path.join(root, '.understand-anything', 'tmp');
const batchesDoc = JSON.parse(fs.readFileSync(path.join(intermediate, 'batches.json'), 'utf8'));
const requested = [2, 4, 6, 8, 11, 18];

const fileSummaries = {
  'config/emotion_defaults.py': 'Định nghĩa cấu hình mặc định, cơ chế merge/override và các ràng buộc fail-closed cho toàn bộ pipeline EmotionCLIP-ReID, bao gồm anatomy routing, geometry fusion, uncertainty và hai giai đoạn huấn luyện.',
  'datasets/anatomy.py': 'Chuẩn hóa artifact landmark độc lập detector thành ba vùng giải phẫu, descriptor hình học fixed-width cùng mask validity/uncertainty. Module xử lý biến đổi crop/flip và ước lượng thống kê hình học theo lớp mà không nội suy điểm thiếu.',
  'datasets/emotion_manifest.py': 'Cài đặt hợp đồng manifest JSONL cho bảy lớp FER, kiểm tra leakage/coverage anatomy, phép biến đổi ảnh an toàn với landmark và các DataLoader tách biệt train/validation/test.',
  'infer_emotionclip.py': 'CLI suy luận EmotionCLIP-ReID cho một ảnh, dựng mô hình từ cấu hình, nạp checkpoint và tùy chọn artifact landmark trước khi xuất dự đoán, confidence và uncertainty.',
  'loss/emotion_losses.py': 'Tập hợp objective Stage 2 gồm classification/alignment, Evidential Deep Learning, reliability correctness/ranking và regularization cho gate, temperature và routing.',
  'processor/processor_emotionclip.py': 'Bộ điều phối huấn luyện và đánh giá hai giai đoạn: quản lý stage/phase, corruption reliability, checkpoint có version, fail-fast khi non-finite, model selection, sealed test và ghi metric/artifact.',
  'tests/test_anatomy_audit.py': 'Kiểm thử đơn vị cho pipeline audit geometry, đặc biệt việc truyền jitter landmark sang uncertainty theo đúng đơn vị của geometry feature.',
  'tests/test_emotion_losses_metrics.py': 'Kiểm thử toán học cho decoupled Dirichlet, gradient của loss Stage 2, FER/calibration metrics và khả năng tách clean–corrupted bằng uncertainty.',
  'tests/test_emotion_manifest.py': 'Kiểm thử chuẩn hóa nhãn, chống split leakage, converter FER2013/RAF-DB, biến đổi landmark theo crop/flip và fallback anatomy có quality bằng không.',
  'tests/test_emotion_processor_smoke.py': 'Smoke/integration tests trên CPU cho Stage 1–2, gradient accumulation, non-finite guard, early stopping, checkpoint contract, sealed test và corruption anatomy.',
  'tests/test_run_artifacts.py': 'Kiểm thử tính bất biến của thư mục run, metadata provenance và yêu cầu run ID tường minh khi tra artifact.',
  'tests/test_train_emotionclip_parallel.py': 'Kiểm thử parse GPU, chính sách topology, unwrap model và các invariant của EmotionDataParallel đối với output dùng chung.',
  'tools/audit_anatomy_geometry.py': 'CLI audit coverage và signal-to-jitter của artifact anatomy để phát hiện descriptor hình học yếu hoặc không ổn định trước huấn luyện.',
  'tools/build_face_landmark_artifacts.py': 'Xây artifact landmark MediaPipe có version, pose/quality, jitter-derived uncertainty và đường dẫn portable cho manifest FER.',
  'tools/convert_affwild2_to_emotion_jsonl.py': 'Chuyển nhãn Aff-Wild2 thành manifest JSONL EmotionCLIP-ReID, bảo toàn split và metadata khung hình/video.',
  'tools/convert_fer2013_to_emotion_jsonl.py': 'Chuyển FER2013 từ CSV pixel hoặc cây thư mục ảnh thành ảnh chuẩn và manifest JSONL với ánh xạ split/nhãn chính thức.',
  'tools/convert_raf_au_to_emotion_jsonl.py': 'Bổ sung nhãn Action Unit của RAF vào bản ghi manifest để phục vụ nhánh anatomy/AU và phân tích vi mô.',
  'tools/convert_rafdb_to_emotion_jsonl.py': 'Giải nén, tìm nhãn và chuyển RAF-DB thành manifest; hỗ trợ official train/test cùng validation stratified xác định bởi seed.',
  'tools/download_hf_emotion_dataset.py': 'Tải dataset cảm xúc từ Hugging Face, chuẩn hóa nhãn/split, lưu ảnh cục bộ và tạo manifest JSONL có giới hạn mẫu tùy chọn.',
  'train_emotionclip.py': 'Entry point huấn luyện EmotionCLIP-ReID: hợp nhất cấu hình, cố định seed, thiết lập GPU an toàn, dựng dữ liệu/mô hình và điều phối Stage 1, Stage 2 cùng sealed test và immutable run.',
  'utils/fer_metrics.py': 'Tính bộ metric FER và uncertainty toàn diện: confusion matrix, per-class F1, balanced accuracy, ECE/ACE/classwise calibration, NLL/Brier, selective risk và OOD ranking.',
  'utils/run_artifacts.py': 'Tạo thư mục thí nghiệm immutable và ghi provenance gồm hash manifest/split, git state, dependency, phần cứng và cấu hình để tái lập kết quả.',
  'model/clip/__init__.py': 'Package entry point tái xuất API tải mô hình và tokenize của backbone CLIP nội bộ.',
  'model/clip/clip.py': 'Tiện ích quản lý model registry, tải weight, preprocessing, JIT và tokenize cho implementation CLIP được nhúng trong dự án.',
  'model/clip/model.py': 'Implementation PyTorch của CLIP gồm Modified ResNet/Vision Transformer, text Transformer, attention pooling và logic dựng model từ state dict với resize positional embedding.',
  'model/clip/simple_tokenizer.py': 'Tokenizer byte-pair encoding tương thích OpenAI CLIP, bao gồm ánh xạ byte Unicode, làm sạch văn bản, encode và decode.',
  'model/emotionclip_model.py': 'Kiến trúc lõi EmotionCLIP-ReID mở rộng CLIP bằng learnable emotion prompts, zero-initialized expression adapters, anatomy-aware patch routing, geometry residual fusion, branch calibration và decoupled reliability/Dirichlet uncertainty.',
  'model/make_model.py': 'Factory baseline ReID dùng CLIP visual backbone với classifier và bottleneck cho nhận dạng người/xe, giữ đường nạp checkpoint của codebase gốc.',
  'model/make_model_clipreid.py': 'Factory CLIP-ReID baseline với PromptLearner, TextEncoder và transformer visual để học text descriptor hai giai đoạn cho re-identification.',
  'tests/test_emotion_model_units.py': 'Kiểm thử cấu trúc và invariant của EmotionCLIPModel: prompt/adapters, fusion simplex/temperature, stage freezing, anatomy router, geometry fallback, uncertainty và forward tích hợp.',
  'README.md': 'Tài liệu vào dự án mô tả baseline CLIP-ReID và phần mở rộng EmotionCLIP-ReID cho FER, từ cài đặt/dữ liệu đến landmark artifacts, huấn luyện hai giai đoạn, inference, kiểm thử và tái lập.',
  'environment_emotionclip_cuda.yml': 'Môi trường Conda cho EmotionCLIP-ReID trên Python 3.10 và CUDA 12.1, kèm PyTorch, MediaPipe, Hugging Face datasets, pytest và các dependency tiền xử lý.',
  'note_edit.md': 'Ghi chú thiết kế và chẩn đoán Stage 2, phân tích temperature, mất cân bằng, corruption clean/shifted và lộ trình calibration–anatomy bootstrap–adapter fine-tuning–rebalancing.',
  'pytest.ini': 'Cấu hình pytest giới hạn discovery vào thư mục tests để bộ kiểm thử dự án chạy nhất quán.',
  'configs/emotion/vit_b16_emotionclip.yml': 'Preset đầy đủ cho ViT-B/16 EmotionCLIP-ReID với anatomy bắt buộc, hybrid routing, gated geometry residual, decoupled uncertainty, calibrated branch fusion và loss/corruption cho Stage 1–2.',
  'configs/emotion/vit_b16_emotionclip_fer2013_quick.yml': 'Preset FER2013 từ manifest nội bộ, chạy 200 epoch hai giai đoạn với anatomy quality, hybrid routing và geometry fusion để thử nghiệm nhanh có checkpoint mỗi epoch.',
  'configs/emotion/vit_b16_emotionclip_hf_fer2013_quick.yml': 'Preset cho bản FER2013 tải từ Hugging Face, dùng cùng hợp đồng anatomy và lịch Stage 1 base+geometry rồi Stage 2 của pipeline quick.',
  'configs/emotion/vit_b16_emotionclip_rafdb_quick.yml': 'Preset RAF-DB có AMP, gradient accumulation, clipping/non-finite guard và early stopping; vẫn yêu cầu anatomy coverage và tách validation/test nghiêm ngặt.',
  'docs/report/emotionclip_reid_feasibility_analysis.md': 'Phân tích khả thi toàn diện từ CLIP-ReID sang FER, đối chiếu nguồn tham khảo, hiện trạng code, nhánh AU/anatomy, dataset, rủi ro leakage, lộ trình triển khai và tiêu chí chấp nhận.',
  'docs/report/emotionclip_reid_model_proposal_report.md': 'Báo cáo đề xuất mô hình hai giai đoạn, nêu phần kế thừa từ CLIP-ReID, MER-CLIP, EA-CLIP và UA-FER, kiến trúc semantic anchoring, kế hoạch ablation và giới hạn claim.',
  'docs/report/emotionclip_reid_pipeline_diagram_script.md': 'Kịch bản thuyết minh sơ đồ pipeline, giải thích từng khối và luồng tensor của Stage 1 prompt/emotion anchor và Stage 2 visual adaptation, anatomy fusion, prediction và uncertainty.',
  'docs/report/emotionclip_reid_presentation_script.md': 'Kịch bản trình bày theo slide về bài toán, related work, research gap, phần đã triển khai, kiến trúc, kết quả, lỗi theo lớp và các cảnh báo tránh claim quá mức.',
  'docs/report/emotionclip_reid_research_gap_proposal_brief.md': 'Bản tóm tắt luận điểm nghiên cứu, các khoảng trống về local facial evidence, uncertainty và explainability, cùng hướng đề xuất two-stage EmotionCLIP-ReID.',
  'outputs/report_w4/emotionclip_reid_w4_uncertainty_detail.drawio': 'Sơ đồ chi tiết nhánh uncertainty W4 từ fused logits qua decoupled Dirichlet strength đến probability, uncertainty, EDL objective, ECE và uncertainty-risk AUC.',
  'processor/__init__.py': 'Đánh dấu processor là Python package; các vòng lặp huấn luyện cụ thể được đặt trong các module processor chuyên biệt.',
  'solver/__init__.py': 'Đánh dấu solver là Python package cho optimizer và learning-rate scheduler của dự án.',
  'utils/__init__.py': 'Đánh dấu utils là Python package tập hợp metric, logging, notebook helper và artifact reproducibility.',
  'utils/notebook_metrics.py': 'Các helper notebook dò metric từ JSON/CSV/log, hợp nhất lịch sử validation/training và vẽ curve, confusion matrix, per-class F1 với lựa chọn best epoch.',
};

const symbolSummaries = {
  'anatomy_requirement_reasons': 'Suy ra các lý do cấu hình hiện tại bắt buộc phải có anatomy evidence để chặn fallback ngoài ý muốn.',
  'validate_emotion_cfg': 'Kiểm tra chéo các ràng buộc Stage 1–2, anatomy, routing, geometry và uncertainty; báo lỗi sớm cho tổ hợp cấu hình không hợp lệ.',
  'load_emotion_cfg': 'Nạp YAML, merge với mặc định và CLI overrides, ghi provenance nguồn rồi xác thực cấu hình hiệu dụng.',
  'geometry_feature_definition_mask': 'Tạo mask các slot geometry có định nghĩa thực cho từng vùng giải phẫu.',
  'apply_horizontal_flip_to_geometry': 'Đổi vai trái/phải của geometry feature, validity và uncertainty sau phép lật ngang.',
  'transform_normalized_landmarks': 'Biến đổi landmark chuẩn hóa theo resize/crop/flip đồng bộ với ảnh đầu vào.',
  '_geometry_features': 'Tính descriptor hình học theo ba vùng mặt cùng validity và uncertainty ở cấp feature.',
  'empty_anatomy_inputs': 'Tạo tensor anatomy rỗng có shape cố định và quality bằng không cho fallback tường minh.',
  'anatomy_to_model_inputs': 'Chuyển artifact landmark thành tensor region points, geometry, mask, quality và uncertainty mà mô hình tiêu thụ.',
  'fit_class_geometry_statistics': 'Ước lượng mean/scale geometry theo lớp từ mẫu đủ quality để condition prompt Stage 1b.',
  'normalize_emotion': 'Chuẩn hóa ID hoặc alias nhãn về bảy emotion canonical và ID ổn định.',
  'load_emotion_manifest': 'Đọc manifest JSONL, kiểm tra từng record và lọc theo split thành các EmotionSample.',
  'validate_split_leakage': 'Phát hiện video/subject xuất hiện ở nhiều split để ngăn leakage đánh giá.',
  'summarize_anatomy_coverage': 'Tổng hợp coverage landmark, vùng hợp lệ và failure mode của anatomy trên tập mẫu.',
  'validate_anatomy_coverage': 'Áp chính sách coverage/fallback từ cấu hình và fail closed khi anatomy không đạt yêu cầu.',
  'emotion_collate_fn': 'Ghép ảnh, nhãn, metadata và cấu trúc anatomy lồng nhau thành mini-batch.',
  'make_emotion_dataloaders': 'Dựng DataLoader riêng cho Stage 1, Stage 2, validation và sealed test sau khi kiểm tra split/anatomy.',
  'dirichlet_kl_to_uniform': 'Tính KL divergence giữa Dirichlet và prior đồng đều để regularize evidence.',
  'evidential_ce_loss': 'Tính evidential cross-entropy với KL annealing trên tham số Dirichlet đã dựng.',
  'dirichlet_statistics': 'Suy ra strength, Dirichlet mean và epistemic uncertainty từ alpha.',
  'reliability_correctness_loss': 'Huấn luyện reliability head dự đoán tính đúng của quyết định classifier đã detach.',
  'reliability_ranking_loss': 'Ép strength của mẫu clean cao hơn mẫu shifted/corrupted theo margin.',
  'emotion_stage2_loss': 'Kết hợp classification, semantic alignment, reliability/EDL và các regularizer Stage 2 thành total loss có log thành phần.',
  'corrupt_images_for_reliability': 'Sinh ảnh nhiễu/che vùng và occlusion mask để học clean–shifted reliability ranking.',
  'corrupt_anatomy_for_reliability': 'Làm mất hiệu lực anatomy evidence trùng vùng ảnh bị che để tránh tín hiệu hình học không nhất quán.',
  'log_training_event': 'Ghi sự kiện huấn luyện dạng JSON có cấu trúc để notebook và audit có thể parse ổn định.',
  'load_emotion_checkpoint': 'Nạp checkpoint có version, kiểm tra chữ ký cấu hình/stage và từ chối model chưa đủ điều kiện inference.',
  'precompute_stage1_features': 'Cache visual features và anatomy cho Stage 1 nhằm tránh lặp backbone không cần thiết.',
  'evaluate_stage1_prompt_model': 'Đánh giá descriptor prompt Stage 1 với hoặc không có geometry conditioning.',
  'do_train_emotion_stage1': 'Huấn luyện Stage 1 theo các phase base prompt rồi geometry residual, chọn checkpoint bằng validation.',
  'evaluate_emotion_model': 'Chạy đánh giá FER và thu logits, uncertainty, routing/region diagnostics cùng calibration metrics.',
  'evaluate_sealed_test': 'Mở sealed test chỉ với checkpoint đã chọn từ validation và ghi rõ selection split.',
  'do_train_emotion_stage2': 'Huấn luyện visual adapters/fusion/reliability Stage 2 với AMP, accumulation, corruption ranking, early stopping và checkpointing.',
  'decoupled_dirichlet': 'Dựng Dirichlet trong đó class probability lấy từ softmax còn evidence strength do reliability head độc lập quyết định.',
  'load_clip_to_cpu': 'Nạp weight CLIP và dựng backbone theo độ phân giải/stride trước khi chuyển thiết bị.',
  'install_expression_adapters': 'Chèn residual expression adapters zero-initialized vào các block của visual encoder.',
  'normalized_fusion_prior': 'Chuẩn hóa trọng số prior của các nhánh fusion thành simplex hợp lệ.',
  'validate_train_last_blocks': 'Xác thực số visual transformer block cuối được phép fine-tune.',
  'convert_weights': 'Chuyển các lớp phù hợp của CLIP sang half precision như implementation gốc.',
  'build_model': 'Suy luận kiến trúc từ state dict, dựng CLIP và nạp weight đã chuẩn hóa.',
  'resize_pos_embed': 'Nội suy positional embedding của visual transformer khi thay đổi lưới patch.',
  'tokenize': 'Mã hóa một hoặc nhiều chuỗi thành token tensor có context length cố định của CLIP.',
  'load': 'Tải hoặc mở weight CLIP, dựng JIT/state-dict model và preprocessing tương ứng.',
  'compute_fer_metrics': 'Tổng hợp accuracy, macro/per-class F1, calibration, selective risk và OOD metrics thành một payload đánh giá.',
  'ood_detection_metrics': 'Tính AUROC/AUPR/FPR95 để đo khả năng uncertainty tách in-distribution và OOD.',
  'initialize_immutable_run': 'Tạo run directory không ghi đè và lưu config, manifest hashes, git/dependency/hardware provenance.',
  'artifact_dir': 'Tra thư mục artifact của một run ID tường minh, không ngầm chọn kết quả mới nhất.',
  'load_validation_metrics': 'Ưu tiên nạp metric validation từ JSON theo epoch rồi fallback sang các CSV/summary cũ.',
  'plot_validation_metric_curves': 'Vẽ các đường accuracy, F1, calibration và uncertainty theo epoch, đánh dấu best checkpoint.',
  'load_training_history': 'Tìm và parse lịch sử training từ log có cấu trúc hoặc CSV tương thích ngược.',
  'plot_training_metric_curves': 'Vẽ loss/learning-rate/diagnostic của Stage 1–2 và lưu hình vào result directory.',
  'plot_confusion_matrix_and_f1': 'Vẽ confusion matrix cùng per-class F1 từ metric bundle đã hợp nhất.',
};

const classSummaries = {
  'EmotionSample': 'Dataclass bất biến biểu diễn một mẫu FER cùng split, nhãn, định danh video/chủ thể, AU, landmark và metadata.',
  'FaceSafeTransform': 'Phép biến đổi ảnh đồng bộ resize/crop/flip/color jitter với anatomy để không phá quan hệ hình học khuôn mặt.',
  'EmotionManifestDataset': 'PyTorch Dataset đọc EmotionSample, ảnh và artifact anatomy, trả về input có fallback quality tường minh.',
  'EmotionDataParallel': 'Biến thể DataParallel kiểm soát scatter/gather cho batch anatomy và bảo toàn các output dùng chung như text features/temperature.',
  'Bottleneck': 'Residual bottleneck block của Modified ResNet trong CLIP visual encoder.',
  'AttentionPool2d': 'Pooling đặc trưng không gian bằng multi-head attention với positional embedding học được.',
  'ModifiedResNet': 'Backbone ResNet biến đổi theo CLIP, dùng three-convolution stem, anti-aliasing và attention pooling.',
  'LayerNorm': 'LayerNorm tương thích mixed precision bằng cách tính ổn định ở float rồi trả về dtype gốc.',
  'QuickGELU': 'Hàm kích hoạt x·sigmoid(1.702x) dùng trong Transformer CLIP.',
  'ResidualAttentionBlock': 'Block Transformer gồm self-attention, MLP và residual pre-norm.',
  'Transformer': 'Chuỗi ResidualAttentionBlock dùng cho encoder văn bản hoặc thị giác CLIP.',
  'VisionTransformer': 'Visual Transformer biến ảnh thành patch tokens, cộng positional embedding và sinh global/local features.',
  'CLIP': 'Mô hình contrastive vision-language hoàn chỉnh với image encoder, text encoder và learnable logit scale.',
  'SimpleTokenizer': 'BPE tokenizer của CLIP với vocabulary/merge cache và API encode/decode.',
  'ExpressionAdapter': 'Residual bottleneck adapter zero-initialized để thích nghi visual token cho biểu cảm mà ít phá backbone.',
  'EmotionTextEncoder': 'Bao bọc text Transformer CLIP để mã hóa prompt embedding học được thành class descriptors.',
  'AnatomyPromptResidual': 'Conditioner geometry/quality sinh residual cho emotion prompts, hỗ trợ mode/phase và thống kê hình học theo lớp.',
  'EmotionPromptLearner': 'Học context token cho bảy lớp cảm xúc và tùy chọn bổ sung geometry-conditioned prompt residual.',
  'BranchFusion': 'Hiệu chỉnh và hợp nhất classifier, global alignment và local alignment logits bằng gate/simplex cùng temperature bị chặn.',
  'RegionPatchRouter': 'Định tuyến patch token tới upper/middle/lower facial regions bằng hybrid anatomy attention và fallback tự do theo quality.',
  'AnatomyRegionFusion': 'Fusion feature vùng với descriptor hình học qua gated residual, đồng thời đo disagreement có trọng số reliability.',
  'EmotionCLIPModel': 'Mô hình đầu-cuối quản lý hai stage, prompt descriptors, visual adapters, anatomy routing/geometry fusion, ba nhánh logits và reliability uncertainty tách rời.',
  'TextEncoder': 'Text encoder của baseline CLIP-ReID dùng Transformer CLIP trên prompt embedding.',
  'PromptLearner': 'Học token prompt theo identity cho Stage 1 của CLIP-ReID baseline.',
  'build_transformer': 'Mô hình ReID/CLIP-ReID bao bọc backbone, bottleneck, classifier và logic forward/load checkpoint.',
};

function fileTags(file, category) {
  const p = file.path;
  if (category === 'docs') return ['tài-liệu', 'nghiên-cứu', p.includes('report/') ? 'báo-cáo' : 'hướng-dẫn', 'emotionclip-reid'];
  if (category === 'config') return ['cấu-hình', p.endsWith('.yml') ? 'yaml' : 'test-config', 'reproducibility', 'emotionclip-reid'];
  if (p.includes('/tests/') || p.startsWith('tests/')) return ['kiểm-thử', 'pytest', 'xác-minh', 'emotionclip-reid'];
  if (p.includes('emotionclip_model')) return ['mô-hình', 'emotionclip', 'anatomy', 'uncertainty'];
  if (p.startsWith('model/clip/')) return ['mô-hình', 'clip', 'vision-language', 'pytorch'];
  if (p.startsWith('model/')) return ['mô-hình', 'clip-reid', 'pytorch', 'factory'];
  if (p.includes('anatomy')) return ['anatomy', 'landmark', 'hình-học-khuôn-mặt', 'uncertainty'];
  if (p.includes('manifest')) return ['dữ-liệu', 'manifest', 'fer', 'validation'];
  if (p.includes('loss')) return ['hàm-mất-mát', 'uncertainty', 'tối-ưu', 'pytorch'];
  if (p.includes('processor')) return ['huấn-luyện', 'đánh-giá', 'checkpoint', 'pipeline'];
  if (p.includes('metrics')) return ['đánh-giá', 'calibration', 'fer', 'trực-quan-hóa'];
  if (p.includes('run_artifacts')) return ['reproducibility', 'provenance', 'artifact', 'thí-nghiệm'];
  if (p.startsWith('tools/')) return ['công-cụ', 'data-pipeline', 'cli', 'tiền-xử-lý'];
  if (p.startsWith('train') || p.startsWith('infer')) return ['entry-point', 'pipeline', 'fer', 'cli'];
  return ['python', 'module', 'emotionclip-reid'];
}

function domainTag(filePath) {
  if (filePath.includes('test_')) return 'kiểm-thử';
  if (filePath.includes('anatomy')) return 'anatomy';
  if (filePath.includes('manifest')) return 'dữ-liệu';
  if (filePath.includes('loss')) return 'hàm-mất-mát';
  if (filePath.includes('processor')) return 'huấn-luyện';
  if (filePath.includes('metrics')) return 'đánh-giá';
  if (filePath.includes('clip')) return 'clip';
  if (filePath.startsWith('tools/')) return 'data-pipeline';
  return 'pipeline';
}

function humanName(name) {
  return name.replace(/^_+/, '').replace(/_/g, ' ');
}

function functionSummary(filePath, fn) {
  if (symbolSummaries[fn.name]) return symbolSummaries[fn.name];
  if (fn.name === 'main') return `Điểm vào CLI điều phối ${fileSummaries[filePath].charAt(0).toLowerCase()}${fileSummaries[filePath].slice(1)}`;
  if (fn.name.startsWith('test_')) return `Kiểm thử hành vi “${humanName(fn.name.slice(5))}” để bảo vệ contract được mô tả bởi ${path.basename(filePath)}.`;
  if (fn.name.startsWith('plot_')) return `Trực quan hóa ${humanName(fn.name.slice(5))} và lưu kết quả phục vụ phân tích notebook/báo cáo.`;
  if (fn.name.startsWith('convert_')) return `Chuyển đổi ${humanName(fn.name.slice(8))} sang hợp đồng dữ liệu mà pipeline EmotionCLIP-ReID sử dụng.`;
  if (fn.name.startsWith('load_') || fn.name.startsWith('_load_')) return `Nạp và chuẩn hóa ${humanName(fn.name.replace(/^_?load_/, ''))} cho luồng xử lý của ${path.basename(filePath)}.`;
  if (fn.name.startsWith('save_') || fn.name.startsWith('_save_')) return `Lưu ${humanName(fn.name.replace(/^_?save_/, ''))} theo định dạng ổn định của pipeline.`;
  if (fn.name.startsWith('validate_')) return `Xác thực ${humanName(fn.name.slice(9))} và báo lỗi sớm khi contract đầu vào bị vi phạm.`;
  if (fn.name.startsWith('evaluate_')) return `Đánh giá ${humanName(fn.name.slice(9))} và trả về metric/diagnostic phục vụ lựa chọn mô hình.`;
  if (fn.name.startsWith('_')) return `Hàm hỗ trợ nội bộ xử lý ${humanName(fn.name)} trong ${path.basename(filePath)}, giữ logic chính tách biệt và có thể kiểm thử.`;
  return `Triển khai thao tác ${humanName(fn.name)} trong ${path.basename(filePath)} theo contract của pipeline EmotionCLIP-ReID.`;
}

function complexity(lines) {
  if (lines > 200) return 'complex';
  if (lines >= 50) return 'moderate';
  return 'simple';
}

function languageNotes(filePath) {
  const notes = {
    'datasets/anatomy.py': 'NumPy xử lý artifact/geometry trước khi chuyển sang tensor PyTorch; mọi feature đều đi cùng validity và uncertainty mask.',
    'datasets/emotion_manifest.py': 'Dataclass bất biến và custom collate duy trì metadata/anatomy lồng nhau qua DataLoader.',
    'processor/processor_emotionclip.py': 'Module dùng AMP/GradScaler, gradient accumulation và checkpoint schema versioned; DataParallel chỉ được hỗ trợ qua wrapper có invariant rõ ràng.',
    'model/clip/model.py': 'Implementation PyTorch theo OpenAI CLIP, có nội suy positional embedding để phù hợp stride và kích thước ảnh ReID/FER.',
    'model/emotionclip_model.py': 'Các nhánh mới dùng zero-initialized residual/gate để khởi đầu gần baseline; reliability strength được tách khỏi thang/offset logits phân lớp.',
    'utils/notebook_metrics.py': 'Helper ưu tiên artifact JSON mới nhưng duy trì fallback CSV/log cho các run lịch sử.',
    'configs/emotion/vit_b16_emotionclip.yml': 'YAML phân cấp tách MODEL, INPUT, DATASETS, SOLVER, TEST và TRAIN để CLI override theo dotted key.',
  };
  return notes[filePath];
}

function nodeIdForFile(file) {
  if (file.fileCategory === 'config') return `config:${file.path}`;
  if (file.fileCategory === 'docs') return `document:${file.path}`;
  return `file:${file.path}`;
}

function analyzeBatch(batchIndex) {
  const batch = batchesDoc.batches.find(item => item.batchIndex === batchIndex);
  if (!batch) throw new Error(`Không tìm thấy batch ${batchIndex}`);
  const extracted = JSON.parse(fs.readFileSync(path.join(temp, `ua-file-extract-results-${batchIndex}.json`), 'utf8'));
  if (!extracted.scriptCompleted || extracted.results.length !== batch.files.length) {
    throw new Error(`Extraction batch ${batchIndex} không đầy đủ`);
  }
  const resultByPath = new Map(extracted.results.map(item => [item.path, item]));
  const allNodes = [];
  const allEdges = [];
  const childIdsByFile = new Map();

  for (const file of batch.files) {
    const result = resultByPath.get(file.path);
    if (!result) throw new Error(`Thiếu extraction result cho ${file.path}`);
    const fileId = nodeIdForFile(file);
    const fileNode = {
      id: fileId,
      type: file.fileCategory === 'config' ? 'config' : file.fileCategory === 'docs' ? 'document' : 'file',
      name: path.basename(file.path),
      filePath: file.path,
      summary: fileSummaries[file.path] || `Thành phần ${path.basename(file.path)} của EmotionCLIP-ReID.`,
      tags: fileTags(file, file.fileCategory),
      complexity: complexity(result.nonEmptyLines),
    };
    const note = languageNotes(file.path);
    if (note) fileNode.languageNotes = note;
    allNodes.push(fileNode);

    const childIds = [];
    for (const fn of result.functions || []) {
      if (!fn.name) continue;
      const span = Math.max(1, fn.endLine - fn.startLine + 1);
      const id = `function:${file.path}:${fn.name}`;
      allNodes.push({
        id,
        type: 'function',
        name: fn.name,
        filePath: file.path,
        lineRange: [fn.startLine, fn.endLine],
        summary: functionSummary(file.path, fn),
        tags: file.path.includes('test_') ? ['kiểm-thử', 'pytest', domainTag(file.path)] : ['hàm', domainTag(file.path), 'python'],
        complexity: complexity(span),
      });
      childIds.push(id);
    }
    for (const cls of result.classes || []) {
      if (!cls.name) continue;
      const span = Math.max(1, cls.endLine - cls.startLine + 1);
      const id = `class:${file.path}:${cls.name}`;
      allNodes.push({
        id,
        type: 'class',
        name: cls.name,
        filePath: file.path,
        lineRange: [cls.startLine, cls.endLine],
        summary: classSummaries[cls.name] || `Lớp ${cls.name} hỗ trợ kịch bản ${domainTag(file.path)} trong ${path.basename(file.path)}.`,
        tags: file.path.includes('test_') ? ['kiểm-thử', 'test-double', 'pytorch'] : ['lớp', domainTag(file.path), 'pytorch'],
        complexity: complexity(span),
      });
      childIds.push(id);
    }
    childIdsByFile.set(file.path, childIds);
    for (const childId of childIds) {
      allEdges.push({source: fileId, target: childId, type: 'contains', direction: 'forward', weight: 1.0});
      allEdges.push({source: fileId, target: childId, type: 'exports', direction: 'forward', weight: 0.8});
    }
  }

  let expectedImports = 0;
  for (const file of batch.files) {
    if (file.fileCategory !== 'code') continue;
    const source = nodeIdForFile(file);
    const imports = batch.batchImportData[file.path] || [];
    expectedImports += imports.length;
    for (const targetPath of imports) {
      allEdges.push({source, target: `file:${targetPath}`, type: 'imports', direction: 'forward', weight: 0.7});
    }
  }
  const actualImports = allEdges.filter(edge => edge.type === 'imports').length;
  if (actualImports !== expectedImports) throw new Error(`Batch ${batchIndex}: imports ${actualImports}/${expectedImports}`);

  for (const file of batch.files.filter(item => item.path.startsWith('tests/'))) {
    for (const productionPath of batch.batchImportData[file.path] || []) {
      if (productionPath.startsWith('tests/')) continue;
      allEdges.push({source: `file:${productionPath}`, target: `file:${file.path}`, type: 'tested_by', direction: 'forward', weight: 0.5});
    }
  }

  const extraEdges = {
    6: [
      ['document:note_edit.md', 'document:README.md', 'related', 0.5],
      ['document:README.md', 'config:environment_emotionclip_cuda.yml', 'related', 0.5],
      ['config:pytest.ini', 'document:README.md', 'related', 0.5],
    ],
    8: [
      ['config:configs/emotion/vit_b16_emotionclip_fer2013_quick.yml', 'config:configs/emotion/vit_b16_emotionclip.yml', 'related', 0.5],
      ['config:configs/emotion/vit_b16_emotionclip_hf_fer2013_quick.yml', 'config:configs/emotion/vit_b16_emotionclip.yml', 'related', 0.5],
      ['config:configs/emotion/vit_b16_emotionclip_rafdb_quick.yml', 'config:configs/emotion/vit_b16_emotionclip.yml', 'related', 0.5],
    ],
    11: [
      ['document:docs/report/emotionclip_reid_model_proposal_report.md', 'document:docs/report/emotionclip_reid_feasibility_analysis.md', 'related', 0.5],
      ['document:docs/report/emotionclip_reid_pipeline_diagram_script.md', 'document:docs/report/emotionclip_reid_model_proposal_report.md', 'related', 0.5],
      ['document:docs/report/emotionclip_reid_presentation_script.md', 'document:docs/report/emotionclip_reid_model_proposal_report.md', 'related', 0.5],
      ['document:docs/report/emotionclip_reid_research_gap_proposal_brief.md', 'document:docs/report/emotionclip_reid_feasibility_analysis.md', 'related', 0.5],
    ],
    18: [
      ['file:outputs/report_w4/emotionclip_reid_w4_uncertainty_detail.drawio', 'file:utils/notebook_metrics.py', 'documents', 0.5],
    ],
  };
  for (const [source, target, type, weight] of extraEdges[batchIndex] || []) {
    allEdges.push({source, target, type, direction: 'forward', weight});
  }

  return {batch, extracted, nodes: allNodes, edges: allEdges};
}

function splitAndWrite(result) {
  const {batch, nodes, edges} = result;
  const nodeCount = nodes.length;
  const edgeCount = edges.length;
  const parts = nodeCount <= 60 && edgeCount <= 120 ? 1 : Math.ceil(Math.max(nodeCount / 60, edgeCount / 120));
  const sortedFiles = [...batch.files].sort((a, b) => a.path.localeCompare(b.path));
  const chunkSize = Math.ceil(sortedFiles.length / parts);
  const groups = [];
  for (let i = 0; i < sortedFiles.length; i += chunkSize) groups.push(sortedFiles.slice(i, i + chunkSize));
  const nodeById = new Map(nodes.map(node => [node.id, node]));
  const written = [];
  for (let i = 0; i < groups.length; i += 1) {
    const paths = new Set(groups[i].map(file => file.path));
    const partNodes = nodes.filter(node => paths.has(node.filePath));
    const partNodeIds = new Set(partNodes.map(node => node.id));
    const partEdges = edges.filter(edge => partNodeIds.has(edge.source));
    const fragment = {nodes: partNodes, edges: partEdges};
    const name = groups.length === 1 ? `batch-${batch.batchIndex}.json` : `batch-${batch.batchIndex}-part-${i + 1}.json`;
    const out = path.join(intermediate, name);
    fs.writeFileSync(out, JSON.stringify(fragment, null, 2) + '\n');
    written.push({name, nodeCount: partNodes.length, edgeCount: partEdges.length});
  }
  return {partsPlanned: parts, files: written, nodeCount, edgeCount};
}

const report = {};
for (const batchIndex of requested) report[batchIndex] = splitAndWrite(analyzeBatch(batchIndex));
process.stdout.write(JSON.stringify(report, null, 2) + '\n');
