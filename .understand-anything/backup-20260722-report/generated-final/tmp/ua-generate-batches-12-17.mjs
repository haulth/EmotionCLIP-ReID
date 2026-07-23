import fs from "node:fs";
import path from "node:path";

const root = "E:/Source/EmotionCLIP-ReID";
const intermediate = path.join(root, ".understand-anything", "intermediate");
const temp = path.join(root, ".understand-anything", "tmp");
const batchesRoot = JSON.parse(fs.readFileSync(path.join(intermediate, "batches.json"), "utf8"));

const S = {
  "outputs/fer2013/README_training_logs.md": [
    "Tài liệu truy vết quá trình trích xuất log FER2013 từ notebook, liệt kê quy mô log, số dòng theo step/epoch/validation và các metric tốt nhất của run ngắn.",
    ["documentation", "training-log", "fer2013", "provenance"],
  ],
  "outputs/fer2013/accuracy_uce_summary.json": [
    "Tóm tắt các epoch tốt nhất của FER2013 theo accuracy, balanced accuracy, Macro-F1 và ECE; trường UCE được ghi rõ chỉ là alias của ECE vì run không log UCE riêng.",
    ["metric-summary", "calibration", "fer2013", "experiment-output"],
  ],
  "outputs/fer2013/best_metrics_summary.json": [
    "Lưu snapshot đánh giá FER2013 tại checkpoint tốt nhất theo Macro-F1: accuracy 0,7059, balanced accuracy 0,6832, Macro-F1 0,6864, ECE 0,3746 cùng confusion matrix và per-class F1 trên 7.178 mẫu.",
    ["metric-artifact", "fer2013", "classification", "calibration"],
  ],
  "outputs/fer2013/inference_sample_uncertainty.json": [
    "Lưu một kết quả suy luận FER2013 cấp mẫu với nhãn neutral, phân bố xác suất bảy lớp, uncertainty 0,5080 và similarity theo từng emotion descriptor.",
    ["inference-output", "uncertainty", "fer2013", "semantic-similarity"],
  ],
  "outputs/fer2013/training_epoch_losses.csv": [
    "Bảng loss và thời gian huấn luyện FER2013 theo stage và epoch, gồm 50 bản ghi cho hai giai đoạn.",
    ["training-history", "loss", "fer2013", "experiment-output"],
  ],
  "outputs/fer2013/training_step_losses.csv": [
    "Bảng chi tiết loss FER2013 theo stage, epoch và step, gồm 1.760 bản ghi dùng để phân tích đường học trong từng epoch.",
    ["training-history", "step-metrics", "loss", "fer2013"],
  ],
  "outputs/fer2013/uncertainty_summary.json": [
    "Tổng hợp evidence uncertainty có sẵn của FER2013 tại checkpoint epoch 24, bao gồm risk AUC 0,2255, ECE 0,3746 và uncertainty của một mẫu; file cũng nêu rõ không có avg uncertainty theo epoch.",
    ["uncertainty", "metric-summary", "fer2013", "limitation-note"],
  ],
  "outputs/fer2013/validation_accuracy.csv": [
    "Bảng accuracy, balanced accuracy và Macro-F1 của FER2013 theo 30 epoch validation, tách riêng khỏi metric calibration.",
    ["validation-history", "fer2013", "classification", "experiment-output"],
  ],
  "outputs/fer2013/validation_metrics.csv": [
    "Bảng validation FER2013 theo 30 epoch với accuracy, balanced accuracy, Macro-F1 và ECE, là nguồn của các bản tóm tắt best/lowest metric.",
    ["validation-history", "calibration", "fer2013", "experiment-output"],
  ],
  "outputs/fer2013/validation_uncertainty.csv": [
    "Bảng một dòng ghi uncertainty-risk AUC và các metric của checkpoint FER2013 epoch 24; ghi chú xác nhận notebook không có uncertainty trung bình theo epoch.",
    ["validation-output", "uncertainty", "fer2013", "provenance"],
  ],
  "outputs/report_w4/emotionclip_outputs_summary.csv": [
    "Bảng đối chiếu gọn kết quả FER2013 và RAF-DB cho báo cáo W4, gồm best epoch, accuracy, balanced accuracy, Macro-F1, ECE, uncertainty-risk AUC và số mẫu.",
    ["report-data", "benchmark-summary", "fer", "experiment-output"],
  ],
  "outputs/report_w4/emotionclip_outputs_summary.json": [
    "Tổng hợp có cấu trúc kết quả FER2013 và RAF-DB cho báo cáo W4, bao gồm best epoch theo nhiều tiêu chí, per-class F1 và metric calibration/uncertainty.",
    ["report-data", "metric-summary", "fer", "calibration"],
  ],
  "outputs/report_w4/rafdb_training_history_extracted.csv": [
    "Bảng lịch sử loss và thời gian huấn luyện RAF-DB theo stage/epoch được trích cho báo cáo W4.",
    ["training-history", "raf-db", "loss", "report-data"],
  ],
  "outputs/report_w4/rafdb_validation_history_extracted.csv": [
    "Bảng lịch sử validation RAF-DB theo epoch với accuracy, balanced accuracy, Macro-F1 và ECE được trích cho báo cáo W4.",
    ["validation-history", "raf-db", "calibration", "report-data"],
  ],
  ".codex/tmp/019f8a29-798f-7c12-a8ee-fdf1d43a6942/edit_paper.mjs": [
    "Script thao tác workbook nghiên cứu: inspect/render các sheet, chèn paper số 89 về ULMFiT sau kiểm tra trùng lặp, cập nhật Research Log, export và xác minh workbook đầu ra.",
    ["spreadsheet-automation", "literature-review", "validation", "artifact-generation"],
    "JavaScript dùng top-level await và @oai/artifact-tool để đọc, render, chỉnh sửa rồi tái nhập XLSX nhằm kiểm tra hậu điều kiện.",
  ],
  "config/defaults_base.py": [
    "Định nghĩa cấu hình YACS nền cho pipeline ReID một-solver, gồm backbone, input augmentation, dataset, dataloader, optimizer, scheduler và chế độ test.",
    ["configuration", "reid", "yacs", "training"],
    "Cấu hình được xây bằng cây CfgNode và gán giá trị mặc định theo namespace MODEL/INPUT/DATASETS/DATALOADER/SOLVER/TEST.",
  ],
  "config/defaults.py": [
    "Định nghĩa cấu hình YACS chính của CLIP-ReID với solver tách STAGE1/STAGE2, các tham số model/input/dataset và thiết lập đánh giá.",
    ["configuration", "two-stage-training", "clip-reid", "yacs"],
    "Hai CfgNode STAGE1 và STAGE2 mã hóa lịch tối ưu độc lập trong cùng cây cấu hình.",
  ],
  "datasets/__init__.py": [
    "File khởi tạo package datasets hiện để trống và không re-export thành phần nào.",
    ["entry-point", "package", "datasets"],
  ],
  "datasets/keypoint_test.txt": [
    "Danh sách keypoint cho ảnh test VeRi: mỗi dòng chứa đường dẫn ảnh, các cặp tọa độ với -1 biểu thị điểm thiếu và một nhãn số ở cuối.",
    ["keypoint-data", "veri", "test-split", "landmark"],
  ],
  "datasets/keypoint_train.txt": [
    "Danh sách keypoint cho ảnh train VeRi: mỗi dòng ánh xạ đường dẫn ảnh sang các cặp tọa độ, giá trị -1 cho điểm thiếu và một nhãn số cuối dòng.",
    ["keypoint-data", "veri", "train-split", "landmark"],
  ],
  "datasets/preprocessing.py": [
    "Cung cấp augmentation Random Erasing áp trực tiếp lên tensor ảnh, lấy mẫu diện tích/tỷ lệ ngẫu nhiên và tô vùng chữ nhật theo mean từng kênh.",
    ["data-augmentation", "random-erasing", "preprocessing", "vision"],
    "Implementation lặp tối đa 100 lần để tìm hình chữ nhật hợp lệ và hỗ trợ tensor một hoặc ba kênh.",
  ],
  "docs/emotionclip_reid_proposed_emotionclip_reid_model.drawio": [
    "Sơ đồ mô hình đề xuất nối face-safe preprocessing, CLIP ViT-B/16, expression-aware adapters và nhánh emotion/AU semantic descriptor cho huấn luyện và suy luận FER.",
    ["architecture-diagram", "emotionclip-reid", "semantic-descriptor", "adapter"],
    "draw.io lưu đồ thị bằng mxGraph XML với nhãn HTML trong thuộc tính value.",
  ],
  "docs/original_scientific_model_diagram.drawio": [
    "Sơ đồ học thuật của CLIP-ReID gốc, mô tả Stage I tối ưu prompt định danh bằng contrastive image-text và Stage II fine-tune ReID với text anchors.",
    ["architecture-diagram", "clip-reid", "two-stage-training", "baseline"],
    "draw.io dùng mxGraph XML để biểu diễn node, edge và công thức trong nhãn HTML.",
  ],
  "docs/original_system_architecture.drawio": [
    "Sơ đồ kiến trúc hệ thống CLIP-ReID nguyên bản theo entrypoint, cấu hình/dữ liệu ReID, dual encoder/loss và training/inference; đồng thời đánh dấu chưa có AU-specific pipeline.",
    ["system-diagram", "clip-reid", "legacy-reid", "baseline"],
    "draw.io dùng mxGraph XML và các mxCell có nhãn mô tả từng tầng hệ thống.",
  ],
  "docs/report/fig/emotionclip_reid_context_gap_hub_map.drawio": [
    "Bản đồ research gap quy tụ VLM adaptation, language-guided ReID, AU/FACS, CLIP-FER, uncertainty và multimodal MER quanh bài toán emotion/AU semantic anchors.",
    ["research-gap", "related-work", "semantic-anchor", "diagram"],
    "Sơ đồ draw.io mã hóa một hub-and-spoke research map bằng mxGraph XML.",
  ],
  "docs/report/fig/emotionclip_reid_current_system.drawio": [
    "Sơ đồ hệ thống EmotionCLIP-ReID hiện tại, phân biệt khối tái sử dụng/frozen, đã triển khai, trainable, loss/metric và optional từ CLIP–CoOp–CLIP-ReID sang FER.",
    ["current-architecture", "emotionclip-reid", "two-stage-training", "diagram"],
    "Sơ đồ draw.io dùng màu/nhãn để phân loại trạng thái các khối kiến trúc.",
  ],
  "docs/report/fig/emotionclip_reid_proposal_pipeline.drawio": [
    "Sơ đồ giả thuyết đề xuất: Stage 1 học candidate emotion descriptors, Stage 2 dùng descriptor cố định để kiểm tra semantic anchoring qua adapter, alignment và calibration.",
    ["proposal", "semantic-descriptor", "ablation", "diagram"],
    "draw.io biểu diễn pipeline và các claim cần metric/ablation bằng mxGraph XML.",
  ],
  "docs/report/fig/emotionclip_reid_publication_pipeline_update.drawio": [
    "Sơ đồ publication pipeline v1 thực dụng, chỉ rõ Stage 1 học class-specific latent prompt tokens và Stage 2 dùng fixed emotion anchors, không diễn giải token như eye/mouth/AU descriptor.",
    ["publication-figure", "prompt-learning", "two-stage-training", "fer"],
    "Nhãn mxGraph làm rõ ranh giới giữa latent prompt token và anatomy descriptor.",
  ],
  "docs/report/fig/emotionclip_reid_publication_pipeline.drawio": [
    "Sơ đồ pipeline học thuật tối giản cho Stage 1 semantic prompt optimization và dòng truyền fixed text descriptor sang Stage 2.",
    ["publication-figure", "semantic-prompt", "two-stage-training", "objective"],
    "draw.io lưu node/edge và ký hiệu objective trong mxGraph XML.",
  ],
  "docs/report/fig/emotionclip_reid_two_stage_structure.drawio": [
    "Sơ đồ kiến trúc anatomy-enabled hai giai đoạn: class-conditioned semantic prototypes, hybrid regional routing/geometry residual và reliability diagnostics tách biệt dưới protocol sealed test.",
    ["architecture-diagram", "anatomy-routing", "geometry-fusion", "reliability"],
    "Sơ đồ mxGraph dùng nhiều lớp chú giải cho frozen/trainable, feature/prototype, anatomy/quality và objective/reliability.",
  ],
  "docs/report/fig/v1/emotionclip_reid_publication_pipeline_update.drawio": [
    "Bản v1 của publication pipeline cập nhật, bổ sung anatomy artifact transform-aware, class geometry statistics và gated geometry residual ở Stage 1B trước Stage 2 adaptation.",
    ["publication-figure", "anatomy", "geometry-residual", "two-stage-training"],
    "draw.io dùng mxGraph XML để biểu diễn shared contract và hai stage.",
  ],
  "docs/report/fig/v1/emotionclip_reid_publication_pipeline.drawio": [
    "Bản v1 của sơ đồ semantic prompt optimization: image embedding, class prompt, label supervision và objective Stage 1 trước khi chuyển anchor sang giai đoạn sau.",
    ["publication-figure", "semantic-prompt", "stage-one", "objective"],
    "draw.io dùng mxGraph XML cho luồng input, frozen/trainable và objective.",
  ],
  "docs/report/fig/v1/emotionclip_reid_two_stage_structure.drawio": [
    "Sơ đồ đề xuất ban đầu chuyển CLIP-ReID từ ID anchoring sang emotion descriptor learning, adapter fine-tuning và AU prior tùy chọn.",
    ["proposal-diagram", "emotion-descriptor", "adapter", "au-prior"],
    "Sơ đồ mxGraph mô tả hai stage và các thành phần optional AU/pseudo-AU.",
  ],
  "docs/research_review/build_emotionclip_anatomy_gap_map.py": [
    "Script sinh sơ đồ draw.io cho kế hoạch anatomy/geometry cuối: shared geometry contract, dual-stream Stage 2, geometry-conditioned Stage 1 và output ambiguity/reliability tách biệt.",
    ["diagram-generator", "research-gap", "anatomy", "geometry"],
    "Python xây mxGraph XML bằng ElementTree rồi pretty-print qua minidom.",
  ],
  "docs/research_review/build_research_review.py": [
    "Pipeline tạo bộ related-work từ danh sách 32 paper nhúng trong source: xuất workbook nhiều sheet, báo cáo DOCX, sơ đồ gap map và kiểm tra render DOCX sang PDF/PNG.",
    ["literature-review", "artifact-generation", "spreadsheet", "document-automation"],
    "Python kết hợp openpyxl, python-docx, OOXML và mxGraph XML; bước QA gọi LibreOffice cùng pdftoppm.",
  ],
  "docs/research_review/build_summary.json": [
    "Manifest kết quả của pipeline related-work, ghi đường dẫn XLSX/DOCX/draw.io, paper_count 32 và xác nhận DOCX đã render thành công.",
    ["build-manifest", "literature-review", "render-verification", "artifact-output"],
  ],
  "docs/research_review/emotionclip_anatomy_research_gap_map.drawio": [
    "Sơ đồ research gap anatomy phiên bản chốt: reliability-gated geometry dùng chung cho Stage 1/2, visual fallback khi landmark kém và roadmap kiểm chứng theo ablation.",
    ["research-gap", "anatomy-routing", "reliability-gate", "diagram"],
    "mxGraph XML chứa trực tiếp công thức, tiêu chí go/no-go và đối chiếu nhóm paper R1–R11.",
  ],
  "docs/research_review/emotionclip_reid_gap_analysis_2026-07-14.md": [
    "Báo cáo audit toàn diện code, protocol, log và novelty: nhận diện leakage/provenance/uncertainty gaps, ghi nhận protocol sealed-test đã sửa, tổng hợp kết quả FER2013/RAF-DB và đề xuất roadmap publication-ready.",
    ["documentation", "code-audit", "research-gap", "experimental-protocol", "novelty"],
    "Markdown kết hợp bảng evidence, Mermaid architecture, mức Verified/Inferred/Proposed và checklist go/no-go.",
  ],
  "docs/research_review/emotionclip_reid_related_work_gap_map.drawio": [
    "Sơ đồ lập luận related-work từ VLM/prompt, language-guided ReID, CLIP-FER, AU/FACS, uncertainty và multimodal đến gap emotion/AU semantic anchors.",
    ["related-work", "research-gap", "semantic-anchor", "diagram"],
    "Sơ đồ được sinh bằng mxGraph XML từ build_research_review.py.",
  ],
  "docs/SOTA/disfa_sota_benchmark.md": [
    "Tổng hợp protocol và benchmark DISFA cho AU detection/intensity, xếp hạng các phương pháp gần đây, nêu caveat so sánh và gợi ý năm dataset AU tương tự.",
    ["documentation", "sota-benchmark", "disfa", "action-unit", "evaluation-protocol"],
    "Bảng Markdown tách rõ AU detection bằng Avg F1 với AU intensity bằng ICC/MSE/MAE.",
  ],
  "fig/emotionclip_reid_current_system.drawio": [
    "Bản sao sơ đồ hệ thống EmotionCLIP-ReID hiện tại, mô tả phần kế thừa CLIP/CoOp/CLIP-ReID và các khối FER đã triển khai hoặc optional.",
    ["current-architecture", "emotionclip-reid", "baseline", "diagram"],
    "draw.io dùng mxGraph XML và chú giải trạng thái khối.",
  ],
  "fig/emotionclip_reid_publication_pipeline.drawio": [
    "Sơ đồ publication pipeline chính cho hai-stage EmotionCLIP-ReID, từ image/label supervision ở Stage 1 đến visual adaptation và prediction ở Stage 2.",
    ["publication-figure", "two-stage-training", "semantic-anchor", "fer"],
    "draw.io mã hóa luồng dữ liệu và objective trong mxGraph XML.",
  ],
  "loss/__init__.py": [
    "Điểm vào package loss, wildcard-import toàn bộ lớp margin-based từ loss/arcface.py.",
    ["entry-point", "package", "loss", "barrel"],
  ],
  "loss/arcface.py": [
    "Cài đặt hai classification head margin-based ArcFace và CircleLoss trên cosine similarity chuẩn hóa cho embedding learning.",
    ["metric-learning", "arcface", "circle-loss", "classification-head"],
    "PyTorch nn.Module chuẩn hóa feature và class weight trước khi áp angular/circle margin; one-hot tensor hiện được tạo trực tiếp trên CUDA.",
  ],
  "loss/metric_learning.py": [
    "Tập hợp objective/head metric learning gồm pairwise ContrastiveLoss, CircleLoss, Arcface, Cosface và AMSoftmax cho bài toán phân biệt embedding.",
    ["metric-learning", "contrastive-loss", "margin-softmax", "classification-head"],
    "Các head PyTorch dùng normalized cosine logits và additive/angular margins; nhiều implementation tạo tensor nhãn phụ trên CUDA.",
  ],
  "notebooks/emotionclip_reid_jupyterhub_build_landmarks.ipynb": [
    "Runbook độc lập để tạo và cache anatomy/landmark artifact cho FER2013 và RAF-DB, audit coverage/jitter, vẽ failure dashboard và bàn giao manifest_anatomy.jsonl cho notebook train.",
    ["notebook", "landmark-pipeline", "anatomy-artifact", "data-audit"],
    "Notebook kết hợp Markdown và code; cache key phụ thuộc manifest, model, schema và jitter, còn mỗi build có LANDMARK_RUN_ID riêng.",
  ],
  "notebooks/emotionclip_reid_jupyterhub_fer2013.ipynb": [
    "Runbook JupyterHub cho FER2013 bảy lớp: chuẩn bị dataset/manifest, dùng landmark offline, khóa run ID, train hai stage và giữ PrivateTest làm sealed test.",
    ["notebook", "fer2013", "two-stage-training", "sealed-test", "reproducibility"],
    "Notebook mô tả DataParallel hai GPU T4, output immutable theo RUN_ID và separation Training/PublicTest/PrivateTest.",
  ],
  "notebooks/emotionclip_reid_jupyterhub_rafdb.ipynb": [
    "Runbook JupyterHub cho RAF-DB Basic: kiểm tra manifest/anatomy, tạo validation xác định từ official train, giữ official test sealed và train EmotionCLIP-ReID theo cấu hình hai stage.",
    ["notebook", "raf-db", "two-stage-training", "sealed-test", "reproducibility"],
    "Notebook tách staging visual khỏi output immutable và yêu cầu manifest_anatomy/anatomy_v3 được build trước.",
  ],
  "outputs/report_w4/emotionclip_reid_w4_functional_change_map.drawio": [
    "Sơ đồ thay đổi chức năng W4, đánh dấu các khối dataset/manifest, emotion descriptors, expression adapters, global-local alignment, evidential uncertainty và evidence artifacts.",
    ["change-map", "week-four", "emotionclip-reid", "diagram"],
    "mxGraph XML phân biệt khối mới W4 với khối sửa từ W3.",
  ],
  "outputs/report_w4/emotionclip_reid_w4_model_architecture.drawio": [
    "Sơ đồ kiến trúc W4 phân nhóm visual feature, semantic descriptor và decision/uncertainty/evaluation, đồng thời đánh dấu NEW, MODIFIED và REUSED/FROZEN.",
    ["architecture-diagram", "week-four", "adapter", "uncertainty"],
    "draw.io lưu chú giải đóng góp W4 trong mxGraph XML.",
  ],
  "outputs/report_w4/emotionclip_reid_w4_two_stage_training.drawio": [
    "Sơ đồ huấn luyện W4: Stage 1 học fixed emotion descriptors với CLIP frozen; Stage 2 kết hợp classifier, global/local alignment và uncertainty term.",
    ["training-diagram", "week-four", "two-stage-training", "uncertainty"],
    "mxGraph XML chứa công thức fusion logits và objective tổng của Stage 2.",
  ],
};

const binaryOffice = new Set([".docx", ".pptx", ".xlsx"]);

function basename(p) {
  return p.split("/").at(-1);
}

function stem(p) {
  const b = basename(p);
  const i = b.lastIndexOf(".");
  return i < 0 ? b : b.slice(0, i);
}

function complexity(nonEmpty = 0) {
  if (nonEmpty > 200) return "complex";
  if (nonEmpty >= 50) return "moderate";
  return "simple";
}

function officeMeta(p) {
  const ext = path.extname(p).toLowerCase();
  if (ext === ".pptx") {
    return [
      `Artifact slide PowerPoint nhị phân ${basename(p)}; structural extraction chỉ xác nhận tệp và quy mô, không cung cấp nội dung slide đủ tin cậy để phân tích sâu.`,
      ["office-artifact", "presentation", "report", "binary-document"],
      "PPTX là gói Office Open XML; node được mô tả bảo thủ vì extractor không xuất cấu trúc slide.",
    ];
  }
  if (ext === ".docx") {
    return [
      `Artifact báo cáo Word nhị phân ${basename(p)}; structural extraction chỉ xác nhận tệp và quy mô, nên không suy diễn nội dung chi tiết.`,
      ["office-artifact", "report", "documentation", "binary-document"],
      "DOCX là gói Office Open XML; node được mô tả bảo thủ vì extractor không xuất cấu trúc đoạn/bảng.",
    ];
  }
  return [
    `Workbook XLSX nhị phân ${basename(p)} dùng làm artifact tổng hợp tài liệu; structural extraction không cung cấp dữ liệu cell đủ tin cậy để mô tả chi tiết.`,
    ["office-artifact", "spreadsheet", "literature-review", "binary-document"],
    "XLSX là gói Office Open XML; node được mô tả bảo thủ vì extractor không xuất sheet/cell.",
  ];
}

function metadata(p) {
  if (S[p]) return S[p];
  if (/^outputs\/RAF-DB\/metrics_epoch_\d+\.json$/.test(p)) {
    const epoch = p.match(/epoch_(\d+)/)[1];
    return [
      `Artifact đánh giá RAF-DB tại epoch ${epoch}, lưu accuracy, balanced accuracy, Macro-F1, per-class F1, confusion matrix, ECE, uncertainty-risk AUC và dự đoán/uncertainty theo 3.068 mẫu.`,
      ["metric-artifact", "raf-db", "validation-output", "uncertainty"],
      "JSON giữ cả metric tổng hợp lẫn mảng cấp mẫu, vì vậy có quy mô lớn hơn một summary thông thường.",
    ];
  }
  const ext = path.extname(p).toLowerCase();
  if (binaryOffice.has(ext)) return officeMeta(p);
  if (ext === ".drawio") {
    return [
      `Sơ đồ draw.io ${basename(p)}; structural extraction xác nhận artifact mxGraph nhưng không cung cấp thêm semantic structure ngoài nội dung tệp.`,
      ["architecture-diagram", "drawio", "documentation"],
      "draw.io thường lưu graph bằng mxGraph XML.",
    ];
  }
  return [
    `Artifact ${basename(p)} thuộc ${p.split("/").slice(0, -1).join("/")} được structural extractor ghi nhận trong codebase.`,
    ["project-artifact", "repository-file", "supporting-data"],
  ];
}

function fileNode(f, result) {
  const [summary, tags, languageNotes] = metadata(f.path);
  let type;
  let id;
  if (f.fileCategory === "config") {
    type = "config";
    id = `config:${f.path}`;
  } else if (f.fileCategory === "docs") {
    type = "document";
    id = `document:${f.path}`;
  } else if (f.fileCategory === "data") {
    type = "table";
    id = `table:${f.path}:${stem(f.path)}`;
  } else {
    type = "file";
    id = `file:${f.path}`;
  }
  const node = {
    id,
    type,
    name: basename(f.path),
    filePath: f.path,
    summary,
    tags,
    complexity: complexity(result?.nonEmptyLines ?? f.sizeLines),
  };
  if (languageNotes) node.languageNotes = languageNotes;
  return node;
}

const structuralSummary = {
  "class:datasets/preprocessing.py:RandomErasing": ["Augmentation callable xóa ngẫu nhiên một vùng chữ nhật hợp lệ của tensor ảnh theo xác suất, tỷ lệ diện tích và aspect ratio cấu hình.", ["data-augmentation", "random-erasing", "callable-class"]],
  "function:docs/research_review/build_emotionclip_anatomy_gap_map.py:box": ["Tạo chuỗi style mxGraph cho hộp bo góc với màu, cỡ chữ, căn lề và dashed tùy chọn.", ["diagram-style", "utility", "mxgraph"]],
  "function:docs/research_review/build_emotionclip_anatomy_gap_map.py:text_style": ["Tạo style mxGraph cho nhãn văn bản không viền/nền với font, màu, bold và căn lề tùy chọn.", ["diagram-style", "text-formatting", "utility"]],
  "function:docs/research_review/build_emotionclip_anatomy_gap_map.py:edge_style": ["Tạo style edge trực giao có mũi tên, màu và dashed tùy chọn cho sơ đồ gap map.", ["diagram-style", "edge", "mxgraph"]],
  "function:docs/research_review/build_emotionclip_anatomy_gap_map.py:vertex": ["Chèn một mxCell vertex cùng mxGeometry vào graph root tại vị trí và kích thước chỉ định.", ["diagram-builder", "vertex", "xml"]],
  "function:docs/research_review/build_emotionclip_anatomy_gap_map.py:edge": ["Chèn một mxCell edge nối source-target, tái sử dụng edge_style và thêm geometry tương đối.", ["diagram-builder", "edge", "xml"]],
  "function:docs/research_review/build_research_review.py:set_cell_shading": ["Gắn phần tử OOXML w:shd vào Word table cell để đặt màu nền.", ["docx-formatting", "ooxml", "utility"]],
  "function:docs/research_review/build_research_review.py:set_cell_text": ["Ghi text và định dạng bold/màu/cỡ chữ vào Word table cell, đồng thời căn giữa theo chiều dọc.", ["docx-formatting", "table-cell", "utility"]],
  "function:docs/research_review/build_research_review.py:build_excel": ["Tạo workbook review 32 paper với các sheet Papers, Synthesis Matrix, Thesis Focus, Self-Critique và Research Gap, áp style/table rồi lưu XLSX.", ["spreadsheet-builder", "literature-review", "artifact-generation"]],
  "function:docs/research_review/build_research_review.py:add_hyperlink": ["Tạo hyperlink ngoài trong paragraph DOCX bằng relationship và các phần tử OOXML có màu xanh, gạch chân.", ["docx-formatting", "hyperlink", "ooxml"]],
  "function:docs/research_review/build_research_review.py:add_heading": ["Thêm heading vào DOCX và chuẩn hóa font Arial cùng màu chủ đề xanh.", ["docx-formatting", "heading", "utility"]],
  "function:docs/research_review/build_research_review.py:add_body": ["Thêm đoạn body vào DOCX với spacing, line height, font Arial và cỡ chữ thống nhất.", ["docx-formatting", "paragraph", "utility"]],
  "function:docs/research_review/build_research_review.py:build_docx": ["Dựng báo cáo related-work tiếng Việt từ luận điểm, sáu nhóm paper, self-critique, synthesis và danh mục nguồn rồi lưu DOCX.", ["document-builder", "literature-review", "artifact-generation"]],
  "function:docs/research_review/build_research_review.py:mx_cell": ["Trả về XML fragment cho một mxCell bo góc sau khi escape nội dung và chuyển newline thành HTML break.", ["diagram-builder", "mxgraph", "xml"]],
  "function:docs/research_review/build_research_review.py:mx_edge": ["Trả về XML fragment cho edge trực giao giữa hai mxCell, kèm label đã escape.", ["diagram-builder", "edge", "xml"]],
  "function:docs/research_review/build_research_review.py:build_drawio": ["Lắp các node/edge related-work thành mxGraph XML và lưu sơ đồ research-gap draw.io.", ["diagram-builder", "research-gap", "artifact-generation"]],
  "function:docs/research_review/build_research_review.py:verify_docx_render": ["Dùng LibreOffice chuyển DOCX sang PDF rồi pdftoppm render PNG, trả trạng thái thành công và thông báo QA.", ["render-verification", "docx", "quality-assurance"]],
  "function:docs/research_review/build_research_review.py:main": ["Điều phối tạo XLSX, DOCX, draw.io, chạy render QA và ghi build_summary.json với đường dẫn cùng paper count.", ["entry-point", "orchestration", "artifact-generation"]],
  "class:loss/arcface.py:ArcFace": ["Head ArcFace chuẩn hóa input/class weights, áp additive angular margin cho lớp đúng và scale logits trước CE.", ["arcface", "metric-learning", "classification-head"]],
  "class:loss/arcface.py:CircleLoss": ["Head Circle Loss sinh class logits từ cosine similarity với hệ số trọng số positive/negative phụ thuộc margin.", ["circle-loss", "metric-learning", "classification-head"]],
  "class:loss/metric_learning.py:ContrastiveLoss": ["Loss khai thác ma trận similarity trong batch, cộng phạt positive dưới 1 và negative vượt margin theo từng mẫu.", ["contrastive-loss", "pair-mining", "metric-learning"]],
  "class:loss/metric_learning.py:CircleLoss": ["Biến cosine similarity thành Circle Loss logits bằng adaptive positive/negative factors và one-hot target.", ["circle-loss", "metric-learning", "classification-head"]],
  "class:loss/metric_learning.py:Arcface": ["Implementation ArcFace hỗ trợ easy margin và label smoothing, tạo angular-margin logits trên normalized features/weights.", ["arcface", "label-smoothing", "metric-learning"]],
  "class:loss/metric_learning.py:Cosface": ["Implementation CosFace trừ cosine margin ở lớp mục tiêu và scale normalized logits, kèm biểu diễn cấu hình.", ["cosface", "metric-learning", "classification-head"]],
  "class:loss/metric_learning.py:AMSoftmax": ["Implementation additive-margin softmax chuẩn hóa feature/weight, trừ margin theo nhãn và trả scaled class logits.", ["am-softmax", "metric-learning", "classification-head"]],
};

function structuralNode(kind, fpath, item) {
  const id = `${kind}:${fpath}:${item.name}`;
  const [summary, tags] = structuralSummary[id] ?? [`Thành phần ${item.name} được structural extractor phát hiện trong ${fpath}.`, ["code-component", "implementation", "project-structure"]];
  return {
    id,
    type: kind,
    name: item.name,
    filePath: fpath,
    lineRange: [item.startLine, item.endLine],
    summary,
    tags,
    complexity: complexity(item.endLine - item.startLine + 1),
  };
}

function edge(source, target, type) {
  const weights = { contains: 1.0, imports: 0.7, exports: 0.8, related: 0.5, documents: 0.5 };
  return { source, target, type, direction: "forward", weight: weights[type] };
}

const extras = {
  14: [
    edge("document:outputs/fer2013/README_training_logs.md", "table:outputs/fer2013/training_epoch_losses.csv:training_epoch_losses", "documents"),
    edge("document:outputs/fer2013/README_training_logs.md", "table:outputs/fer2013/training_step_losses.csv:training_step_losses", "documents"),
    edge("document:outputs/fer2013/README_training_logs.md", "table:outputs/fer2013/validation_metrics.csv:validation_metrics", "documents"),
    edge("document:outputs/fer2013/README_training_logs.md", "config:outputs/fer2013/best_metrics_summary.json", "documents"),
    edge("document:outputs/fer2013/README_training_logs.md", "config:outputs/fer2013/uncertainty_summary.json", "documents"),
    edge("config:outputs/fer2013/accuracy_uce_summary.json", "table:outputs/fer2013/validation_metrics.csv:validation_metrics", "related"),
    edge("config:outputs/fer2013/best_metrics_summary.json", "table:outputs/fer2013/validation_metrics.csv:validation_metrics", "related"),
    edge("config:outputs/fer2013/inference_sample_uncertainty.json", "config:outputs/fer2013/uncertainty_summary.json", "related"),
    edge("config:outputs/fer2013/uncertainty_summary.json", "config:outputs/fer2013/best_metrics_summary.json", "related"),
    edge("table:outputs/fer2013/validation_accuracy.csv:validation_accuracy", "table:outputs/fer2013/validation_metrics.csv:validation_metrics", "related"),
    edge("table:outputs/fer2013/validation_uncertainty.csv:validation_uncertainty", "config:outputs/fer2013/uncertainty_summary.json", "related"),
  ],
  15: [
    edge("config:outputs/report_w4/emotionclip_outputs_summary.json", "table:outputs/report_w4/emotionclip_outputs_summary.csv:emotionclip_outputs_summary", "related"),
    edge("config:outputs/report_w4/emotionclip_outputs_summary.json", "table:outputs/report_w4/rafdb_training_history_extracted.csv:rafdb_training_history_extracted", "related"),
    edge("config:outputs/report_w4/emotionclip_outputs_summary.json", "table:outputs/report_w4/rafdb_validation_history_extracted.csv:rafdb_validation_history_extracted", "related"),
  ],
  16: [
    edge("document:datasets/keypoint_train.txt", "document:datasets/keypoint_test.txt", "related"),
  ],
  17: [
    edge("config:docs/research_review/build_summary.json", "file:docs/research_review/build_research_review.py", "related"),
    edge("document:docs/research_review/emotionclip_reid_gap_analysis_2026-07-14.md", "file:notebooks/emotionclip_reid_jupyterhub_fer2013.ipynb", "documents"),
    edge("document:docs/research_review/emotionclip_reid_gap_analysis_2026-07-14.md", "file:notebooks/emotionclip_reid_jupyterhub_rafdb.ipynb", "documents"),
    edge("document:docs/SOTA/disfa_sota_benchmark.md", "document:docs/research_review/emotionclip_reid_gap_analysis_2026-07-14.md", "related"),
  ],
};

for (const index of [12, 13, 14, 15, 16, 17]) {
  const batch = batchesRoot.batches.find((b) => b.batchIndex === index);
  if (!batch) throw new Error(`Không tìm thấy batch ${index}`);
  const extract = JSON.parse(fs.readFileSync(path.join(temp, `ua-file-extract-results-${index}.json`), "utf8"));
  if (!extract.scriptCompleted) throw new Error(`Extractor chưa hoàn tất batch ${index}`);
  const resultByPath = new Map(extract.results.map((r) => [r.path, r]));
  const nodes = [];
  const edges = [];

  for (const f of batch.files) {
    const result = resultByPath.get(f.path);
    if (!result) throw new Error(`Batch ${index} thiếu extraction result cho ${f.path}`);
    const parent = fileNode(f, result);
    nodes.push(parent);

    if (f.fileCategory === "code") {
      const exported = new Set((result.exports ?? []).map((x) => x.name));
      for (const fn of result.functions ?? []) {
        const significant = exported.has(fn.name) || fn.endLine - fn.startLine + 1 >= 10;
        if (!significant) continue;
        const child = structuralNode("function", f.path, fn);
        nodes.push(child);
        edges.push(edge(parent.id, child.id, "contains"));
        if (exported.has(fn.name)) edges.push(edge(parent.id, child.id, "exports"));
      }
      for (const cls of result.classes ?? []) {
        const significant = exported.has(cls.name) || cls.endLine - cls.startLine + 1 >= 20 || (cls.methods ?? []).length >= 2;
        if (!significant) continue;
        const child = structuralNode("class", f.path, cls);
        nodes.push(child);
        edges.push(edge(parent.id, child.id, "contains"));
        if (exported.has(cls.name)) edges.push(edge(parent.id, child.id, "exports"));
      }

      for (const importedPath of batch.batchImportData[f.path] ?? []) {
        edges.push(edge(parent.id, `file:${importedPath}`, "imports"));
      }
    }
  }

  edges.push(...(extras[index] ?? []));

  const ids = new Set(nodes.map((n) => n.id));
  if (ids.size !== nodes.length) throw new Error(`Batch ${index} có node ID trùng`);
  if (batch.files.some((f) => !nodes.some((n) => n.filePath === f.path))) throw new Error(`Batch ${index} thiếu file node`);
  const expectedImports = batch.files.filter((f) => f.fileCategory === "code").reduce((n, f) => n + (batch.batchImportData[f.path] ?? []).length, 0);
  const actualImports = edges.filter((e) => e.type === "imports").length;
  if (expectedImports !== actualImports) throw new Error(`Batch ${index} import edge ${actualImports}/${expectedImports}`);
  for (const e of edges) {
    if (e.source === e.target) throw new Error(`Batch ${index} có self edge ${e.source}`);
    if (!ids.has(e.source) || !ids.has(e.target)) throw new Error(`Batch ${index} dangling edge ${e.source} -> ${e.target}`);
  }
  for (const n of nodes) {
    if (!n.summary || !Array.isArray(n.tags) || n.tags.length < 3) throw new Error(`Batch ${index} node thiếu semantic field: ${n.id}`);
    if (!n.tags.every((t) => t === t.toLowerCase() && !t.includes(" "))) throw new Error(`Batch ${index} tag không hợp lệ: ${n.id}`);
  }
  if (nodes.length > 60 || edges.length > 120) throw new Error(`Batch ${index} cần split nhưng generator hiện chờ single-part`);

  const out = path.join(intermediate, `batch-${index}.json`);
  fs.writeFileSync(out, JSON.stringify({ nodes, edges }, null, 2) + "\n", "utf8");
  console.log(JSON.stringify({ index, out, nodes: nodes.length, edges: edges.length, files: batch.files.length }));
}
