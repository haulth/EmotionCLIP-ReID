import fs from "node:fs";
import path from "node:path";

const root = process.cwd();
const intermediate = path.join(root, ".understand-anything", "intermediate");
const tmp = path.join(root, ".understand-anything", "tmp");
const batchesPayload = JSON.parse(fs.readFileSync(path.join(intermediate, "batches.json"), "utf8"));
const selected = [1, 3, 5, 7, 9, 10];

const FILE_SUMMARIES = {
  "config/__init__.py": "Khởi tạo giao diện cấu hình dùng chung cho các entry point ReID, xuất ba biến thể cấu hình YACS cho huấn luyện chuẩn, kiểm thử và baseline. Mô-đun suy giảm an toàn về `None` khi môi trường chỉ thiếu riêng gói `yacs`.",
  "loss/center_loss.py": "Cài đặt Center Loss để kéo embedding của từng mẫu về tâm lớp tương ứng, hỗ trợ cả CPU và GPU. Thành phần này được factory loss khởi tạo như một criterion phụ cho metric learning.",
  "loss/make_loss.py": "Factory ghép classification loss, Triplet Loss và tùy chọn image-to-text loss theo cấu hình sampler. Hàm trả về closure tính loss cùng Center Loss để processor dùng trong huấn luyện ReID.",
  "loss/softmax_loss.py": "Cung cấp hai biến thể cross-entropy có label smoothing cho logits nhận dạng. Các criterion PyTorch này giảm overconfidence khi tối ưu phân loại identity.",
  "loss/supcontrast.py": "Cài đặt supervised contrastive loss trên ma trận tương đồng giữa hai tập đặc trưng. Loss được dùng trong Stage 1 CLIP-ReID để căn chỉnh image embedding và text prompt embedding theo identity.",
  "loss/triplet_loss.py": "Tập hợp phép chuẩn hóa, khoảng cách Euclidean/cosine, hard-example mining và Triplet Loss cho metric learning. Đây là lõi tối ưu độ phân tách identity trong baseline và Stage 2.",
  "processor/processor.py": "Điều phối vòng lặp train/inference baseline với AMP, checkpoint định kỳ và đánh giá CMC/mAP. Processor hỗ trợ DataParallel, tùy chọn center-loss optimizer và nhãn camera/view cho SIE.",
  "processor/processor_clipreid_stage1.py": "Thực thi Stage 1 của CLIP-ReID: cache image features, sau đó tối ưu text prompt bằng supervised contrastive loss hai chiều image-to-text và text-to-image. Vòng lặp dùng AMP, cosine scheduler và checkpoint riêng cho Stage 1.",
  "processor/processor_clipreid_stage2.py": "Thực thi Stage 2 của CLIP-ReID bằng cách cache toàn bộ text features theo lớp rồi huấn luyện visual ReID với classification, triplet và image-to-text logits. Mô-đun đồng thời đánh giá CMC/mAP, lưu checkpoint và cung cấp inference.",
  "solver/cosine_lr.py": "Cài đặt cosine learning-rate scheduler có warmup, chu kỳ/restart, decay và giới hạn số chu kỳ. Lớp mở rộng scheduler nền và được dùng để tối ưu prompt ở Stage 1.",
  "solver/lr_scheduler.py": "Cài đặt WarmupMultiStepLR với warmup constant hoặc linear trước các mốc giảm learning rate. Scheduler này phục vụ huấn luyện baseline và Stage 2.",
  "solver/make_optimizer.py": "Factory tạo optimizer cho toàn bộ model và optimizer SGD riêng cho Center Loss. Nó áp dụng learning-rate/weight-decay riêng cho bias và tùy chọn learning rate lớn ở classifier.",
  "solver/make_optimizer_prompt.py": "Tạo optimizer tách biệt cho hai giai đoạn CLIP-ReID: Stage 1 chỉ cập nhật prompt learner, Stage 2 cập nhật các tham số còn lại cùng Center Loss. Learning rate và weight decay được lấy từ từng nhánh cấu hình STAGE1/STAGE2.",
  "solver/scheduler.py": "Lớp nền quản lý learning-rate state, cập nhật param groups theo epoch/update và chèn noise có kiểm soát. Nó cung cấp protocol chung để các scheduler cụ thể như cosine decay kế thừa.",
  "solver/scheduler_factory.py": "Factory cấu hình CosineLRScheduler cho Stage 1 từ số epoch, warmup, learning-rate tối thiểu và noise range. Hàm trả scheduler chạy theo epoch.",
  "test.py": "Entry point inference cho baseline ReID: nạp YAML/CLI, dựng dataloader và model, tải checkpoint rồi báo Rank-1/Rank-5 cùng mAP. Script dùng chung logger và processor đánh giá chuẩn.",
  "test_clipreid.py": "Entry point inference cho mô hình CLIP-ReID hai giai đoạn, từ cấu hình đến dựng model và tải trọng số kiểm thử. Kết quả được tính bằng processor Stage 2 với CMC/mAP.",
  "train.py": "Entry point huấn luyện baseline ReID, chịu trách nhiệm merge cấu hình, đặt seed, khởi tạo distributed runtime, dữ liệu, model, loss, optimizer và scheduler. Sau đó script chuyển toàn bộ thành phần vào processor huấn luyện chuẩn.",
  "train_clipreid.py": "Entry point huấn luyện CLIP-ReID hai giai đoạn: Stage 1 căn chỉnh prompt/text-image rồi Stage 2 tối ưu ReID với text prototypes đã học. Script điều phối cấu hình, seed, distributed runtime, dataloader, model, loss, optimizer và scheduler riêng cho từng stage.",
  "utils/logger.py": "Tạo logger thống nhất cho console và file, tránh gắn handler trùng lặp giữa các lần khởi tạo. Tiện ích ghi log train/test vào thư mục output của experiment.",
  "utils/meter.py": "Cung cấp bộ tích lũy AverageMeter cho giá trị hiện tại, tổng, số mẫu và trung bình chạy. Processor dùng nó để theo dõi loss và accuracy theo mini-batch.",
  "utils/metrics.py": "Tính ma trận khoảng cách và metric đánh giá ReID theo giao thức Market1501, gồm CMC và mAP sau khi loại cặp cùng identity/camera. Lớp evaluator tích lũy feature theo batch, tùy chọn chuẩn hóa và k-reciprocal re-ranking.",
  "utils/reranking.py": "Cài đặt k-reciprocal encoding re-ranking kết hợp Jaccard distance với khoảng cách gốc cho bài toán person ReID. Thuật toán theo công trình CVPR 2017 của Zhong và cộng sự nhằm cải thiện thứ hạng query-gallery.",
  "datasets/bases.py": "Định nghĩa abstraction chung cho image ReID dataset, thống kê identity/camera/view và adapter PyTorch Dataset để đọc ảnh, áp transform. Hàm đọc ảnh thử lại khi gặp lỗi I/O và cho phép ảnh bị cắt ngắn.",
  "datasets/dukemtmcreid.py": "Parser cho DukeMTMC-reID, kiểm tra cấu trúc thư mục và chuyển tên ảnh thành các tuple đường dẫn, person ID, camera ID và view ID. Lớp công bố train/query/gallery cùng thống kê dataset.",
  "datasets/make_dataloader.py": "Factory dataloader baseline cho các bộ person/vehicle ReID, kết hợp augmentation, identity sampler hoặc softmax sampling và collate riêng cho train/validation. Hàm trả loader cùng số query, lớp, camera và view để dựng model.",
  "datasets/make_dataloader_clipreid.py": "Factory dữ liệu CLIP-ReID tạo riêng Stage 1 loader không augmentation mạnh, Stage 2 identity-sampled loader và validation loader. Nó hỗ trợ các dataset person/vehicle cùng DistributedDataParallel sampler.",
  "datasets/market1501.py": "Parser Market1501 đọc train/query/gallery từ quy ước tên ảnh, loại junk identity và relabel tập train. Kết quả chuẩn hóa thành tuple dùng chung cho pipeline ReID.",
  "datasets/msmt17.py": "Parser MSMT17 đọc các danh sách train/val/query/gallery, relabel identity huấn luyện và chuẩn hóa camera/view metadata. Lớp cung cấp các split và thống kê cho dataloader factory.",
  "datasets/occ_duke.py": "Parser cho Occluded-DukeMTMC, tải/kiểm tra dữ liệu và dựng các split train, occluded query và gallery theo quy ước filename. Đây là biến thể đánh giá ReID dưới che khuất.",
  "datasets/sampler.py": "Cài đặt RandomIdentitySampler theo chiến lược P identities × K instances để tạo mini-batch phù hợp Triplet Loss. Sampler bù mẫu khi identity có ít ảnh và ước lượng độ dài epoch.",
  "datasets/sampler_ddp.py": "Mở rộng identity-balanced sampling cho DistributedDataParallel, đồng bộ seed giữa ranks và chia các block mini-batch theo worker. Mô-đun có tiện ích all-gather đối tượng Python qua tensor được padding.",
  "datasets/vehicleid.py": "Parser VehicleID với các protocol test 800/1600/2400 identities, relabel tập train và chọn ngẫu nhiên một ảnh gallery cho mỗi vehicle test. Camera giả lập khác nhau được gán cho query/gallery để tương thích evaluator ReID.",
  "datasets/veri.py": "Parser VeRi-776 kết hợp person/vehicle ID và camera từ filename với viewpoint annotation ngoài. Các ảnh thiếu viewpoint bị loại trước khi tạo train/query/gallery.",
  "utils/iotools.py": "Cung cấp helper filesystem và JSON gồm tạo thư mục, kiểm tra file, đọc và ghi JSON. Các tiện ích này hỗ trợ quản lý dữ liệu/checkpoint của pipeline ReID.",
  "tests/test_notebook_helpers.py": "Kiểm thử regression cho notebook FER: run ID, publish artifact, progress renderer, cache landmark, precedence cấu hình và các invariant anatomy/geometry mặc định. Suite cũng kiểm tra notebook FER2013/RAF-DB thực sự chuyển các điều khiển GPU, Stage 1/2 và uncertainty xuống CLI.",
  "utils/notebook_landmarks.py": "Quản lý download-once và cache landmark artifacts cho notebook bằng SHA-256, schema version và chữ ký tham số tiền xử lý. Mô-đun xác minh manifest có tính di động, gọi builder một cách tái lập và ghi metadata/log cache.",
  "utils/notebook_progress.py": "Chuyển output subprocess của các phase Stage1-base, Stage1-geometry và Stage2 thành progress bar HTML an toàn trong notebook, đồng thời stream log văn bản. Bộ đệm xử lý cả carriage-return của tqdm lẫn newline thông thường.",
  "utils/notebook_run.py": "Quản lý vòng đời artifact của notebook bằng run ID có timezone, staging directory riêng và chỉ publish sau khi provenance đã được khởi tạo. Visual, console log và metadata được gom vào đúng immutable run directory.",
};

const ENTITY_SUMMARIES = {
  "CenterLoss": "Tính bình phương khoảng cách giữa embedding và tâm lớp tương ứng rồi lấy trung bình theo batch.",
  "make_loss": "Đọc cấu hình để ghép classification, triplet và image-to-text loss, đồng thời khởi tạo Center Loss.",
  "CrossEntropyLabelSmooth": "Tính cross-entropy với target one-hot đã được làm mượt theo epsilon và số lớp.",
  "LabelSmoothingCrossEntropy": "Biến thể label-smoothed cross-entropy kết hợp negative log-likelihood với trung bình log-probability.",
  "SupConLoss": "Tính supervised contrastive objective hai chiều từ similarity logits và mask nhãn dương.",
  "normalize": "Chuẩn hóa tensor theo L2 norm trên trục được chỉ định.",
  "euclidean_dist": "Tạo ma trận khoảng cách Euclidean đôi một giữa hai tập embedding.",
  "cosine_dist": "Tạo ma trận cosine distance đôi một sau khi chuẩn hóa embedding.",
  "hard_example_mining": "Chọn positive khó nhất và negative khó nhất trong ma trận khoảng cách theo nhãn identity.",
  "TripletLoss": "Bao gói hard-example mining và MarginRankingLoss hoặc soft-margin loss cho embedding ReID.",
  "do_train": "Chạy toàn bộ epoch train baseline với AMP, cập nhật optimizer, checkpoint và đánh giá định kỳ.",
  "do_train_stage1": "Cache image embedding rồi tối ưu prompt bằng contrastive loss image-to-text và text-to-image.",
  "do_train_stage2": "Cache text prototype theo lớp và huấn luyện visual ReID bằng loss tổng hợp cùng logits image-to-text.",
  "do_inference": "Trích xuất feature query/gallery và trả metric xếp hạng từ evaluator CMC/mAP.",
  "CosineLRScheduler": "Sinh learning rate cosine có warmup, restart, decay theo chu kỳ và tùy chọn noise.",
  "WarmupMultiStepLR": "Warmup learning rate rồi giảm theo gamma tại các milestone đã cấu hình.",
  "make_optimizer": "Tạo optimizer model theo từng parameter group và optimizer riêng cho Center Loss.",
  "make_optimizer_1stage": "Chỉ đưa tham số prompt learner vào optimizer Stage 1 với hyperparameter riêng.",
  "make_optimizer_2stage": "Tạo optimizer Stage 2 cho model trừ prompt learner và optimizer riêng cho Center Loss.",
  "Scheduler": "Cung cấp state/update protocol chung và cơ chế noise cho các learning-rate scheduler.",
  "create_scheduler": "Khởi tạo cosine scheduler theo epoch với warmup và learning-rate tối thiểu.",
  "set_seed": "Đồng bộ seed của Python, NumPy và PyTorch/CUDA, đồng thời cấu hình cuDNN cho lần chạy.",
  "setup_logger": "Cấu hình logger console/file và ngăn nhân đôi handler.",
  "AverageMeter": "Theo dõi giá trị hiện tại, tổng, số mẫu và trung bình chạy của một metric.",
  "euclidean_distance": "Tính ma trận squared Euclidean distance giữa query và gallery features.",
  "cosine_similarity": "Biến cosine similarity thành angular distance ổn định số bằng clipping.",
  "eval_func": "Tính CMC và mean Average Precision theo protocol Market1501 sau khi loại same-camera matches.",
  "R1_mAP_eval": "Tích lũy features theo batch rồi tính CMC/mAP, có normalization và re-ranking tùy chọn.",
  "re_ranking": "Tái xếp hạng query-gallery bằng k-reciprocal expansion, local query expansion và Jaccard distance.",
  "read_image": "Đọc ảnh RGB và thử lại khi có lỗi I/O tạm thời.",
  "BaseDataset": "Cung cấp thống kê identity, ảnh, camera và view chung cho dataset ReID.",
  "BaseImageDataset": "Mở rộng dataset nền bằng bảng thống kê train/query/gallery dành cho image ReID.",
  "ImageDataset": "Adapter PyTorch Dataset đọc tuple metadata, áp transform và trả tensor ảnh cùng nhãn.",
  "DukeMTMCreID": "Kiểm tra và phân tích các split DukeMTMC-reID thành metadata chuẩn hóa.",
  "Market1501": "Phân tích tên ảnh Market1501, bỏ junk ID và relabel identity huấn luyện.",
  "MSMT17": "Đọc split list MSMT17 và hợp nhất train/val thành tập huấn luyện được relabel.",
  "OCC_DukeMTMCreID": "Dựng protocol Occluded-Duke cho train, query bị che khuất và gallery.",
  "VehicleID": "Dựng split VehicleID theo kích thước test và tạo protocol một-gallery-mỗi-identity.",
  "VeRi": "Dựng các split VeRi-776 có camera và viewpoint annotation.",
  "train_collate_fn": "Gộp batch train thành tensor ảnh, identity, camera và view IDs.",
  "val_collate_fn": "Gộp batch validation, giữ cả camera gốc và đường dẫn ảnh cho evaluator.",
  "make_dataloader": "Dựng augmentation, sampler và các DataLoader train/validation theo dataset cấu hình.",
  "RandomIdentitySampler": "Lấy mẫu P identities × K instances để mỗi mini-batch có positive pairs cho metric learning.",
  "_get_global_gloo_group": "Trả process group Gloo toàn cục để trao đổi dữ liệu tùy ý giữa distributed ranks.",
  "_serialize_to_tensor": "Serialize đối tượng picklable thành byte tensor trên thiết bị phù hợp backend.",
  "_pad_to_largest_tensor": "Thu thập kích thước và padding byte tensor về kích thước lớn nhất trước all-gather.",
  "all_gather": "Thu thập đối tượng Python có kích thước biến thiên từ mọi distributed rank.",
  "shared_random_seed": "Phát sinh seed chung bằng cách all-gather giá trị từ rank đầu.",
  "RandomIdentitySampler_DDP": "Tạo identity-balanced batches nhất quán rồi phân phối các block mẫu cho từng DDP rank.",
  "mkdir_if_missing": "Tạo thư mục đích theo kiểu idempotent nếu chưa tồn tại.",
  "check_isfile": "Kiểm tra đường dẫn file và phát cảnh báo nếu không tồn tại.",
  "read_json": "Đọc và giải mã một tài liệu JSON UTF-8.",
  "write_json": "Tạo thư mục cha rồi ghi đối tượng thành JSON có thụt lề.",
  "_sha256": "Tính SHA-256 theo chunk để nhận diện nội dung artifact lớn.",
  "download_once": "Tải model vào file tạm và atomically replace đích, bỏ qua nếu cache hợp lệ đã tồn tại.",
  "_cache_metadata_path": "Suy ra đường dẫn metadata cache từ landmark manifest đầu ra.",
  "_read_json": "Đọc JSON an toàn và trả object rỗng khi file lỗi hoặc payload không hợp lệ.",
  "_nonempty_line_count": "Đếm số record không rỗng trong manifest dạng JSONL.",
  "validate_landmark_manifest_layout": "Xác minh manifest đồng vị trí với dataset và mọi landmark_path tương đối, nằm trong data_dir, có artifact thật.",
  "prepare_cached_landmarks": "So khớp cache signature; nếu miss thì gọi builder landmark, kiểm tra output và ghi metadata tái lập.",
  "_is_progress_update": "Nhận diện dòng tqdm thuộc Stage1-base, Stage1-geometry hoặc Stage2.",
  "_progress_html": "Render một dòng tiến độ thành HTML progress bar đã escape nội dung.",
  "stream_process_output": "Stream stdout theo ký tự, cập nhật progress display tại chỗ và đồng thời ghi log đầy đủ.",
  "timestamped_run_id": "Tạo run ID an toàn cho filesystem từ prefix, local timezone, microsecond và seed tùy chọn.",
  "prepare_notebook_staging": "Tạo staging directory theo run dưới outputs/.notebook_staging mà chưa chiếm output chính thức.",
  "publish_notebook_artifacts": "Chỉ sau khi có provenance, sao chép visual và ghi console/metadata vào immutable run directory.",
};

const TEST_SUMMARIES = {
  test_timestamped_run_id_is_local_time_safe_and_precise: "Kiểm tra run ID bảo toàn local timezone, microsecond và seed trong định dạng an toàn cho filesystem.",
  test_publish_notebook_artifacts_requires_initialized_run: "Kiểm tra publish bị từ chối khi output run chưa có provenance.json.",
  test_publish_notebook_artifacts_copies_visuals_and_log: "Kiểm tra visual, console log và notebook metadata được publish đúng vào run đã khởi tạo.",
  test_notebook_progress_recognizes_all_training_phase_labels: "Kiểm tra progress parser nhận đủ nhãn Stage1-base, Stage1-geometry và Stage2.",
  test_notebook_progress_html_keeps_stage1_phase_in_single_display: "Kiểm tra HTML renderer giữ phase Stage 1 trong cùng một progress display.",
  test_training_notebooks_forward_resolved_runtime_controls: "Kiểm tra notebook FER2013/RAF-DB chuyển toàn bộ runtime controls đã resolve xuống lệnh train.",
  test_validate_landmark_manifest_layout_accepts_colocated_relative_artifacts: "Kiểm tra validator chấp nhận manifest đồng vị trí có artifact path tương đối hợp lệ.",
  test_validate_landmark_manifest_layout_rejects_missing_artifact: "Kiểm tra validator báo lỗi khi landmark manifest trỏ đến artifact bị thiếu.",
  test_rafdb_notebooks_share_explicit_colocated_landmark_paths: "Kiểm tra notebook build/train RAF-DB dùng chung đường dẫn anatomy manifest và artifact rõ ràng.",
  test_anatomy_is_required_by_default_and_quick_presets: "Khóa invariant rằng anatomy, hybrid routing, geometry và anatomy-aware uncertainty được bật trong default/quick presets.",
  test_notebook_cli_overrides_have_highest_precedence_and_source_provenance: "Kiểm tra CLI/notebook overrides có độ ưu tiên cao nhất và nguồn cấu hình được ghi đúng.",
  test_stage1_both_rejects_a_missing_phase: "Kiểm tra Stage 1 mode `both` từ chối cấu hình thiếu base hoặc geometry epochs.",
  test_stage1_geometry_rejects_an_inactive_prompt_conditioner: "Kiểm tra geometry phase từ chối anatomy prompt conditioner không hoạt động.",
  test_rafdb_notebook_exposes_gpu_safety_controls: "Kiểm tra notebook RAF-DB công khai batch size, gradient accumulation, AMP, clipping và corruption controls an toàn GPU.",
  test_hf_fer2013_notebook_and_presets_enable_stage1b_geometry_prompt: "Kiểm tra FER2013 notebook/presets bật Stage 1b geometry prompt với phân bổ epoch đúng.",
};

function basename(filePath) {
  return filePath.split("/").at(-1);
}

function complexity(lines) {
  if (lines < 50) return "simple";
  if (lines <= 200) return "moderate";
  return "complex";
}

function isConfig(filePath) {
  return filePath.endsWith(".yml") || filePath.endsWith(".yaml");
}

function configInfo(filePath) {
  const lower = filePath.toLowerCase();
  const vehicleId = lower.includes("/vehicleid/");
  const veri = lower.includes("/veri/");
  const dataset = vehicleId ? "VehicleID" : veri ? "VeRi-776" : "person ReID (Market1501/DukeMTMC/MSMT17/Occluded-Duke)";
  const domainTag = vehicleId || veri ? "vehicle-reid" : "person-reid";
  const vit = lower.includes("vit_");
  const backbone = vit ? "ViT-B/16" : "RN50";
  const backboneTag = vit ? "vit-b16" : "rn50";
  const twoStage = lower.includes("clipreid") || lower.includes("_prom");
  const summary = twoStage
    ? `Cấu hình ${dataset} cho ${backbone} theo quy trình CLIP-ReID hai giai đoạn: học prompt/text alignment ở Stage 1 rồi tối ưu ReID ở Stage 2. File đặt kích thước ảnh, identity sampler, loss weights, optimizer, warmup, lịch giảm learning rate và protocol đánh giá.`
    : `Cấu hình baseline ${dataset} với backbone ${backbone}, softmax-triplet sampling và augmentation chuẩn ReID. File đặt loss weights, batch size, optimizer, warmup/milestones và protocol CMC/mAP.`;
  const tags = twoStage
    ? ["configuration", domainTag, backboneTag, "clip-reid", "two-stage"]
    : ["configuration", domainTag, backboneTag, "baseline"];
  const languageNotes = twoStage
    ? "YAML phân cấp SOLVER.STAGE1/STAGE2 để hai giai đoạn dùng batch size, learning rate và scheduler độc lập."
    : "YAML được merge vào cấu hình YACS; các trường DATASETS/OUTPUT_DIR được để dạng gợi ý để người chạy ghi đè.";
  return { summary, tags, languageNotes };
}

function fileTags(filePath) {
  if (isConfig(filePath)) return configInfo(filePath).tags;
  if (filePath === "config/__init__.py") return ["configuration", "entry-point", "python"];
  if (filePath.startsWith("loss/")) {
    if (filePath.endsWith("make_loss.py")) return ["factory", "loss-function", "metric-learning", "pytorch"];
    if (filePath.endsWith("supcontrast.py")) return ["loss-function", "contrastive-learning", "clip-reid"];
    return ["loss-function", "metric-learning", "pytorch"];
  }
  if (filePath.startsWith("processor/")) return ["training-loop", "evaluation", "mixed-precision", filePath.includes("clipreid") ? "clip-reid" : "reid"];
  if (filePath.startsWith("solver/")) return ["optimization", "learning-rate", filePath.includes("optimizer") ? "factory" : "scheduler"];
  if (filePath === "train.py") return ["entry-point", "training", "reid", "reproducibility"];
  if (filePath === "train_clipreid.py") return ["entry-point", "training", "clip-reid", "two-stage"];
  if (filePath === "test.py") return ["entry-point", "inference", "evaluation", "reid"];
  if (filePath === "test_clipreid.py") return ["entry-point", "inference", "evaluation", "clip-reid"];
  if (filePath === "utils/logger.py") return ["utility", "logging", "experiment-management"];
  if (filePath === "utils/meter.py") return ["utility", "metrics", "training-loop"];
  if (filePath === "utils/metrics.py") return ["evaluation", "reid-metrics", "cmc", "map"];
  if (filePath === "utils/reranking.py") return ["re-ranking", "metric-learning", "retrieval"];
  if (filePath === "datasets/bases.py") return ["data-model", "dataset", "image-loading", "reid"];
  if (filePath.includes("make_dataloader")) return ["factory", "data-pipeline", "augmentation", filePath.includes("clipreid") ? "clip-reid" : "reid"];
  if (filePath.includes("sampler_ddp")) return ["data-pipeline", "distributed-training", "identity-sampling"];
  if (filePath.includes("sampler.py")) return ["data-pipeline", "identity-sampling", "metric-learning"];
  if (filePath.startsWith("datasets/")) return ["dataset", "data-parser", filePath.includes("vehicle") || filePath.includes("veri") ? "vehicle-reid" : "person-reid"];
  if (filePath === "utils/iotools.py") return ["utility", "filesystem", "serialization"];
  if (filePath === "tests/test_notebook_helpers.py") return ["test", "notebook", "reproducibility", "anatomy"];
  if (filePath === "utils/notebook_landmarks.py") return ["artifact-cache", "preprocessing", "anatomy", "reproducibility"];
  if (filePath === "utils/notebook_progress.py") return ["notebook", "progress-streaming", "subprocess", "logging"];
  if (filePath === "utils/notebook_run.py") return ["notebook", "artifact-management", "provenance", "reproducibility"];
  return ["python", "project-module", "utility"];
}

function fileLanguageNotes(filePath) {
  if (isConfig(filePath)) return configInfo(filePath).languageNotes;
  const notes = {
    "config/__init__.py": "Import có điều kiện chỉ nuốt ModuleNotFoundError khi dependency thiếu chính xác là `yacs`; các lỗi import khác vẫn được ném lại.",
    "processor/processor.py": "PyTorch AMP dùng GradScaler/autocast; DataParallel và nhánh distributed checkpoint được xử lý trong cùng processor.",
    "processor/processor_clipreid_stage1.py": "Image features được tính một lần dưới `torch.no_grad()`, nên gradient Stage 1 chỉ đi qua text/prompt branch.",
    "processor/processor_clipreid_stage2.py": "Text features theo toàn bộ class được cache trước epoch loop để làm classifier prototype cố định trong Stage 2.",
    "solver/cosine_lr.py": "Implementation kế thừa scheduler kiểu timm và dẫn chiếu SGDR/cosine annealing with warm restarts.",
    "utils/reranking.py": "Implementation theo k-reciprocal encoding re-ranking của Zhong et al., CVPR 2017.",
    "datasets/sampler_ddp.py": "Đối tượng Python được pickle thành byte tensor, padding đồng kích thước rồi all-gather qua Gloo/NCCL.",
    "utils/notebook_landmarks.py": "Cache key kết hợp schema version, SHA-256 input/model và toàn bộ tham số jitter để phát hiện thay đổi có ảnh hưởng kết quả.",
    "utils/notebook_progress.py": "Đọc stdout từng ký tự để bảo toàn ngữ nghĩa carriage-return của tqdm trong môi trường Jupyter.",
    "utils/notebook_run.py": "Cổng `provenance.json` ngăn notebook ghi artifact vào output chưa được entry point khởi tạo.",
  };
  return notes[filePath];
}

function entitySummary(filePath, name, type) {
  if (TEST_SUMMARIES[name]) return TEST_SUMMARIES[name];
  if (name === "make_dataloader" && filePath.includes("clipreid")) {
    return "Dựng riêng Stage 1 loader, Stage 2 identity-sampled loader và validation loader cho CLIP-ReID.";
  }
  if (name === "make_dataloader") return ENTITY_SUMMARIES.make_dataloader;
  if (name === "do_inference" && filePath.includes("clipreid")) {
    return "Chạy inference CLIP-ReID, tích lũy feature và trả Rank-1/Rank-5 từ CMC/mAP evaluator.";
  }
  if (ENTITY_SUMMARIES[name]) return ENTITY_SUMMARIES[name];
  return type === "class"
    ? `Đóng gói trách nhiệm ${name} trong pipeline và cung cấp các phương thức nghiệp vụ được file này sử dụng.`
    : `Thực hiện thao tác ${name} phục vụ luồng xử lý chính của mô-đun.`;
}

function entityTags(filePath, name, type) {
  const kind = type === "class" ? "class" : "function";
  if (name.startsWith("test_")) return ["test", "regression", "notebook", "reproducibility"];
  if (filePath.startsWith("loss/")) return [kind, "loss-function", name === "SupConLoss" ? "contrastive-learning" : "metric-learning"];
  if (filePath.startsWith("processor/")) return [kind, "training-loop", name === "do_inference" ? "evaluation" : "mixed-precision"];
  if (filePath.startsWith("solver/")) return [kind, "optimization", name.includes("cheduler") || name.includes("scheduler") ? "scheduler" : "factory"];
  if (["train.py", "train_clipreid.py"].includes(filePath)) return [kind, "reproducibility", "entry-point"];
  if (filePath === "utils/metrics.py") return [kind, "evaluation", "reid-metrics"];
  if (filePath === "utils/reranking.py") return [kind, "re-ranking", "retrieval"];
  if (filePath === "datasets/bases.py") return [kind, "dataset", "image-loading"];
  if (filePath.includes("make_dataloader")) return [kind, "data-pipeline", "dataloader"];
  if (filePath.includes("sampler_ddp")) return [kind, "distributed-training", "identity-sampling"];
  if (filePath.includes("sampler.py")) return [kind, "identity-sampling", "metric-learning"];
  if (filePath.startsWith("datasets/")) return [kind, "dataset", "data-parser"];
  if (filePath === "utils/iotools.py") return [kind, "filesystem", "serialization"];
  if (filePath === "utils/notebook_landmarks.py") return [kind, "artifact-cache", "anatomy"];
  if (filePath === "utils/notebook_progress.py") return [kind, "progress-streaming", "notebook"];
  if (filePath === "utils/notebook_run.py") return [kind, "artifact-management", "provenance"];
  if (filePath === "utils/logger.py") return [kind, "logging", "utility"];
  if (filePath === "utils/meter.py") return [kind, "metrics", "utility"];
  return [kind, "python", "utility"];
}

function significant(item, exported, type) {
  const length = item.endLine - item.startLine + 1;
  if (exported.has(item.name)) return true;
  if (type === "function") return length >= 10;
  return length >= 20 || (item.methods?.length ?? 0) >= 2;
}

function addEdge(edges, source, target, type, weight) {
  if (source === target) throw new Error(`Self edge: ${source}`);
  if (!edges.some((edge) => edge.source === source && edge.target === target && edge.type === type)) {
    edges.push({ source, target, type, direction: "forward", weight });
  }
}

function semanticEdges(batchIndex, edges) {
  if (batchIndex === 1) {
    addEdge(edges, "class:solver/cosine_lr.py:CosineLRScheduler", "class:solver/scheduler.py:Scheduler", "inherits", 0.9);
    addEdge(edges, "function:loss/make_loss.py:make_loss", "class:loss/center_loss.py:CenterLoss", "calls", 0.8);
    addEdge(edges, "function:loss/make_loss.py:make_loss", "class:loss/triplet_loss.py:TripletLoss", "calls", 0.8);
    addEdge(edges, "function:loss/make_loss.py:make_loss", "class:loss/softmax_loss.py:CrossEntropyLabelSmooth", "calls", 0.8);
    addEdge(edges, "function:processor/processor.py:do_train", "class:utils/meter.py:AverageMeter", "calls", 0.8);
    addEdge(edges, "function:processor/processor.py:do_train", "class:utils/metrics.py:R1_mAP_eval", "calls", 0.8);
    addEdge(edges, "function:processor/processor.py:do_inference", "class:utils/metrics.py:R1_mAP_eval", "calls", 0.8);
    addEdge(edges, "function:processor/processor_clipreid_stage1.py:do_train_stage1", "class:loss/supcontrast.py:SupConLoss", "calls", 0.8);
    addEdge(edges, "function:processor/processor_clipreid_stage1.py:do_train_stage1", "class:utils/meter.py:AverageMeter", "calls", 0.8);
    addEdge(edges, "function:processor/processor_clipreid_stage2.py:do_train_stage2", "class:loss/supcontrast.py:SupConLoss", "calls", 0.8);
    addEdge(edges, "function:processor/processor_clipreid_stage2.py:do_train_stage2", "class:utils/metrics.py:R1_mAP_eval", "calls", 0.8);
    addEdge(edges, "function:solver/scheduler_factory.py:create_scheduler", "class:solver/cosine_lr.py:CosineLRScheduler", "calls", 0.8);
    addEdge(edges, "class:utils/metrics.py:R1_mAP_eval", "function:utils/metrics.py:eval_func", "calls", 0.8);
    addEdge(edges, "class:utils/metrics.py:R1_mAP_eval", "function:utils/metrics.py:euclidean_distance", "calls", 0.8);
    addEdge(edges, "class:utils/metrics.py:R1_mAP_eval", "function:utils/reranking.py:re_ranking", "calls", 0.8);
  }
  if (batchIndex === 3) {
    addEdge(edges, "class:datasets/bases.py:BaseImageDataset", "class:datasets/bases.py:BaseDataset", "inherits", 0.9);
    for (const [filePath, className] of [
      ["datasets/dukemtmcreid.py", "DukeMTMCreID"],
      ["datasets/market1501.py", "Market1501"],
      ["datasets/msmt17.py", "MSMT17"],
      ["datasets/occ_duke.py", "OCC_DukeMTMCreID"],
      ["datasets/vehicleid.py", "VehicleID"],
      ["datasets/veri.py", "VeRi"],
    ]) addEdge(edges, `class:${filePath}:${className}`, "class:datasets/bases.py:BaseImageDataset", "inherits", 0.9);
    for (const loaderPath of ["datasets/make_dataloader.py", "datasets/make_dataloader_clipreid.py"]) {
      const source = `function:${loaderPath}:make_dataloader`;
      addEdge(edges, source, "class:datasets/bases.py:ImageDataset", "calls", 0.8);
      addEdge(edges, source, "class:datasets/sampler.py:RandomIdentitySampler", "calls", 0.8);
      addEdge(edges, source, "class:datasets/sampler_ddp.py:RandomIdentitySampler_DDP", "calls", 0.8);
    }
    addEdge(edges, "function:datasets/sampler_ddp.py:all_gather", "function:datasets/sampler_ddp.py:_get_global_gloo_group", "calls", 0.8);
    addEdge(edges, "function:datasets/sampler_ddp.py:all_gather", "function:datasets/sampler_ddp.py:_serialize_to_tensor", "calls", 0.8);
    addEdge(edges, "function:datasets/sampler_ddp.py:all_gather", "function:datasets/sampler_ddp.py:_pad_to_largest_tensor", "calls", 0.8);
    addEdge(edges, "function:datasets/sampler_ddp.py:shared_random_seed", "function:datasets/sampler_ddp.py:all_gather", "calls", 0.8);
    addEdge(edges, "class:datasets/sampler_ddp.py:RandomIdentitySampler_DDP", "function:datasets/sampler_ddp.py:shared_random_seed", "calls", 0.8);
  }
  if (batchIndex === 5) {
    addEdge(edges, "function:utils/notebook_landmarks.py:prepare_cached_landmarks", "function:utils/notebook_landmarks.py:_sha256", "calls", 0.8);
    addEdge(edges, "function:utils/notebook_landmarks.py:prepare_cached_landmarks", "function:utils/notebook_landmarks.py:_cache_metadata_path", "calls", 0.8);
    addEdge(edges, "function:utils/notebook_landmarks.py:prepare_cached_landmarks", "function:utils/notebook_landmarks.py:_read_json", "calls", 0.8);
    addEdge(edges, "function:utils/notebook_landmarks.py:prepare_cached_landmarks", "function:utils/notebook_landmarks.py:_nonempty_line_count", "calls", 0.8);
    addEdge(edges, "function:utils/notebook_progress.py:stream_process_output", "function:utils/notebook_progress.py:_is_progress_update", "calls", 0.8);
    addEdge(edges, "function:utils/notebook_progress.py:stream_process_output", "function:utils/notebook_progress.py:_progress_html", "calls", 0.8);
    const testFile = "file:tests/test_notebook_helpers.py";
    for (const production of [
      "utils/notebook_progress.py",
      "utils/notebook_run.py",
      "utils/notebook_landmarks.py",
      "config/emotion_defaults.py",
    ]) addEdge(edges, `file:${production}`, testFile, "tested_by", 0.5);
  }
  if ([7, 9, 10].includes(batchIndex)) {
    const pairs = batchIndex === 7
      ? [["configs/VehicleID/cnn_base.yml", "configs/VehicleID/cnn_clipreid.yml"], ["configs/VehicleID/vit_base.yml", "configs/VehicleID/vit_clipreid.yml"]]
      : batchIndex === 9
        ? [["configs/person/cnn_base.yml", "configs/person/cnn_clipreid.yml"], ["configs/person/vit_base.yml", "configs/person/vit_clipreid.yml"]]
        : [["configs/veri/cnn_base.yml", "configs/veri/cnn_prom.yml"], ["configs/veri/vit_base.yml", "configs/veri/vit_prom.yml"]];
    for (const [base, prompt] of pairs) addEdge(edges, `config:${base}`, `config:${prompt}`, "related", 0.5);
  }
}

function buildBatch(batchIndex) {
  const batch = batchesPayload.batches.find((item) => item.batchIndex === batchIndex);
  const extract = JSON.parse(fs.readFileSync(path.join(tmp, `ua-file-extract-results-${batchIndex}.json`), "utf8"));
  if (!extract.scriptCompleted || extract.filesAnalyzed !== batch.files.length || extract.filesSkipped.length) {
    throw new Error(`Batch ${batchIndex}: structural extraction incomplete`);
  }
  const nodes = [];
  const edges = [];
  for (const result of extract.results) {
    const filePath = result.path;
    const config = isConfig(filePath);
    const fileId = `${config ? "config" : "file"}:${filePath}`;
    const cfgInfo = config ? configInfo(filePath) : undefined;
    const fileNode = {
      id: fileId,
      type: config ? "config" : "file",
      name: basename(filePath),
      filePath,
      summary: cfgInfo?.summary ?? FILE_SUMMARIES[filePath],
      tags: fileTags(filePath),
      complexity: complexity(result.nonEmptyLines),
    };
    const note = cfgInfo?.languageNotes ?? fileLanguageNotes(filePath);
    if (note) fileNode.languageNotes = note;
    if (!fileNode.summary) throw new Error(`Missing file summary: ${filePath}`);
    nodes.push(fileNode);

    if (!config) {
      const exported = new Set((result.exports ?? []).map((item) => item.name));
      for (const [type, items] of [["function", result.functions ?? []], ["class", result.classes ?? []]]) {
        for (const item of items) {
          if (!item?.name || !Number.isInteger(item.startLine) || !Number.isInteger(item.endLine)) continue;
          if (!significant(item, exported, type)) continue;
          const id = `${type}:${filePath}:${item.name}`;
          nodes.push({
            id,
            type,
            name: item.name,
            filePath,
            lineRange: [item.startLine, item.endLine],
            summary: entitySummary(filePath, item.name, type),
            tags: entityTags(filePath, item.name, type),
            complexity: complexity(item.endLine - item.startLine + 1),
          });
          addEdge(edges, fileId, id, "contains", 1.0);
          if (exported.has(item.name)) addEdge(edges, fileId, id, "exports", 0.8);
        }
      }
      for (const targetPath of batch.batchImportData[filePath] ?? []) {
        addEdge(edges, fileId, `file:${targetPath}`, "imports", 0.7);
      }
    }
  }
  semanticEdges(batchIndex, edges);
  return { nodes, edges, batch, extract };
}

function validateBatch(batchIndex, fragment, batch) {
  const validNodeTypes = new Set(["file", "function", "class", "config", "document", "service", "table", "endpoint", "pipeline", "schema", "resource"]);
  const weights = { contains: 1.0, imports: 0.7, calls: 0.8, inherits: 0.9, implements: 0.9, exports: 0.8, depends_on: 0.6, tested_by: 0.5, configures: 0.6, documents: 0.5, deploys: 0.7, migrates: 0.7, triggers: 0.6, defines_schema: 0.8, serves: 0.7, provisions: 0.7, routes: 0.6, related: 0.5 };
  const ids = new Set();
  for (const node of fragment.nodes) {
    if (ids.has(node.id)) throw new Error(`Batch ${batchIndex}: duplicate node ${node.id}`);
    ids.add(node.id);
    if (!validNodeTypes.has(node.type)) throw new Error(`Batch ${batchIndex}: invalid node type ${node.type}`);
    if (!node.summary || !Array.isArray(node.tags) || node.tags.length < 3 || node.tags.length > 5) throw new Error(`Batch ${batchIndex}: invalid textual metadata ${node.id}`);
    if (!["simple", "moderate", "complex"].includes(node.complexity)) throw new Error(`Batch ${batchIndex}: invalid complexity ${node.id}`);
  }
  for (const file of batch.files) {
    const prefix = file.fileCategory === "config" ? "config" : "file";
    if (!ids.has(`${prefix}:${file.path}`)) throw new Error(`Batch ${batchIndex}: missing file node ${file.path}`);
  }
  const allowedPaths = new Set();
  for (const file of batch.files) {
    for (const imported of batch.batchImportData[file.path] ?? []) allowedPaths.add(imported);
    for (const neighbor of batch.neighborMap?.[file.path] ?? []) allowedPaths.add(neighbor.path);
  }
  for (const edge of fragment.edges) {
    if (edge.source === edge.target) throw new Error(`Batch ${batchIndex}: self edge ${edge.source}`);
    if (!(edge.type in weights) || edge.weight !== weights[edge.type] || edge.direction !== "forward") throw new Error(`Batch ${batchIndex}: invalid edge ${JSON.stringify(edge)}`);
    const validEndpoint = (id) => {
      if (ids.has(id)) return true;
      const match = id.match(/^(?:file|function|class):(.+?)(?::[^:]+)?$/);
      return Boolean(match && allowedPaths.has(match[1]));
    };
    if (!validEndpoint(edge.source) || !validEndpoint(edge.target)) throw new Error(`Batch ${batchIndex}: dangling edge ${JSON.stringify(edge)}`);
  }
  const expectedImports = batch.files.reduce((sum, file) => sum + (batch.batchImportData[file.path]?.length ?? 0), 0);
  const actualImports = fragment.edges.filter((edge) => edge.type === "imports").length;
  if (expectedImports !== actualImports) throw new Error(`Batch ${batchIndex}: imports ${actualImports}/${expectedImports}`);
  if (fragment.nodes.length > 60 || fragment.edges.length > 120) throw new Error(`Batch ${batchIndex}: requires multipart output (${fragment.nodes.length} nodes, ${fragment.edges.length} edges)`);
}

const report = [];
for (const batchIndex of selected) {
  const { nodes, edges, batch, extract } = buildBatch(batchIndex);
  const fragment = { nodes, edges };
  validateBatch(batchIndex, fragment, batch);
  const outputPath = path.join(intermediate, `batch-${batchIndex}.json`);
  fs.writeFileSync(outputPath, JSON.stringify(fragment, null, 2) + "\n", "utf8");
  const reread = JSON.parse(fs.readFileSync(outputPath, "utf8"));
  validateBatch(batchIndex, reread, batch);
  report.push({ batchIndex, output: path.basename(outputPath), nodes: nodes.length, edges: edges.length, skipped: extract.filesSkipped });
}
process.stdout.write(JSON.stringify(report, null, 2) + "\n");
