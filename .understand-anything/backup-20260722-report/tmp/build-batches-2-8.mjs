import fs from 'node:fs';
import path from 'node:path';

const projectRoot = 'E:/Source/EmotionCLIP-ReID';
const intermediateDir = path.join(projectRoot, '.understand-anything', 'intermediate');
const tmpDir = path.join(projectRoot, '.understand-anything', 'tmp');
const batches = JSON.parse(fs.readFileSync(path.join(intermediateDir, 'batches.json'), 'utf8')).batches;

const fileDescriptions = {
  'config/emotion_defaults.py': {
    summary: 'Định nghĩa cấu hình mặc định, cơ chế hợp nhất override và các kiểm tra bất biến cho toàn bộ pipeline EmotionCLIP hai giai đoạn. Tệp cũng theo dõi nguồn gốc từng giá trị cấu hình để việc huấn luyện có thể tái lập.',
    tags: ['configuration', 'validation', 'training-controls', 'provenance'],
  },
  'datasets/anatomy.py': {
    summary: 'Chuyển landmark khuôn mặt và metadata chất lượng thành đặc trưng hình học theo vùng dùng bởi nhánh anatomy. Mô-đun xử lý phép biến đổi crop/flip, độ bất định và thống kê hình học theo lớp.',
    tags: ['anatomy', 'landmarks', 'feature-engineering', 'uncertainty'],
  },
  'datasets/emotion_manifest.py': {
    summary: 'Nạp, chuẩn hóa và kiểm tra manifest dữ liệu cảm xúc, đồng thời xây dựng Dataset, transform an toàn cho khuôn mặt và các DataLoader tách biệt train/validation/test. Mô-đun còn đánh giá độ phủ anatomy và phát hiện rò rỉ split.',
    tags: ['dataset', 'data-loader', 'validation', 'anatomy'],
  },
  'infer_emotionclip.py': {
    summary: 'Cung cấp CLI suy luận EmotionCLIP: nạp cấu hình và checkpoint, chuẩn bị ảnh cùng anatomy artifact, chạy mô hình và xuất phân phối cảm xúc kèm độ bất định.',
    tags: ['entry-point', 'inference', 'checkpoint', 'uncertainty'],
  },
  'loss/emotion_losses.py': {
    summary: 'Cài đặt các loss cho Stage 2 gồm evidential cross-entropy, KL Dirichlet, calibration độ tin cậy, giám sát routing và ranking giữa mẫu sạch với mẫu bị làm hỏng.',
    tags: ['loss', 'evidential-learning', 'reliability', 'optimization'],
  },
  'processor/processor_emotionclip.py': {
    summary: 'Điều phối huấn luyện và đánh giá EmotionCLIP ở cả Stage 1 lẫn Stage 2, gồm AMP, gradient accumulation, corruption-based reliability, checkpoint và early stopping. Tệp thực thi nhiều guardrail để dừng an toàn khi loss, output hoặc tham số trở nên không hữu hạn.',
    tags: ['training-loop', 'evaluation', 'checkpoint', 'reliability', 'numerical-safety'],
  },
  'tests/test_anatomy_audit.py': {
    summary: 'Kiểm thử công cụ audit hình học anatomy, đặc biệt việc quy đổi jitter landmark sang đúng đơn vị đặc trưng và phép đo signal-to-jitter.',
    tags: ['test', 'anatomy', 'geometry', 'regression'],
  },
  'tests/test_emotion_losses_metrics.py': {
    summary: 'Kiểm thử các loss evidential/reliability và metric FER, bao gồm tính bất biến, gradient, calibration và khả năng phân tách OOD bằng uncertainty.',
    tags: ['test', 'loss', 'metrics', 'uncertainty'],
  },
  'tests/test_emotion_manifest.py': {
    summary: 'Kiểm thử chuẩn hóa nhãn, phân tách dữ liệu, chống leakage và phép biến đổi landmark trong manifest FER2013/RAF-DB. Bộ test cũng xác minh fallback rõ ràng khi anatomy bị thiếu.',
    tags: ['test', 'dataset', 'split-validation', 'landmarks'],
  },
  'tests/test_emotion_model_units.py': {
    summary: 'Kiểm thử cấp đơn vị cho prompt learner, adapter, fusion, router và các quy tắc đóng băng tham số theo stage của EmotionCLIP. Các test bảo vệ tính đúng đắn của fallback khi chất lượng anatomy thấp hoặc vùng bị thiếu.',
    tags: ['test', 'model', 'fusion', 'routing', 'training-stages'],
  },
  'tests/test_emotion_processor_smoke.py': {
    summary: 'Chạy smoke test CPU cho processor Stage 1/2 và các guardrail huấn luyện như gradient accumulation, warmup reliability, early stopping, checkpoint schema và lỗi non-finite.',
    tags: ['test', 'training-loop', 'checkpoint', 'numerical-safety'],
  },
  'tests/test_run_artifacts.py': {
    summary: 'Kiểm thử tính bất biến của thư mục run, metadata provenance và yêu cầu chỉ định run ID khi tra cứu artifact.',
    tags: ['test', 'artifacts', 'provenance', 'reproducibility'],
  },
  'tests/test_train_emotionclip_parallel.py': {
    summary: 'Kiểm thử lựa chọn GPU, giới hạn topology multi-GPU và hành vi gather/unwrap của lớp song song hóa EmotionCLIP. Bộ test cũng kiểm tra bất biến confidence và text feature dùng chung.',
    tags: ['test', 'multi-gpu', 'data-parallel', 'validation'],
  },
  'tools/audit_anatomy_geometry.py': {
    summary: 'CLI audit chất lượng geometry feature từ anatomy artifact, tổng hợp phân phối tín hiệu và so sánh tín hiệu giữa lớp với nhiễu jitter đo được.',
    tags: ['cli', 'anatomy', 'quality-audit', 'statistics'],
  },
  'tools/build_face_landmark_artifacts.py': {
    summary: 'Tạo anatomy artifact từ ảnh bằng MediaPipe Face Landmarker, kèm pose, visibility/presence và ước lượng bất định qua jitter. Công cụ ghi tham chiếu artifact di động để manifest có thể tái sử dụng.',
    tags: ['cli', 'landmarks', 'mediapipe', 'artifact-generation'],
  },
  'tools/convert_affwild2_to_emotion_jsonl.py': {
    summary: 'Chuyển annotation Aff-Wild2 theo frame thành manifest JSONL thống nhất cho pipeline EmotionCLIP, giữ thông tin video và split.',
    tags: ['data-pipeline', 'conversion', 'aff-wild2', 'manifest'],
  },
  'tools/convert_fer2013_to_emotion_jsonl.py': {
    summary: 'Chuyển FER2013 từ CSV pixel hoặc cây thư mục ảnh sang manifest JSONL, ánh xạ nhãn và bảo toàn official split khi có.',
    tags: ['data-pipeline', 'conversion', 'fer2013', 'manifest'],
  },
  'tools/convert_raf_au_to_emotion_jsonl.py': {
    summary: 'Chuyển bảng Action Unit của RAF sang bản ghi JSONL bổ sung AU label và mô tả văn bản cho dữ liệu cảm xúc.',
    tags: ['data-pipeline', 'conversion', 'raf-db', 'action-units'],
  },
  'tools/convert_rafdb_to_emotion_jsonl.py': {
    summary: 'Khám phá archive và label RAF-DB, phân giải ảnh aligned/original rồi tạo manifest JSONL với official split hoặc validation phân tầng xác định.',
    tags: ['data-pipeline', 'conversion', 'raf-db', 'split-management'],
  },
  'tools/download_hf_emotion_dataset.py': {
    summary: 'Tải dataset cảm xúc từ Hugging Face, chuẩn hóa nhãn và split, lưu ảnh cục bộ rồi sinh manifest JSONL tương thích EmotionCLIP.',
    tags: ['data-pipeline', 'hugging-face', 'download', 'manifest'],
  },
  'train_emotionclip.py': {
    summary: 'Entry point huấn luyện EmotionCLIP: hợp nhất cấu hình, thiết lập seed/device/logging, dựng dữ liệu và mô hình, sau đó chạy Stage 1/Stage 2 cùng đánh giá sealed test. Tệp lưu cấu hình hiệu lực và provenance của run để tái lập kết quả.',
    tags: ['entry-point', 'training', 'configuration', 'reproducibility', 'gpu'],
  },
  'utils/fer_metrics.py': {
    summary: 'Tính bộ metric FER và uncertainty gồm confusion matrix, macro/per-class F1, balanced accuracy, calibration, selective risk và OOD ranking.',
    tags: ['metrics', 'evaluation', 'calibration', 'uncertainty'],
  },
  'utils/run_artifacts.py': {
    summary: 'Khởi tạo thư mục run bất biến và ghi provenance về Git, manifest, dependency cùng phần cứng; đồng thời phân giải artifact theo run ID rõ ràng.',
    tags: ['artifacts', 'provenance', 'reproducibility', 'utility'],
  },
  'configs/emotion/vit_b16_emotionclip.yml': {
    summary: 'Cấu hình huấn luyện đầy đủ cho EmotionCLIP ViT-B/16, bao quát backbone, anatomy prompt/fusion, dữ liệu, optimizer, lịch học, loss Stage 2 và tiêu chí đánh giá.',
    tags: ['configuration', 'vit-b16', 'training-controls', 'emotionclip'],
  },
  'configs/emotion/vit_b16_emotionclip_fer2013_quick.yml': {
    summary: 'Preset chạy nhanh EmotionCLIP ViT-B/16 trên FER2013 với ngân sách epoch và tài nguyên thu gọn, dùng để kiểm tra pipeline end-to-end trước khi huấn luyện đầy đủ.',
    tags: ['configuration', 'fer2013', 'quick-run', 'training-controls'],
  },
  'configs/emotion/vit_b16_emotionclip_hf_fer2013_quick.yml': {
    summary: 'Preset chạy nhanh cho biến thể FER2013 lấy từ Hugging Face, cấu hình đường dẫn manifest và tham số huấn luyện thu gọn cho vòng lặp xác minh.',
    tags: ['configuration', 'fer2013', 'hugging-face', 'quick-run'],
  },
  'configs/emotion/vit_b16_emotionclip_rafdb_quick.yml': {
    summary: 'Preset chạy nhanh EmotionCLIP trên RAF-DB, bật các điều khiển anatomy và huấn luyện rút gọn để smoke-test dữ liệu cùng mô hình.',
    tags: ['configuration', 'raf-db', 'quick-run', 'training-controls'],
  },
};

const importantFunctions = {
  anatomy_requirement_reasons: 'Xác định các lý do cấu hình hiện tại bắt buộc phải có anatomy artifact, giúp lỗi dữ liệu được phát hiện trước khi chạy.',
  validate_emotion_cfg: 'Kiểm tra các bất biến và tổ hợp tham số của cấu hình EmotionCLIP, từ stage training đến anatomy, loss và lựa chọn checkpoint.',
  load_emotion_cfg: 'Nạp cấu hình mặc định, YAML và CLI override theo thứ tự ưu tiên, đồng thời có thể trả về provenance của từng giá trị.',
  anatomy_to_model_inputs: 'Biến anatomy artifact thành tensor landmark, geometry, validity, uncertainty và quality đúng định dạng đầu vào mô hình.',
  fit_class_geometry_statistics: 'Ước lượng thống kê geometry theo lớp từ các mẫu hợp lệ, có trọng số uncertainty và ngưỡng số mẫu tối thiểu.',
  make_emotion_dataloaders: 'Xây dựng DataLoader train, validation và test từ manifest, giữ chính sách split và collate anatomy nhất quán.',
  emotion_stage2_loss: 'Kết hợp classification, evidential uncertainty, gate/temperature, routing và reliability-ranking thành objective Stage 2.',
  corrupt_images_for_reliability: 'Tạo ảnh nhiễu hoặc bị che khuất để huấn luyện tín hiệu reliability và trả lại mask occlusion khi cần.',
  corrupt_anatomy_for_reliability: 'Làm mất hiệu lực bằng chứng anatomy tương ứng với vùng ảnh bị che để tránh supervision mâu thuẫn.',
  load_emotion_checkpoint: 'Nạp checkpoint với kiểm tra schema, stage, chữ ký mô hình và mức đầy đủ của nhánh anatomy trước khi tiếp tục hoặc suy luận.',
  do_train_emotion_stage1: 'Huấn luyện prompt nền và geometry residual theo các phase Stage 1, đánh giá validation và quản lý checkpoint được chọn.',
  do_train_emotion_stage2: 'Thực thi vòng lặp Stage 2 với AMP, tích lũy gradient, corruption ranking, kiểm tra non-finite, validation và early stopping.',
  evaluate_emotion_model: 'Chạy suy luận trên một DataLoader và tổng hợp metric phân loại, calibration, uncertainty cùng phân tích nhánh khi được yêu cầu.',
  evaluate_sealed_test: 'Đánh giá test set chỉ định bằng checkpoint đã chọn và ghi rõ split dùng cho model selection để tránh leakage.',
  compute_fer_metrics: 'Tổng hợp metric FER, calibration, uncertainty-risk và thống kê theo lớp từ nhãn, xác suất và uncertainty.',
  initialize_immutable_run: 'Tạo run directory không ghi đè và lưu cấu hình cùng provenance dữ liệu, mã nguồn, dependency và phần cứng.',
  configure_device: 'Xác thực yêu cầu GPU và chọn thiết bị CUDA/CPU phù hợp với cấu hình runtime.',
  parameter_report: 'Tổng hợp số lượng và tên nhóm tham số trainable/frozen để kiểm tra đúng phạm vi cập nhật theo stage.',
  _model_output_error: 'Kiểm tra output của mô hình có hữu hạn và thỏa các bất biến xác suất, confidence, uncertainty trước khi cập nhật optimizer.',
  _write_training_failure: 'Ghi artifact chẩn đoán khi huấn luyện thất bại, gồm loss, output, tham số và batch liên quan.',
};

function complexityFromLines(lines) {
  if (lines < 50) return 'simple';
  if (lines <= 200) return 'moderate';
  return 'complex';
}

function domainFor(filePath) {
  if (filePath.startsWith('tests/')) return 'test';
  if (filePath.includes('anatomy') || filePath.includes('landmark')) return 'anatomy';
  if (filePath.includes('emotion_manifest')) return 'dataset';
  if (filePath.includes('emotion_losses')) return 'loss';
  if (filePath.includes('fer_metrics')) return 'metrics';
  if (filePath.includes('processor')) return 'training-loop';
  if (filePath.startsWith('config/')) return 'configuration';
  if (filePath.startsWith('infer_')) return 'inference';
  if (filePath.startsWith('train_')) return 'training';
  if (filePath.includes('run_artifacts')) return 'provenance';
  if (filePath.startsWith('tools/')) return 'data-pipeline';
  return 'utility';
}

function actionFor(name) {
  const n = name.toLowerCase();
  if (n.startsWith('test_')) return 'regression';
  if (n.includes('validate') || n.includes('error') || n.includes('finite')) return 'validation';
  if (n.includes('train') || n.includes('optimizer') || n.includes('grad')) return 'optimization';
  if (n.includes('evaluat') || n.includes('metric') || n.includes('accuracy')) return 'evaluation';
  if (n.includes('load') || n.includes('read') || n.includes('download')) return 'loading';
  if (n.includes('save') || n.includes('write') || n.includes('record') || n.includes('artifact')) return 'serialization';
  if (n.includes('convert') || n.includes('collate') || n.includes('transform')) return 'transformation';
  if (n.includes('geometry') || n.includes('landmark') || n.includes('anatomy')) return 'geometry';
  if (n.includes('corrupt') || n.includes('jitter') || n.includes('flip')) return 'augmentation';
  if (n.includes('config') || n.includes('default') || n.includes('coerce')) return 'configuration';
  if (n === 'main' || n.includes('parse_args')) return 'entry-point';
  return 'utility';
}

function uniqueTags(tags) {
  const result = [];
  for (const tag of tags) {
    const normalized = tag.toLowerCase().replace(/[^a-z0-9-]+/g, '-').replace(/^-|-$/g, '');
    if (normalized && !result.includes(normalized)) result.push(normalized);
  }
  while (result.length < 3) result.push(['python', 'component', 'implementation'][result.length]);
  return result.slice(0, 5);
}

function symbolSummary(filePath, item, kind) {
  if (importantFunctions[item.name]) return importantFunctions[item.name];
  if (item.name.startsWith('test_')) {
    return `Xác minh kịch bản \`${item.name.slice(5)}\` và bảo vệ hành vi tương ứng khỏi hồi quy.`;
  }
  if (item.name === 'main') {
    const area = fileDescriptions[filePath]?.summary.split('.')[0].toLowerCase() ?? 'pipeline của tệp';
    return `Điều phối entry point cho ${area}, bao gồm xử lý tham số, thực thi luồng chính và ghi kết quả.`;
  }
  if (kind === 'class') {
    const classDescriptions = {
      EmotionSample: 'Mô hình dữ liệu cho một mẫu cảm xúc, liên kết ảnh, nhãn, split, anatomy/AU và metadata nhận dạng chuỗi.',
      FaceSafeTransform: 'Áp dụng resize, crop, jitter và horizontal flip đồng bộ giữa ảnh với landmark để bảo toàn ngữ nghĩa khuôn mặt.',
      EmotionManifestDataset: 'Dataset đọc mẫu từ manifest, nạp ảnh/anatomy và trả tensor đã biến đổi cho pipeline FER.',
      EmotionDataParallel: 'Bọc DataParallel với scatter/gather chuyên biệt cho batch anatomy và các output dùng chung giữa GPU.',
      TinyEmotionModel: 'Test double nhỏ mô phỏng API theo stage của EmotionCLIP để smoke-test processor trên CPU.',
      _CheckpointPrompt: 'Test double tối giản cho prompt learner và các buffer cần được bảo toàn trong checkpoint.',
      _CheckpointModel: 'Test double tối giản dùng để kiểm tra schema, tính đầy đủ và khả năng nạp checkpoint.',
      _TinyVisual: 'Backbone thị giác tối giản dùng để kiểm thử tích hợp model mà không cần CLIP đầy đủ.',
      _TinyClip: 'Mô hình CLIP giả lập gọn nhẹ dùng trong kiểm thử constructor và forward integration.',
      _DummyBlock: 'Residual block giả lập dùng để kiểm tra chính sách mở khóa các block cuối.',
      _Stage1Prompt: 'Prompt module giả lập dùng để kiểm tra phân tách tham số trainable giữa các phase Stage 1.',
      _TinyResidualBlock: 'Residual block tối giản hỗ trợ test backbone thị giác thu gọn.',
    };
    return classDescriptions[item.name] ?? `Đóng gói trách nhiệm \`${item.name}\` trong miền ${domainFor(filePath)}, với các method phối hợp trạng thái và xử lý chính của lớp.`;
  }
  const params = Array.isArray(item.params) && item.params.length ? ` từ ${item.params.map((p) => `\`${p}\``).join(', ')}` : '';
  return `Thực hiện phép xử lý \`${item.name}\`${params} trong miền ${domainFor(filePath)} và cung cấp kết quả cho các bước kế tiếp của pipeline.`;
}

function fileNodeType(category) {
  if (category === 'config') return ['config', 'config'];
  if (category === 'docs') return ['document', 'document'];
  if (category === 'infra') return ['service', 'service'];
  if (category === 'data') return ['schema', 'schema'];
  return ['file', 'file'];
}

function buildBatch(batchIndex) {
  const batch = batches.find((item) => item.batchIndex === batchIndex);
  if (!batch) throw new Error(`Không tìm thấy batch ${batchIndex}`);
  const extraction = JSON.parse(fs.readFileSync(path.join(tmpDir, `ua-file-extract-results-${batchIndex}.json`), 'utf8'));
  if (!extraction.scriptCompleted || extraction.results.length !== batch.files.length) {
    throw new Error(`Extraction batch ${batchIndex} không đầy đủ`);
  }

  const resultByPath = new Map(extraction.results.map((item) => [item.path, item]));
  const nodes = [];
  const edges = [];

  for (const batchFile of batch.files) {
    const result = resultByPath.get(batchFile.path);
    if (!result) throw new Error(`Thiếu extraction result cho ${batchFile.path}`);
    if (result.language !== batchFile.language || result.fileCategory !== batchFile.fileCategory) {
      throw new Error(`Metadata extraction lệch với batch cho ${batchFile.path}`);
    }
    const description = fileDescriptions[batchFile.path];
    if (!description) throw new Error(`Thiếu mô tả file ${batchFile.path}`);
    const [nodeType, prefix] = fileNodeType(batchFile.fileCategory);
    const fileId = `${prefix}:${batchFile.path}`;
    const fileNode = {
      id: fileId,
      type: nodeType,
      name: path.posix.basename(batchFile.path),
      filePath: batchFile.path,
      summary: description.summary,
      tags: description.tags,
      complexity: complexityFromLines(result.nonEmptyLines),
    };
    if (batchFile.language === 'python' && result.nonEmptyLines > 200) {
      fileNode.languageNotes = 'Mô-đun Python có nhiều hàm cấp mô-đun; các bất biến và cấu hình được truyền rõ ràng qua tham số thay vì dựa vào trạng thái ẩn.';
    } else if (batchFile.language === 'yaml') {
      fileNode.languageNotes = 'YAML được chia theo các khối MODEL, INPUT, DATASETS, DATALOADER, SOLVER, TEST và TRAIN để cấu hình toàn bộ pipeline.';
    }
    nodes.push(fileNode);

    const exported = new Set((result.exports ?? []).map((item) => item.name));
    for (const fn of result.functions ?? []) {
      const length = fn.endLine - fn.startLine + 1;
      if (length < 10 && !exported.has(fn.name)) continue;
      const fnId = `function:${batchFile.path}:${fn.name}`;
      nodes.push({
        id: fnId,
        type: 'function',
        name: fn.name,
        filePath: batchFile.path,
        lineRange: [fn.startLine, fn.endLine],
        summary: symbolSummary(batchFile.path, fn, 'function'),
        tags: uniqueTags(['function', domainFor(batchFile.path), actionFor(fn.name)]),
        complexity: complexityFromLines(length),
      });
      edges.push({ source: fileId, target: fnId, type: 'contains', direction: 'forward', weight: 1.0 });
      if (exported.has(fn.name)) {
        edges.push({ source: fileId, target: fnId, type: 'exports', direction: 'forward', weight: 0.8 });
      }
    }

    for (const cls of result.classes ?? []) {
      const length = cls.endLine - cls.startLine + 1;
      if (length < 20 && (cls.methods ?? []).length < 2 && !exported.has(cls.name)) continue;
      const clsId = `class:${batchFile.path}:${cls.name}`;
      nodes.push({
        id: clsId,
        type: 'class',
        name: cls.name,
        filePath: batchFile.path,
        lineRange: [cls.startLine, cls.endLine],
        summary: symbolSummary(batchFile.path, cls, 'class'),
        tags: uniqueTags(['class', domainFor(batchFile.path), batchFile.path.startsWith('tests/') ? 'test-double' : 'component']),
        complexity: complexityFromLines(length),
      });
      edges.push({ source: fileId, target: clsId, type: 'contains', direction: 'forward', weight: 1.0 });
      if (exported.has(cls.name)) {
        edges.push({ source: fileId, target: clsId, type: 'exports', direction: 'forward', weight: 0.8 });
      }
    }

    if (batchFile.fileCategory === 'code') {
      for (const targetPath of batch.batchImportData[batchFile.path] ?? []) {
        edges.push({
          source: fileId,
          target: `file:${targetPath}`,
          type: 'imports',
          direction: 'forward',
          weight: 0.7,
        });
      }
    }
  }

  const nodeIds = new Set(nodes.map((node) => node.id));
  if (nodeIds.size !== nodes.length) throw new Error(`Batch ${batchIndex} có node ID trùng`);
  if (edges.some((edge) => edge.source === edge.target)) throw new Error(`Batch ${batchIndex} có self edge`);
  const expectedImports = batch.files
    .filter((file) => file.fileCategory === 'code')
    .reduce((sum, file) => sum + (batch.batchImportData[file.path] ?? []).length, 0);
  const actualImports = edges.filter((edge) => edge.type === 'imports').length;
  if (actualImports !== expectedImports) {
    throw new Error(`Batch ${batchIndex}: imports ${actualImports}/${expectedImports}`);
  }

  const partCount = Math.ceil(Math.max(nodes.length / 60, edges.length / 120, 1));
  const sortedPaths = batch.files.map((file) => file.path).sort((a, b) => a.localeCompare(b));
  const chunkSize = Math.ceil(sortedPaths.length / partCount);
  const written = [];
  const fragments = [];
  for (let index = 0; index < partCount; index += 1) {
    const partPaths = new Set(sortedPaths.slice(index * chunkSize, (index + 1) * chunkSize));
    if (partPaths.size === 0) continue;
    const partNodes = nodes.filter((node) => partPaths.has(node.filePath));
    const partNodeIds = new Set(partNodes.map((node) => node.id));
    const partEdges = edges.filter((edge) => partNodeIds.has(edge.source));
    const fragment = { nodes: partNodes, edges: partEdges };
    const fileName = partCount === 1 ? `batch-${batchIndex}.json` : `batch-${batchIndex}-part-${index + 1}.json`;
    const outputPath = path.join(intermediateDir, fileName);
    fs.writeFileSync(outputPath, `${JSON.stringify(fragment, null, 2)}\n`, 'utf8');
    const parsed = JSON.parse(fs.readFileSync(outputPath, 'utf8'));
    if (!Array.isArray(parsed.nodes) || !Array.isArray(parsed.edges)) {
      throw new Error(`${fileName} không phải GraphFragment hợp lệ`);
    }
    fragments.push(parsed);
    written.push({ fileName, nodes: partNodes.length, edges: partEdges.length });
  }

  const outputNodes = fragments.flatMap((fragment) => fragment.nodes);
  const outputEdges = fragments.flatMap((fragment) => fragment.edges);
  if (outputNodes.length !== nodes.length || outputEdges.length !== edges.length) {
    throw new Error(`Batch ${batchIndex}: tổng node/edge sau khi chia part bị lệch`);
  }
  const outputNodeIds = new Set(outputNodes.map((node) => node.id));
  if (outputNodeIds.size !== outputNodes.length) throw new Error(`Batch ${batchIndex}: node trùng giữa các part`);
  const filePathsCovered = outputNodes
    .filter((node) => ['file', 'config', 'document', 'service', 'pipeline', 'schema', 'resource'].includes(node.type))
    .map((node) => node.filePath)
    .sort((a, b) => a.localeCompare(b));
  if (JSON.stringify(filePathsCovered) !== JSON.stringify(sortedPaths)) {
    throw new Error(`Batch ${batchIndex}: coverage file node không khớp batchFiles`);
  }
  for (const node of outputNodes) {
    if (!node.id || !node.type || !node.name || !node.summary || !Array.isArray(node.tags) || node.tags.length < 3) {
      throw new Error(`Batch ${batchIndex}: node thiếu field bắt buộc: ${node.id ?? '<unknown>'}`);
    }
    if (!['simple', 'moderate', 'complex'].includes(node.complexity)) {
      throw new Error(`Batch ${batchIndex}: complexity không hợp lệ tại ${node.id}`);
    }
  }
  const importKeysExpected = [];
  for (const batchFile of batch.files.filter((file) => file.fileCategory === 'code')) {
    for (const targetPath of batch.batchImportData[batchFile.path] ?? []) {
      importKeysExpected.push(`file:${batchFile.path}->file:${targetPath}`);
    }
  }
  const importKeysActual = outputEdges
    .filter((edge) => edge.type === 'imports')
    .map((edge) => `${edge.source}->${edge.target}`);
  importKeysExpected.sort();
  importKeysActual.sort();
  if (JSON.stringify(importKeysActual) !== JSON.stringify(importKeysExpected)) {
    throw new Error(`Batch ${batchIndex}: import edge không khớp 1:1 với batchImportData`);
  }
  const allowedExternalFiles = new Set(batch.files.map((file) => file.path));
  for (const values of Object.values(batch.batchImportData)) {
    for (const targetPath of values) allowedExternalFiles.add(targetPath);
  }
  for (const [sourcePath, neighbors] of Object.entries(batch.neighborMap ?? {})) {
    allowedExternalFiles.add(sourcePath);
    for (const neighbor of neighbors) allowedExternalFiles.add(neighbor.path);
  }
  for (let index = 0; index < fragments.length; index += 1) {
    const partIds = new Set(fragments[index].nodes.map((node) => node.id));
    for (const edge of fragments[index].edges) {
      if (!partIds.has(edge.source)) {
        throw new Error(`batch-${batchIndex} part ${index + 1}: source ngoài part: ${edge.source}`);
      }
      if (partIds.has(edge.target)) continue;
      if (edge.target.startsWith('file:') && allowedExternalFiles.has(edge.target.slice(5))) continue;
      throw new Error(`batch-${batchIndex} part ${index + 1}: target không xác minh được: ${edge.target}`);
    }
  }

  return {
    batchIndex,
    nodeCount: nodes.length,
    edgeCount: edges.length,
    importCount: actualImports,
    expectedImports,
    partCount: written.length,
    filesSkipped: extraction.filesSkipped ?? [],
    written,
  };
}

const reports = [buildBatch(2), buildBatch(8)];
process.stdout.write(`${JSON.stringify(reports, null, 2)}\n`);
