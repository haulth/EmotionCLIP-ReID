const fs = require('fs');

const inputPath = process.argv[2];
const analysisPath = process.argv[3];
const outputPath = process.argv[4];
if (!inputPath || !analysisPath || !outputPath) {
  console.error('Usage: node ua-arch-assign.js <input.json> <analysis.json> <layers.json>');
  process.exit(1);
}

try {
  const input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
  const analysis = JSON.parse(fs.readFileSync(analysisPath, 'utf8'));
  if (!analysis.scriptCompleted) throw new Error('Structural analysis did not complete.');

  const layers = [
    {
      id: 'layer:entry-orchestration',
      name: 'Điểm Vào và Điều Phối Thực Nghiệm',
      description: 'Các entry point huấn luyện/suy luận, processor điều phối vòng lặp hai giai đoạn và notebook thực thi thí nghiệm EmotionCLIP-ReID.',
      nodeIds: [],
    },
    {
      id: 'layer:model',
      name: 'Mô Hình và Biểu Diễn',
      description: 'CLIP backbone, tokenizer, factory ReID và kiến trúc EmotionCLIP với prompt, adapter, anatomy routing, fusion và reliability head.',
      nodeIds: [],
    },
    {
      id: 'layer:data-pipeline',
      name: 'Dữ Liệu và Tiền Xử Lý',
      description: 'Dataset loaders, manifest cảm xúc, schema landmark/anatomy, augmentation và sampler cho hai nhánh FER và ReID.',
      nodeIds: [],
    },
    {
      id: 'layer:optimization',
      name: 'Tối Ưu và Hàm Mất Mát',
      description: 'Các loss phân loại/metric/evidential, optimizer và learning-rate scheduler dùng trong huấn luyện ReID và EmotionCLIP.',
      nodeIds: [],
    },
    {
      id: 'layer:shared-utility',
      name: 'Tiện Ích Dùng Chung',
      description: 'Logging, metrics FER/ReID, quản lý run artifact, notebook helper và các cross-cutting utility dùng xuyên suốt pipeline.',
      nodeIds: [],
    },
    {
      id: 'layer:data-tooling',
      name: 'Công Cụ Dữ Liệu và Bảo Trì',
      description: 'Các CLI chuyển đổi dataset, tạo/audit face-landmark artifact, tải dữ liệu và script bảo trì tài liệu nghiên cứu.',
      nodeIds: [],
    },
    {
      id: 'layer:configuration',
      name: 'Cấu Hình Thực Nghiệm',
      description: 'YACS defaults, YAML preset cho ReID/FER, môi trường CUDA và cấu hình Pytest kiểm soát hành vi chạy thực nghiệm.',
      nodeIds: [],
    },
    {
      id: 'layer:test',
      name: 'Kiểm Thử và Kiểm Chứng',
      description: 'Bộ Pytest kiểm chứng anatomy, manifest, loss/metrics, model units, processor, run artifacts và notebook safety contracts.',
      nodeIds: [],
    },
    {
      id: 'layer:documentation',
      name: 'Tài Liệu và Tổng Quan Nghiên Cứu',
      description: 'README, ghi chú phát triển, báo cáo, slide, sơ đồ kiến trúc và tài sản tổng quan tài liệu của công trình.',
      nodeIds: [],
    },
    {
      id: 'layer:experimental-artifacts',
      name: 'Kết Quả và Artifact Thực Nghiệm',
      description: 'Metric theo epoch, bảng CSV, uncertainty summaries, log huấn luyện và sơ đồ kết quả đã xuất từ các lần chạy trước.',
      nodeIds: [],
    },
  ];
  const byId = new Map(layers.map((layer) => [layer.id, layer]));
  const assign = (layerId, ids) => byId.get(layerId).nodeIds.push(...ids);
  const groups = analysis.directoryGroups;

  assign('layer:configuration', [...groups.config, ...groups.configs]);
  assign('layer:optimization', [...groups.loss, ...groups.solver]);
  assign('layer:entry-orchestration', [...groups.processor, ...groups.notebooks]);
  assign('layer:shared-utility', groups.utils);
  assign('layer:data-pipeline', groups.datasets);
  assign('layer:test', groups.tests);
  assign('layer:data-tooling', [...groups.tools, ...groups['.codex']]);
  assign('layer:model', groups.model);
  assign('layer:documentation', [...groups.docs, ...groups.fig]);
  assign('layer:experimental-artifacts', groups.outputs);

  const nodeById = new Map(input.fileNodes.map((node) => [node.id, node]));
  for (const id of groups.root) {
    const node = nodeById.get(id);
    if (node.type === 'config') assign('layer:configuration', [id]);
    else if (node.type === 'document') assign('layer:documentation', [id]);
    else assign('layer:entry-orchestration', [id]);
  }

  for (const layer of layers) layer.nodeIds.sort();
  const allAssigned = layers.flatMap((layer) => layer.nodeIds);
  const assignedSet = new Set(allAssigned);
  const inputSet = new Set(input.fileNodes.map((node) => node.id));
  const duplicates = [...assignedSet].filter((id) => allAssigned.filter((x) => x === id).length !== 1);
  const missing = [...inputSet].filter((id) => !assignedSet.has(id));
  const invented = [...assignedSet].filter((id) => !inputSet.has(id));
  if (layers.length < 3 || layers.length > 10) throw new Error(`Invalid layer count: ${layers.length}`);
  if (layers.some((layer) => layer.nodeIds.length === 0)) throw new Error('A layer is empty.');
  if (allAssigned.length !== input.fileNodes.length || duplicates.length || missing.length || invented.length) {
    throw new Error(JSON.stringify({ expected: input.fileNodes.length, assigned: allAssigned.length, duplicates, missing, invented }, null, 2));
  }

  fs.writeFileSync(outputPath, JSON.stringify(layers, null, 2) + '\n', 'utf8');
  console.log(JSON.stringify({
    layerCount: layers.length,
    totalFileNodes: allAssigned.length,
    counts: Object.fromEntries(layers.map((layer) => [layer.id, layer.nodeIds.length])),
  }, null, 2));
} catch (error) {
  console.error(error && error.stack ? error.stack : String(error));
  process.exit(1);
}
