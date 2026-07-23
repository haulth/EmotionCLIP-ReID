import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const intermediate = path.join(root, '.understand-anything', 'intermediate');
const temp = path.join(root, '.understand-anything', 'tmp');
const data = JSON.parse(fs.readFileSync(path.join(intermediate, 'batches.json'), 'utf8'));
const requested = [2, 4, 6, 8, 11, 18];
const nodeTypes = new Set(['file', 'function', 'class', 'config', 'document', 'service', 'table', 'endpoint', 'pipeline', 'schema', 'resource']);
const complexities = new Set(['simple', 'moderate', 'complex']);
const edgeWeights = {contains: 1.0, imports: 0.7, calls: 0.8, inherits: 0.9, implements: 0.9, exports: 0.8, depends_on: 0.6, tested_by: 0.5, configures: 0.6, documents: 0.5, deploys: 0.7, migrates: 0.7, triggers: 0.6, defines_schema: 0.8, serves: 0.7, provisions: 0.7, routes: 0.6, related: 0.5};
const report = {};
const errors = [];

for (const index of requested) {
  const batch = data.batches.find(item => item.batchIndex === index);
  const extraction = JSON.parse(fs.readFileSync(path.join(temp, `ua-file-extract-results-${index}.json`), 'utf8'));
  const pattern = new RegExp(`^batch-${index}(?:-part-(\\d+))?\\.json$`);
  const files = fs.readdirSync(intermediate).filter(name => pattern.test(name)).sort((a, b) => a.localeCompare(b, undefined, {numeric: true}));
  if (!files.length) errors.push(`batch ${index}: không có output`);
  const fragments = files.map(name => ({name, value: JSON.parse(fs.readFileSync(path.join(intermediate, name), 'utf8'))}));
  const nodes = fragments.flatMap(item => item.value.nodes || []);
  const edges = fragments.flatMap(item => item.value.edges || []);
  const nodeIds = new Set();
  for (const fragment of fragments) {
    if (!Array.isArray(fragment.value.nodes) || !Array.isArray(fragment.value.edges)) errors.push(`${fragment.name}: thiếu nodes/edges array`);
    const partIds = new Set((fragment.value.nodes || []).map(node => node.id));
    for (const edge of fragment.value.edges || []) if (!partIds.has(edge.source)) errors.push(`${fragment.name}: source không thuộc part: ${edge.source}`);
  }
  for (const node of nodes) {
    if (nodeIds.has(node.id)) errors.push(`batch ${index}: duplicate node ${node.id}`);
    nodeIds.add(node.id);
    if (!node.id || !nodeTypes.has(node.type) || !node.name || !node.summary || !Array.isArray(node.tags) || node.tags.length < 3 || !complexities.has(node.complexity)) errors.push(`batch ${index}: node không hợp lệ ${node.id}`);
    if (['file','config','document','service','pipeline','schema','resource'].includes(node.type) && !node.filePath) errors.push(`batch ${index}: file-level node thiếu filePath ${node.id}`);
    if (['function','class'].includes(node.type) && (!Array.isArray(node.lineRange) || node.lineRange.length !== 2)) errors.push(`batch ${index}: symbol thiếu lineRange ${node.id}`);
  }
  const expectedFilePaths = new Set(batch.files.map(file => file.path));
  const actualFilePaths = new Set(nodes.filter(node => ['file','config','document','service','pipeline','schema','resource'].includes(node.type)).map(node => node.filePath));
  for (const filePath of expectedFilePaths) if (!actualFilePaths.has(filePath)) errors.push(`batch ${index}: thiếu file node ${filePath}`);
  for (const filePath of actualFilePaths) if (!expectedFilePaths.has(filePath)) errors.push(`batch ${index}: file node ngoài batch ${filePath}`);

  const expectedSymbols = new Set();
  for (const result of extraction.results) {
    for (const fn of result.functions || []) if (fn.name) expectedSymbols.add(`function:${result.path}:${fn.name}`);
    for (const cls of result.classes || []) if (cls.name) expectedSymbols.add(`class:${result.path}:${cls.name}`);
  }
  for (const id of expectedSymbols) if (!nodeIds.has(id)) errors.push(`batch ${index}: thiếu symbol ${id}`);

  const knownFilePaths = new Set(batch.files.map(file => file.path));
  for (const [source, targets] of Object.entries(batch.batchImportData || {})) {
    knownFilePaths.add(source);
    for (const target of targets) knownFilePaths.add(target);
  }
  for (const [source, neighbors] of Object.entries(batch.neighborMap || {})) {
    knownFilePaths.add(source);
    for (const neighbor of neighbors) knownFilePaths.add(neighbor.path);
  }
  for (const edge of edges) {
    if (!(edge.type in edgeWeights) || edge.direction !== 'forward' || edge.weight !== edgeWeights[edge.type]) errors.push(`batch ${index}: edge schema/weight sai ${JSON.stringify(edge)}`);
    if (!nodeIds.has(edge.source)) {
      const sourcePath = edge.source.startsWith('file:') ? edge.source.slice(5) : null;
      if (!sourcePath || !knownFilePaths.has(sourcePath)) errors.push(`batch ${index}: dangling source ${edge.source}`);
    }
    if (!nodeIds.has(edge.target)) {
      const targetPath = edge.target.startsWith('file:') ? edge.target.slice(5) : null;
      if (!targetPath || !knownFilePaths.has(targetPath)) errors.push(`batch ${index}: dangling target ${edge.target}`);
    }
    if (edge.source === edge.target) errors.push(`batch ${index}: self-edge ${edge.source}`);
  }
  const expectedImports = batch.files.filter(file => file.fileCategory === 'code').reduce((sum, file) => sum + (batch.batchImportData[file.path] || []).length, 0);
  const actualImports = edges.filter(edge => edge.type === 'imports').length;
  if (expectedImports !== actualImports) errors.push(`batch ${index}: imports ${actualImports}/${expectedImports}`);
  report[index] = {outputs: files, nodes: nodes.length, edges: edges.length, imports: `${actualImports}/${expectedImports}`, skipped: extraction.filesSkipped || []};
}

if (errors.length) {
  process.stderr.write(errors.join('\n') + '\n');
  process.exit(1);
}
process.stdout.write(JSON.stringify(report, null, 2) + '\n');
