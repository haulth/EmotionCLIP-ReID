import fs from 'node:fs';
import path from 'node:path';

const projectRoot = process.cwd();
const batchPath = path.join(projectRoot, '.understand-anything', 'intermediate', 'batches.json');
const data = JSON.parse(fs.readFileSync(batchPath, 'utf8'));
const wanted = new Set([2, 4, 6, 8, 11, 18]);

for (const batch of data.batches) {
  if (!wanted.has(batch.batchIndex)) continue;
  const payload = {
    projectRoot,
    batchFiles: batch.files,
    batchImportData: batch.batchImportData,
  };
  const outputPath = path.join(projectRoot, '.understand-anything', 'tmp', `ua-file-analyzer-input-${batch.batchIndex}.json`);
  fs.writeFileSync(outputPath, JSON.stringify(payload, null, 2) + '\n');
}
