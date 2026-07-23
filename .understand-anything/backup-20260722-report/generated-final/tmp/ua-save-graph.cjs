#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

const root = process.argv[2];
if (!root) {
  process.stderr.write('Usage: node ua-save-graph.cjs <project-root>\n');
  process.exit(1);
}

const uaDir = path.join(root, '.understand-anything');
const intermediate = path.join(uaDir, 'intermediate');
const assembledPath = path.join(intermediate, 'assembled-graph.json');
const graphPath = path.join(uaDir, 'knowledge-graph.json');
const scanPath = path.join(intermediate, 'scan-result.json');
const fingerprintInputPath = path.join(intermediate, 'fingerprint-input.json');
const gitCommitHash = 'cffe8839e00543557ba7ec3114e22a419de41c52';

const graph = JSON.parse(fs.readFileSync(assembledPath, 'utf8'));
const scan = JSON.parse(fs.readFileSync(scanPath, 'utf8'));
const sourceFilePaths = scan.files.map((file) => file.path).filter(Boolean);
if (sourceFilePaths.length !== scan.totalFiles) {
  throw new Error(`Expected ${scan.totalFiles} scan paths, found ${sourceFilePaths.length}`);
}

fs.writeFileSync(graphPath, JSON.stringify(graph, null, 2));
fs.writeFileSync(
  fingerprintInputPath,
  JSON.stringify({ projectRoot: root, sourceFilePaths, gitCommitHash }, null, 2)
);
process.stdout.write(`Saved graph and fingerprint input for ${sourceFilePaths.length} files\n`);
