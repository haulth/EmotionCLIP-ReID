#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

const root = process.argv[2];
if (!root) {
  process.stderr.write('Usage: node ua-assemble-final.cjs <project-root>\n');
  process.exit(1);
}

const intermediate = path.join(root, '.understand-anything', 'intermediate');
const graphPath = path.join(intermediate, 'assembled-graph.json');
const layersPath = path.join(intermediate, 'layers.json');
const tourPath = path.join(intermediate, 'tour.json');
const scanPath = path.join(intermediate, 'scan-result.json');

const base = JSON.parse(fs.readFileSync(graphPath, 'utf8'));
const layers = JSON.parse(fs.readFileSync(layersPath, 'utf8'));
const tour = JSON.parse(fs.readFileSync(tourPath, 'utf8'));
const scan = JSON.parse(fs.readFileSync(scanPath, 'utf8'));
const nodeIds = new Set(base.nodes.map((node) => node.id));

if (!Array.isArray(layers) || !Array.isArray(tour)) {
  throw new Error('layers and tour must both be arrays');
}
for (const layer of layers) {
  for (const key of ['id', 'name', 'description', 'nodeIds']) {
    if (!(key in layer)) throw new Error(`Layer missing required field ${key}`);
  }
  if (!Array.isArray(layer.nodeIds)) throw new Error(`Layer ${layer.id} nodeIds must be an array`);
  for (const id of layer.nodeIds) {
    if (!nodeIds.has(id)) throw new Error(`Layer ${layer.id} references missing node ${id}`);
  }
}
for (const step of tour) {
  for (const key of ['order', 'title', 'description', 'nodeIds']) {
    if (!(key in step)) throw new Error(`Tour step missing required field ${key}`);
  }
  if (!Array.isArray(step.nodeIds)) throw new Error(`Tour step ${step.order} nodeIds must be an array`);
  for (const id of step.nodeIds) {
    if (!nodeIds.has(id)) throw new Error(`Tour step ${step.order} references missing node ${id}`);
  }
  if ('languageLesson' in step && typeof step.languageLesson !== 'string') {
    throw new Error(`Tour step ${step.order} languageLesson must be a string`);
  }
}

const fullGraph = {
  version: '1.0.0',
  project: {
    name: scan.name || 'EmotionCLIP-ReID',
    languages: scan.languages || [],
    frameworks: scan.frameworks || [],
    description: scan.description || 'Two-stage CLIP-based facial expression recognition with anatomy-aware routing and reliability estimation.',
    analyzedAt: new Date().toISOString(),
    gitCommitHash: 'cffe8839e00543557ba7ec3114e22a419de41c52'
  },
  nodes: base.nodes,
  edges: base.edges,
  layers,
  tour
};

fs.writeFileSync(graphPath, JSON.stringify(fullGraph, null, 2));
process.stdout.write(`Assembled final graph: ${fullGraph.nodes.length} nodes, ${fullGraph.edges.length} edges, ${layers.length} layers, ${tour.length} tour steps\n`);
