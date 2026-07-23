#!/usr/bin/env node
const fs = require('fs');

const [root, commit, analyzedAt] = process.argv.slice(2);
const base = `${root}/.understand-anything/intermediate`;
const graphPath = `${base}/assembled-graph.json`;
const scan = JSON.parse(fs.readFileSync(`${base}/scan-result.json`, 'utf8'));
const graph = JSON.parse(fs.readFileSync(graphPath, 'utf8'));
const nodeIds = new Set((graph.nodes || []).map(node => node.id));
const knownPrefixes = ['file:', 'config:', 'document:', 'service:', 'pipeline:', 'table:', 'schema:', 'resource:', 'endpoint:'];

let layers = JSON.parse(fs.readFileSync(`${base}/layers.json`, 'utf8'));
if (!Array.isArray(layers)) layers = Array.isArray(layers.layers) ? layers.layers : [];
layers = layers.map((layer, index) => ({
  id: String(layer.id || `layer:${index + 1}`),
  name: String(layer.name || `Lớp ${index + 1}`),
  description: String(layer.description || 'Chưa có mô tả.'),
  nodeIds: [...new Set((Array.isArray(layer.nodeIds) ? layer.nodeIds : [])
    .map(String)
    .map(id => knownPrefixes.some(prefix => id.startsWith(prefix)) ? id : `file:${id}`)
    .filter(id => nodeIds.has(id)))]
}));

let tour = JSON.parse(fs.readFileSync(`${base}/tour.json`, 'utf8'));
if (!Array.isArray(tour)) tour = Array.isArray(tour.steps) ? tour.steps : [];
tour = tour.map((step, index) => {
  const rawIds = step.nodeIds || step.nodesToInspect || [];
  const normalized = {
    order: Number.isFinite(Number(step.order)) ? Number(step.order) : index + 1,
    title: String(step.title || `Bước ${index + 1}`),
    description: String(step.description || step.whyItMatters || 'Chưa có mô tả.'),
    nodeIds: [...new Set((Array.isArray(rawIds) ? rawIds : [])
      .map(String)
      .map(id => knownPrefixes.some(prefix => id.startsWith(prefix)) ? id : `file:${id}`)
      .filter(id => nodeIds.has(id)))]
  };
  if (typeof step.languageLesson === 'string') normalized.languageLesson = step.languageLesson;
  return normalized;
}).sort((a, b) => a.order - b.order);

const finalGraph = {
  version: '1.0.0',
  project: {
    name: String(scan.name || 'EmotionCLIP-ReID'),
    languages: Array.isArray(scan.languages) ? scan.languages : [],
    frameworks: Array.isArray(scan.frameworks) ? scan.frameworks : [],
    description: String(scan.description || ''),
    analyzedAt,
    gitCommitHash: commit
  },
  nodes: Array.isArray(graph.nodes) ? graph.nodes : [],
  edges: Array.isArray(graph.edges) ? graph.edges : [],
  layers,
  tour
};

fs.writeFileSync(graphPath, JSON.stringify(finalGraph, null, 2));
process.stdout.write(`Assembled ${finalGraph.nodes.length} nodes, ${finalGraph.edges.length} edges, ${layers.length} layers, ${tour.length} tour steps\n`);
