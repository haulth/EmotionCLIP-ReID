import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const repoRoot = "E:/Source/EmotionCLIP-ReID";
const inputPath = path.join(repoRoot, "docs/research_review/emotionclip_reid_papers.xlsx");
const scratchDir = path.join(repoRoot, ".codex/tmp/019f8a29-798f-7c12-a8ee-fdf1d43a6942");
const mode = process.argv[2] || "inspect";

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));

if (mode === "inspect") {
  const overview = await workbook.inspect({
    kind: "workbook,sheet,table",
    maxChars: 12000,
    tableMaxRows: 12,
    tableMaxCols: 24,
    tableMaxCellChars: 240,
  });
  console.log("OVERVIEW");
  console.log(overview.ndjson);

  const sheets = await workbook.inspect({ kind: "sheet", include: "id,name", maxChars: 4000 });
  console.log("SHEETS");
  console.log(sheets.ndjson);

  await fs.mkdir(scratchDir, { recursive: true });
  for (const sheet of workbook.worksheets.items) {
    const used = sheet.getUsedRange();
    if (!used) continue;
    const preview = await workbook.render({
      sheetName: sheet.name,
      autoCrop: "all",
      scale: 1,
      format: "png",
    });
    const safeName = sheet.name.replace(/[^a-zA-Z0-9_-]+/g, "_");
    await fs.writeFile(path.join(scratchDir, `before_${safeName}.png`), new Uint8Array(await preview.arrayBuffer()));
  }
} else if (mode === "detail") {
  const papers = workbook.worksheets.getItem("32 Papers");
  const log = workbook.worksheets.getItem("Research Log");
  console.log("PAPERS_BOTTOM_VALUES");
  console.log(JSON.stringify(papers.getRange("A77:Q88").values, null, 2));
  console.log("PAPERS_BOTTOM_FORMULAS");
  console.log(JSON.stringify(papers.getRange("A77:Q88").formulas, null, 2));
  console.log("PAPERS_BOTTOM_STYLE");
  console.log((await workbook.inspect({ kind: "computedStyle", sheetId: "32 Papers", range: "A86:Q88", maxChars: 8000 })).ndjson);
  console.log("RESEARCH_LOG_BOTTOM");
  console.log(JSON.stringify(log.getRange("A33:B42").values, null, 2));
  console.log("TABLES");
  for (const sheet of workbook.worksheets.items) {
    console.log(sheet.name, sheet.tables.items.map((table) => ({ name: table.name, style: table.style })));
  }
} else if (mode === "edit") {
  const papers = workbook.worksheets.getItem("32 Papers");
  const log = workbook.worksheets.getItem("Research Log");
  const title = "Universal Language Model Fine-tuning for Text Classification";
  const previousTitle = "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution";
  const normalize = (value) => String(value ?? "")
    .normalize("NFKD")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();

  const usedValues = papers.getUsedRange().values;
  const existingIndexes = usedValues
    .map((row, index) => ({ index, title: normalize(row[1]) }))
    .filter((item) => item.title === normalize(title))
    .map((item) => item.index);
  if (existingIndexes.length > 1) {
    throw new Error(`Duplicate title already exists ${existingIndexes.length} times.`);
  }

  const maxNumber = Math.max(...usedValues.slice(1).map((row) => Number(row[0]) || 0));
  const isNew = existingIndexes.length === 0;
  const targetExcelRow = isNew ? usedValues.length + 1 : existingIndexes[0] + 1;
  const paperNumber = isNew ? maxNumber + 1 : Number(usedValues[existingIndexes[0]][0]);
  if (paperNumber !== 89) {
    throw new Error(`Expected paper No. 89 but resolved No. ${paperNumber}.`);
  }

  const paperRow = [[
    paperNumber,
    title,
    "Jeremy Howard; Sebastian Ruder",
    2018,
    "Conference",
    "ACL 2018 (Long Papers)",
    "Transfer learning / gradual unfreezing / discriminative fine-tuning",
    "Đề xuất ULMFiT, một quy trình transfer learning phổ quát từ language model pretrained sang tác vụ phân loại văn bản, đồng thời giảm overfitting và catastrophic forgetting khi fine-tune.",
    "Ba giai đoạn rõ ràng; discriminative learning rate theo tầng, slanted triangular learning rate và gradual unfreezing có ablation; hiệu quả dữ liệu cao và không cần kiến trúc riêng theo tác vụ.",
    "Được kiểm chứng trên AWD-LSTM cho NLP, không phải Transformer/VLM hay dữ liệu khuôn mặt; chưa đánh giá OOD/cross-dataset FER và lịch learning rate cần điều chỉnh cho CLIP dual encoder.",
    "WikiText-103 (LM pretraining); TREC-6; IMDb; Yelp-bi; Yelp-full; AG News; DBpedia.",
    "N/A trực tiếp cho FER; giảm error 18-24% trên phần lớn 6 bộ text classification và với 100 nhãn đạt mức của training from scratch dùng nhiều dữ liệu hơn tới 100 lần.",
    "General-domain LM pretraining → target-task LM fine-tuning → classifier fine-tuning; discriminative layer-wise LR, STLR, gradual unfreezing, concat pooling và BPT3C.",
    "https://aclanthology.org/P18-1031/",
    "ACL Anthology / DOI 10.18653/v1/P18-1031",
    "Nguyên lý tối ưu hóa áp dụng trực tiếp cho EmotionCLIP-ReID: huấn luyện theo giai đoạn, LR theo tầng và gradual unfreezing để hạn chế catastrophic forgetting/feature distortion khi thích nghi CLIP.",
    "2026-07-22",
  ]];

  const targetRange = papers.getRange(`A${targetExcelRow}:Q${targetExcelRow}`);
  if (isNew) {
    const previousRange = papers.getRange(`A${targetExcelRow - 1}:Q${targetExcelRow - 1}`);
    targetRange.copyFrom(previousRange, "all");
    targetRange.clear({ applyTo: "contents" });
    targetRange.format.rowHeight = previousRange.format.rowHeight;
  }
  targetRange.values = paperRow;
  targetRange.format.wrapText = true;
  targetRange.format.verticalAlignment = "top";
  targetRange.format.rowHeight = 108;

  log.getRange("B1").values = [["2026-07-22"]];
  log.getRange("B3").values = [["2 new papers, No. 88-89"]];
  log.getRange("B6").values = [["Fine-tuning strategy; linear probing; LP-FT; discriminative fine-tuning; STLR; gradual unfreezing; OOD robustness and pretrained-feature preservation."]];

  const logValues = log.getUsedRange().values;
  const logMarker = "2026-07-22 update";
  let logStart = logValues.findIndex((row) => row[0] === logMarker) + 1;
  if (logStart === 0) {
    logStart = logValues.length + 2;
    const sourceBlock = log.getRange("A38:B42");
    const destinationBlock = log.getRange(`A${logStart}:B${logStart + 4}`);
    destinationBlock.copyFrom(sourceBlock, "all");
    destinationBlock.clear({ applyTo: "contents" });
    for (let i = 0; i < 5; i += 1) {
      log.getRange(`A${logStart + i}:B${logStart + i}`).format.rowHeight =
        log.getRange(`A${38 + i}:B${38 + i}`).format.rowHeight;
    }
  }
  log.getRange(`A${logStart}:B${logStart + 4}`).values = [
    [logMarker, "Added 2 unique papers as No. 88-89; the main sheet now contains 89 papers."],
    ["Latest additions", `${previousTitle}; ${title}`],
    ["Main clusters added", "Fine-tuning strategy; linear probing; LP-FT; discriminative fine-tuning; STLR; gradual unfreezing; OOD robustness and pretrained-feature preservation."],
    ["Quality-source rule", "Used the supplied arXiv/ICLR metadata and ACL Anthology/DOI record; no mirror-only source."],
    ["Duplicate control", "Exact-title and normalized-title checks before insertion; no duplicate rows added."],
  ];

  const outputDir = path.join(repoRoot, "outputs/019f8a29-798f-7c12-a8ee-fdf1d43a6942");
  const outputPath = path.join(outputDir, "emotionclip_reid_papers.xlsx");
  await fs.mkdir(outputDir, { recursive: true });
  const exported = await SpreadsheetFile.exportXlsx(workbook);
  await exported.save(outputPath);
  await fs.copyFile(outputPath, inputPath);

  const verified = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
  const verifiedPapers = verified.worksheets.getItem("32 Papers");
  const verifiedLog = verified.worksheets.getItem("Research Log");
  console.log("VERIFY_PAPER");
  console.log((await verified.inspect({
    kind: "table",
    range: "32 Papers!A89:Q90",
    include: "values,formulas",
    tableMaxRows: 4,
    tableMaxCols: 17,
    tableMaxCellChars: 500,
    maxChars: 14000,
  })).ndjson);
  console.log("VERIFY_LOG");
  console.log((await verified.inspect({
    kind: "table",
    range: `Research Log!A${logStart}:B${logStart + 4}`,
    include: "values,formulas",
    tableMaxRows: 8,
    tableMaxCols: 2,
    tableMaxCellChars: 500,
    maxChars: 8000,
  })).ndjson);
  console.log("VERIFY_COUNTS");
  console.log(JSON.stringify({
    paperRows: verifiedPapers.getUsedRange().values.length,
    paperRecords: verifiedPapers.getUsedRange().values.length - 1,
    logRows: verifiedLog.getUsedRange().values.length,
    titleMatches: verifiedPapers.getUsedRange().values.filter((row) => normalize(row[1]) === normalize(title)).length,
  }));
  console.log("FORMULA_ERRORS");
  console.log((await verified.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 300 },
    summary: "final formula error scan",
    maxChars: 4000,
  })).ndjson);

  await fs.mkdir(scratchDir, { recursive: true });
  for (const sheet of verified.worksheets.items) {
    const used = sheet.getUsedRange();
    if (!used) continue;
    const preview = await verified.render({
      sheetName: sheet.name,
      autoCrop: "all",
      scale: 1,
      format: "png",
    });
    const safeName = sheet.name.replace(/[^a-zA-Z0-9_-]+/g, "_");
    await fs.writeFile(path.join(scratchDir, `after_${safeName}.png`), new Uint8Array(await preview.arrayBuffer()));
  }
  console.log(`OUTPUT=${outputPath}`);
} else if (mode === "render-focus") {
  const papers = workbook.worksheets.getItem("32 Papers");
  console.log(JSON.stringify({
    row88Height: papers.getRange("A88:Q88").format.rowHeight,
    row89Height: papers.getRange("A89:Q89").format.rowHeight,
    row90Height: papers.getRange("A90:Q90").format.rowHeight,
  }));
  console.log((await workbook.inspect({ kind: "computedStyle", sheetId: "32 Papers", range: "A90:Q90", maxChars: 10000 })).ndjson);
  const preview = await workbook.render({
    sheetName: "32 Papers",
    range: "A88:Q90",
    scale: 2,
    format: "png",
  });
  await fs.writeFile(path.join(scratchDir, "focus_rows_88_90.png"), new Uint8Array(await preview.arrayBuffer()));
}
