from pathlib import Path
from xml.dom import minidom
from xml.etree import ElementTree as ET


OUTPUT = Path(__file__).with_name("emotionclip_anatomy_research_gap_map.drawio")


COLORS = {
    "navy": "#0F2742",
    "slate": "#475569",
    "blue": "#2563EB",
    "blue_fill": "#EAF2FF",
    "green": "#16A34A",
    "green_fill": "#DCFCE7",
    "orange": "#D97706",
    "orange_fill": "#FEF3C7",
    "red": "#B42318",
    "red_fill": "#FFF1F2",
    "purple": "#6D28D9",
    "purple_fill": "#F3E8FF",
    "pink": "#DB2777",
    "pink_fill": "#FCE7F3",
    "teal": "#00827C",
    "teal_fill": "#E8F7F4",
    "gray": "#64748B",
    "gray_fill": "#F8FAFC",
    "white": "#FFFFFF",
}


def style_box(fill, stroke, font_size=18, align="center", dashed=False, bold=False):
    return (
        "rounded=1;arcSize=10;whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth=2.6;"
        "fontFamily=Arial;"
        f"fontColor=#111827;fontSize={font_size};spacing=12;"
        f"align={align};verticalAlign=middle;"
        + ("dashed=1;dashPattern=8 6;" if dashed else "")
        + ("fontStyle=1;" if bold else "")
    )


def style_panel(stroke, fill="#FFFFFF"):
    return (
        "rounded=1;arcSize=12;whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth=3.2;"
        "fontFamily=Arial;fontColor=#0F2742;fontSize=27;spacing=18;"
        "align=left;verticalAlign=top;container=1;pointerEvents=0;"
    )


def style_text(font_size=18, color="#111827", bold=False, align="left"):
    return (
        "text;whiteSpace=wrap;html=1;strokeColor=none;fillColor=none;"
        f"fontFamily=Arial;fontColor={color};fontSize={font_size};"
        f"align={align};verticalAlign=middle;"
        + ("fontStyle=1;" if bold else "")
    )


def style_edge(color, dashed=False, width=2.6):
    return (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;"
        "jettySize=auto;html=1;"
        f"strokeColor={color};strokeWidth={width};"
        "endArrow=block;endFill=1;fontFamily=Arial;fontSize=15;"
        "fontColor=#111827;labelBackgroundColor=#FFFFFF;"
        + ("dashed=1;dashPattern=8 6;" if dashed else "")
    )


mxfile = ET.Element(
    "mxfile",
    {"host": "app.diagrams.net", "agent": "Codex", "version": "30.2.6"},
)
diagram = ET.SubElement(
    mxfile,
    "diagram",
    {"id": "emotionclip-anatomy-gap-map", "name": "Research Gap to Implementation"},
)
model = ET.SubElement(
    diagram,
    "mxGraphModel",
    {
        "dx": "2200",
        "dy": "1350",
        "grid": "1",
        "gridSize": "10",
        "guides": "1",
        "tooltips": "1",
        "connect": "1",
        "arrows": "1",
        "fold": "1",
        "page": "1",
        "pageScale": "1",
        "pageWidth": "3700",
        "pageHeight": "2400",
        "math": "0",
        "shadow": "0",
    },
)
root = ET.SubElement(model, "root")
ET.SubElement(root, "mxCell", {"id": "0"})
ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})


def vertex(cell_id, value, x, y, w, h, style, parent="1"):
    cell = ET.SubElement(
        root,
        "mxCell",
        {
            "id": cell_id,
            "value": value,
            "style": style,
            "vertex": "1",
            "parent": parent,
        },
    )
    ET.SubElement(
        cell,
        "mxGeometry",
        {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"},
    )
    return cell


def edge(cell_id, source, target, color, value="", dashed=False, width=2.6, points=None):
    cell = ET.SubElement(
        root,
        "mxCell",
        {
            "id": cell_id,
            "value": value,
            "style": style_edge(color, dashed=dashed, width=width),
            "edge": "1",
            "parent": "1",
            "source": source,
            "target": target,
        },
    )
    geo = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
    if points:
        arr = ET.SubElement(geo, "Array", {"as": "points"})
        for px, py in points:
            ET.SubElement(arr, "mxPoint", {"x": str(px), "y": str(py)})
    return cell


# Header and legend
vertex(
    "title",
    "<b>EmotionCLIP-ReID Research Gap Map</b>",
    70,
    28,
    2200,
    52,
    style_text(38, COLORS["navy"], bold=True),
)
vertex(
    "subtitle",
    "Từ bằng chứng hiện tại → khoảng trống khoa học → anatomy-conditioned prompting → reliability-aware patch routing → kiểm chứng và triển khai",
    75,
    82,
    2600,
    38,
    style_text(20, COLORS["slate"]),
)

legend = [
    ("L_GAP", "GAP / RISK", COLORS["red"]),
    ("L_DEC", "RESEARCH DECISION", COLORS["purple"]),
    ("L_IMPL", "PROPOSED MODULE", COLORS["blue"]),
    ("L_MET", "LOSS / METRIC", COLORS["pink"]),
    ("L_GATE", "GO / NO-GO", COLORS["teal"]),
]
lx = 2660
for lid, label, color in legend:
    vertex(
        lid,
        f"<b>{label}</b>",
        lx,
        42,
        180,
        30,
        style_box(color, color, 12, bold=True).replace("fontColor=#111827", "fontColor=#FFFFFF").replace("spacing=12", "spacing=2"),
    )
    lx += 192

# Main panels
vertex(
    "gap_panel",
    "<b>1. Bằng chứng hiện tại &amp; research gaps</b><br><font style='font-size:18px;color:#475569'>Những vấn đề phải giải quyết trước khi claim novelty/generalization</font>",
    70,
    150,
    790,
    1690,
    style_panel(COLORS["red"]),
)
vertex(
    "decision_panel",
    "<b>2. Research decision</b><br><font style='font-size:18px;color:#475569'>Đổi câu hỏi từ “landmark có tăng accuracy?” sang phân biệt ambiguity thật và unreliability</font>",
    910,
    150,
    710,
    1690,
    style_panel(COLORS["purple"], "#FCFAFF"),
)
vertex(
    "stage1_panel",
    "<b>3. Stage 1 — Anatomy-conditioned prompting</b><br><font style='font-size:18px;color:#475569'>Class-level robust geometry → role-structured prompt tokens → fixed semantic anchors</font>",
    1670,
    150,
    940,
    1000,
    style_panel(COLORS["orange"], "#FFFDF7"),
)
vertex(
    "stage2_panel",
    "<b>4. Stage 2 — Reliability-aware anatomical routing</b><br><font style='font-size:18px;color:#475569'>Instance landmarks guide CLIP patches; quality controls how strongly anatomy is trusted</font>",
    2660,
    150,
    970,
    1000,
    style_panel(COLORS["blue"], "#F8FBFF"),
)
vertex(
    "outputs_panel",
    "<b>5. Outputs &amp; uncertainty decomposition</b>",
    1670,
    1200,
    940,
    640,
    style_panel(COLORS["pink"], "#FFF9FC"),
)
vertex(
    "eval_panel",
    "<b>6. Validation, ablation &amp; publication evidence</b>",
    2660,
    1200,
    970,
    640,
    style_panel(COLORS["teal"], "#F7FFFD"),
)
vertex(
    "roadmap_panel",
    "<b>7. Triển khai theo causal ladder</b>",
    70,
    1900,
    3560,
    410,
    style_panel(COLORS["navy"], "#F8FAFC"),
)

# Gap evidence boxes
gaps = [
    (
        "g_protocol",
        "<b>G0 — Protocol &amp; provenance</b><br>Split đã được sửa, nhưng metric cũ phải rerun.<br>Run cần Git/config/manifest hash, immutable run_id, 3–5 seeds.",
        270,
        COLORS["red_fill"],
        COLORS["red"],
    ),
    (
        "g_novelty",
        "<b>G1 — Novelty collision</b><br>Two-stage prompt, adapter, global/local alignment và landmark regions đều đã có tiền lệ.<br><b>Không claim “chia vùng mặt” là mới.</b>",
        485,
        COLORS["red_fill"],
        COLORS["red"],
    ),
    (
        "g_unc",
        "<b>G2 — Uncertainty không identifiable</b><br>softplus(fused logits) đổi theo common-logit offset dù softmax không đổi.<br>Cần probability và evidence strength tách biệt.",
        700,
        COLORS["red_fill"],
        COLORS["red"],
    ),
    (
        "g_fusion",
        "<b>G3 — Fusion scale không kiểm soát</b><br>Classifier/global/local khác scale; raw weights có thể âm.<br>Cần branch temperature + simplex gate + telemetry.",
        915,
        COLORS["orange_fill"],
        COLORS["orange"],
    ),
    (
        "g_petl",
        "<b>G4 — Baseline chưa phải adapter-only PETL</b><br>Stage 2 unfreeze last blocks.<br>Baseline chính: original CLIP frozen, last_blocks=0; ablate 0/1/2.",
        1130,
        COLORS["orange_fill"],
        COLORS["orange"],
    ),
    (
        "g_local",
        "<b>G5 — Local Top-K thiếu anatomy &amp; faithfulness</b><br>Hard Top-K có thể chọn tóc/nền và nhạy biên patch.<br>Heatmap đơn thuần chưa chứng minh causal explanation.",
        1345,
        COLORS["red_fill"],
        COLORS["red"],
    ),
    (
        "g_geometry",
        "<b>G6 — Geometry chưa đi vào semantic stage</b><br>Prompt hiện chỉ có 4 latent [X] tokens; chưa biết upper/middle/lower-face geometry, asymmetry hay class variation.",
        1560,
        COLORS["orange_fill"],
        COLORS["orange"],
    ),
]
for gid, text, y, fill, stroke in gaps:
    vertex(gid, text, 115, y, 700, 165, style_box(fill, stroke, 17, align="left"))

# Decision / research framing
vertex(
    "rq",
    "<b>RESEARCH QUESTION</b><br><br>Có thể dùng cấu trúc hình học khuôn mặt để:<br><b>(1)</b> cập nhật emotion semantics ở Stage 1,<br><b>(2)</b> điều hướng visual evidence ở Stage 2,<br>và <b>(3)</b> phân biệt bất đồng cảm xúc có ý nghĩa giữa các vùng đáng tin với bất đồng giả do landmark/ảnh bị lỗi hay không?",
    955,
    275,
    620,
    255,
    style_box(COLORS["purple_fill"], COLORS["purple"], 20, align="left", bold=False),
)
vertex(
    "novelty_no",
    "<b>Không phải novelty</b><br>• crop mắt–mũi–miệng<br>• landmark region tokens<br>• face parsing + ViT<br>• local attention thông thường",
    955,
    585,
    285,
    250,
    style_box(COLORS["red_fill"], COLORS["red"], 17, align="left"),
)
vertex(
    "novelty_yes",
    "<b>Khoảng trống khả thi</b><br>• geometry-to-prompt-to-patch nhất quán<br>• soft routing chịu landmark jitter<br>• quality-aware hybrid anatomy/free attention<br>• conditional regional disagreement",
    1280,
    585,
    295,
    250,
    style_box(COLORS["green_fill"], COLORS["green"], 17, align="left"),
)
vertex(
    "geometry_contract",
    "<b>Shared anatomy representation g(x)</b><br><br>Eye-aligned coordinates; scale bởi inter-ocular distance.<br>6 vùng: brow · eyes · cheeks · nose · mouth · jaw.<br>Robust median/MAD + visibility + detector stability.<br><br><b>Rule:</b> pose/blur/confidence chỉ làm quality gate; không được encode thành emotion semantics.",
    955,
    900,
    620,
    300,
    style_box(COLORS["blue_fill"], COLORS["blue"], 18, align="left"),
)
vertex(
    "related_work",
    "<b>Comparator bắt buộc</b><br>CLIP-ReID (two-stage anchors) · CEPrompt/MPA-FER (prompt + local alignment) · FineCLIPER (landmark/parsing + CLIP) · LA-Net (landmark-aware FER) · Face-LLaVA (region landmark tokens + cross-attention).<br><br><b>Claim phải vượt khỏi:</b> “landmark giúp FER accuracy”.",
    955,
    1260,
    620,
    305,
    style_box(COLORS["gray_fill"], COLORS["gray"], 17, align="left", dashed=True),
)
vertex(
    "claim",
    "<b>Proposed claim</b><br>Robust facial geometry conditions class prompts and provides quality-aware priors for visual patch routing; only disagreement among reliable regions is interpreted as intrinsic ambiguity.",
    955,
    1620,
    620,
    160,
    style_box(COLORS["green_fill"], COLORS["green"], 18, align="left", bold=False),
)

# Stage 1 nodes
vertex(
    "s1_train",
    "<b>Train split only</b><br>landmarks + emotion labels",
    1720,
    275,
    210,
    90,
    style_box(COLORS["gray_fill"], COLORS["gray"], 16),
)
vertex(
    "s1_stats",
    "<b>Robust class geometry</b><br>μ<sub>y,r</sub> = median(g<sub>r</sub>|y)<br>s<sub>y,r</sub> = MAD/IQR<br>q<sub>y,r</sub> = valid/stable rate",
    1970,
    250,
    270,
    140,
    style_box(COLORS["teal_fill"], COLORS["teal"], 16),
)
vertex(
    "s1_prompt_structure",
    "<b>Role-structured prompt</b><br><font style='font-size:15px'>“A photo of a face with<br>[GLOBAL] [UPPER] [MIDDLE] [LOWER]<br>[ASYMMETRY] [VARIATION]<br>showing a {emotion} expression.”</font>",
    2280,
    230,
    285,
    180,
    style_box(COLORS["orange_fill"], COLORS["orange"], 17),
)
vertex(
    "s1_vars",
    "<b>Prompt variables</b><br>UPPER: brow–eye distance, EAR, slope<br>MIDDLE: cheek geometry, nose anchor<br>LOWER: MAR, mouth corners, jaw opening<br>ASYM: left–right differences<br>VAR: class dispersion/quantiles",
    1720,
    465,
    395,
    230,
    style_box("#FFF7ED", COLORS["orange"], 16, align="left"),
)
vertex(
    "s1_equation",
    "<b>Anatomy residual</b><br>C*<sub>y,r</sub> = C<sup>base</sup><sub>y,r</sub> + γ<sub>r</sub> q<sub>y,r</sub> tanh(P<sub>r</sub>[μ<sub>y,r</sub>, log s<sub>y,r</sub>])<br><br>Projector zero-init; γ starts near 0.<br>Numeric coordinates are <b>not</b> converted to text.",
    2160,
    465,
    405,
    230,
    style_box(COLORS["purple_fill"], COLORS["purple"], 16, align="left"),
)
vertex(
    "s1_text_encoder",
    "<b>Frozen CLIP text encoder</b><br>one conditioned descriptor per class",
    1720,
    770,
    260,
    100,
    style_box(COLORS["blue_fill"], COLORS["blue"], 16),
)
vertex(
    "s1_loss",
    "<b>Stage 1 objective</b><br>L<sub>1</sub> = CE(image·text, y)<br>+ prompt-shift regularization<br>+ semantic retention",
    2020,
    750,
    265,
    135,
    style_box(COLORS["pink_fill"], COLORS["pink"], 16),
)
vertex(
    "s1_anchors",
    "<b>Fixed anchors T*</b><br>7 class descriptors<br>reused by Stage 2",
    2325,
    770,
    240,
    100,
    style_box(COLORS["green_fill"], COLORS["green"], 16),
)
vertex(
    "s1_guard",
    "<b>Leakage guard</b><br>Fit median/MAD/quantiles trên train only; lưu landmark model, feature schema và manifest hash.",
    1720,
    950,
    845,
    120,
    style_box(COLORS["red_fill"], COLORS["red"], 16, align="left", dashed=True),
)

# Stage 2 nodes
vertex(
    "s2_input",
    "<b>Face image x</b><br>transform-aware landmarks",
    2710,
    270,
    205,
    90,
    style_box(COLORS["gray_fill"], COLORS["gray"], 16),
)
vertex(
    "s2_clip",
    "<b>Frozen CLIP ViT</b><br>global h<sub>g</sub> + patch tokens v<sub>p</sub><br><font style='font-size:14px'>baseline last_blocks=0</font>",
    2960,
    245,
    250,
    140,
    style_box(COLORS["blue_fill"], COLORS["blue"], 16),
)
vertex(
    "s2_quality",
    "<b>Region quality q<sub>r</sub></b><br>confidence · visibility · pose<br>augmentation stability · occlusion<br>outside-crop fraction",
    3250,
    245,
    325,
    140,
    style_box(COLORS["teal_fill"], COLORS["teal"], 16),
)
vertex(
    "s2_anat",
    "<b>Soft anatomical prior</b><br>M<sup>anat</sup><sub>pr</sub> ∝ exp[-d(p,R<sub>r</sub>)²/σ²<sub>r</sub>]<br>soft membership; 0.5–1.5 patch σ ablation",
    2710,
    465,
    365,
    155,
    style_box(COLORS["orange_fill"], COLORS["orange"], 16),
)
vertex(
    "s2_free",
    "<b>Free learned attention</b><br>M<sup>free</sup><sub>pr</sub> = softmax(query<sub>r</sub>·key<sub>p</sub>)<br>fallback when geometry is unreliable",
    3115,
    465,
    365,
    155,
    style_box(COLORS["purple_fill"], COLORS["purple"], 16),
)
vertex(
    "s2_hybrid",
    "<b>Reliability-aware hybrid routing</b><br>M*<sub>pr</sub> = q<sub>r</sub>M<sup>anat</sup><sub>pr</sub> + (1−q<sub>r</sub>)M<sup>free</sup><sub>pr</sub>",
    2800,
    685,
    585,
    115,
    style_box(COLORS["green_fill"], COLORS["green"], 17),
)
vertex(
    "s2_regions",
    "<b>Region features &amp; distributions</b><br>h<sub>r</sub> = Σ<sub>p</sub>M*<sub>pr</sub>v<sub>p</sub><br>p<sup>(r)</sup> = softmax(W<sub>shared</sub>h<sub>r</sub>)",
    2710,
    870,
    315,
    140,
    style_box(COLORS["blue_fill"], COLORS["blue"], 16),
)
vertex(
    "s2_fusion",
    "<b>Controlled global/local fusion</b><br>branch temperatures + simplex gate<br>Σw=1, w≥0; log gate &amp; logit scale",
    3065,
    870,
    300,
    140,
    style_box(COLORS["orange_fill"], COLORS["orange"], 16),
)
vertex(
    "s2_logits",
    "<b>Final logits z</b><br>global + reliable regions",
    3400,
    885,
    180,
    110,
    style_box(COLORS["green_fill"], COLORS["green"], 16),
)

# Output / uncertainty decomposition
vertex(
    "o_class",
    "<b>A<sub>class</sub> — Class ambiguity</b><br>H[p(y|x)]<br>Các emotion classes cạnh tranh đến mức nào?",
    1720,
    1320,
    260,
    160,
    style_box(COLORS["blue_fill"], COLORS["blue"], 16),
)
vertex(
    "o_region",
    "<b>A<sub>region</sub> — Reliable disagreement</b><br>quality-weighted Jensen–Shannon<br>Các vùng đáng tin có bất đồng không?",
    2010,
    1320,
    265,
    160,
    style_box(COLORS["purple_fill"], COLORS["purple"], 16),
)
vertex(
    "o_ext",
    "<b>U<sub>ext</sub> — Extrinsic unreliability</b><br>U = K/S(x)<br>S = K + softplus(g(h,q))",
    2305,
    1320,
    255,
    160,
    style_box(COLORS["pink_fill"], COLORS["pink"], 16),
)
vertex(
    "o_dirichlet",
    "<b>Offset-invariant evidence</b><br>p = softmax(z)<br>α<sub>k</sub> = 1 + p<sub>k</sub>[S−K]<br>Probability dùng cho decision; strength dùng cho reliability.",
    1720,
    1540,
    385,
    180,
    style_box(COLORS["teal_fill"], COLORS["teal"], 16, align="left"),
)
vertex(
    "o_rule",
    "<b>Interpretation rule</b><br>A<sub>region</sub> cao + U<sub>ext</sub> thấp → ambiguity có bằng chứng.<br>A<sub>region</sub> cao + U<sub>ext</sub> cao → không kết luận; có thể do blur/occlusion/landmark failure.<br><b>Không cộng ba đại lượng thành một scalar ở v1.</b>",
    2145,
    1540,
    415,
    180,
    style_box(COLORS["red_fill"], COLORS["red"], 16, align="left"),
)

# Evaluation
vertex(
    "e_primary",
    "<b>Primary FER evidence</b><br>Macro-F1 · UAR · worst-class F1<br>3–5 seeds · 95% CI · paired bootstrap",
    2710,
    1320,
    270,
    145,
    style_box(COLORS["green_fill"], COLORS["green"], 16),
)
vertex(
    "e_rel",
    "<b>Reliability evidence</b><br>NLL · Brier · E-AURC<br>risk@coverage · error AUROC<br>OOD AUROC/AUPR/FPR95",
    3010,
    1320,
    260,
    145,
    style_box(COLORS["pink_fill"], COLORS["pink"], 16),
)
vertex(
    "e_robust",
    "<b>Robustness &amp; faithfulness</b><br>blur · occlusion · low-res · pose<br>cross-dataset FER<br>deletion/insertion · landmark perturbation",
    3300,
    1320,
    280,
    145,
    style_box(COLORS["blue_fill"], COLORS["blue"], 16),
)
vertex(
    "e_ablation",
    "<b>Causal ablation ladder</b><br>N0 static prompt + Top-K<br>N1 geometry prompt only<br>N2 hard regions<br>N3 soft anatomical routing<br>N5 quality-aware hybrid<br>N6 region-aware reliability",
    2710,
    1535,
    300,
    210,
    style_box(COLORS["gray_fill"], COLORS["gray"], 16, align="left"),
)
vertex(
    "e_eff",
    "<b>PETL/compute report</b><br>last_blocks=0/1/2<br>trainable params · forward GFLOPs<br>peak VRAM · images/s · wall-time<br>in-domain + cross-dataset",
    3045,
    1535,
    260,
    210,
    style_box(COLORS["orange_fill"], COLORS["orange"], 16, align="left"),
)
vertex(
    "e_gate",
    "<b>GO / NO-GO</b><br>• N1 phải vượt role-token/random-stat controls<br>• N3 phải vượt hard routing<br>• N5/N6 phải cải thiện E-AURC, corruption và cross-dataset<br>• không đổi novelty chỉ vì in-domain accuracy tăng",
    3340,
    1535,
    240,
    210,
    style_box(COLORS["teal_fill"], COLORS["teal"], 15, align="left", bold=False),
)

# Roadmap cards
roadmap = [
    (
        "r0",
        "<b>P0 — Evidence hygiene</b><br>sealed test · immutable run<br>full provenance · rerun baselines",
        125,
        COLORS["red_fill"],
        COLORS["red"],
    ),
    (
        "r1",
        "<b>P1 — Geometry audit</b><br>extract landmarks · eye alignment<br>median/MAD · jitter/SJR · train-only stats",
        695,
        COLORS["blue_fill"],
        COLORS["blue"],
    ),
    (
        "r2",
        "<b>P2 — Prompt baseline</b><br>P0/P1/P2 role tokens<br>N0→N1 geometry-conditioned prompt",
        1265,
        COLORS["orange_fill"],
        COLORS["orange"],
    ),
    (
        "r3",
        "<b>P3 — Routing ladder</b><br>hard region → soft anatomy<br>→ quality-aware hybrid",
        1835,
        COLORS["purple_fill"],
        COLORS["purple"],
    ),
    (
        "r4",
        "<b>P4 — Reliability</b><br>decoupled strength<br>three outputs · corruption/OOD",
        2405,
        COLORS["pink_fill"],
        COLORS["pink"],
    ),
    (
        "r5",
        "<b>P5 — Publication test</b><br>3–5 seeds · cross-dataset<br>faithfulness · compute Pareto",
        2975,
        COLORS["green_fill"],
        COLORS["green"],
    ),
]
for rid, text, x, fill, stroke in roadmap:
    vertex(rid, text, x, 2040, 520, 170, style_box(fill, stroke, 17, align="left"))

# Edges: gap → decision
for index, gid in enumerate(["g_protocol", "g_novelty", "g_unc", "g_fusion", "g_petl", "g_local", "g_geometry"]):
    edge(f"eg{index}", gid, "rq", COLORS["red"] if index in {0, 1, 2, 5} else COLORS["orange"], dashed=index == 0, width=2.0)

# Decision flow
edge("ed1", "rq", "geometry_contract", COLORS["purple"], "operationalize")
edge("ed2", "novelty_no", "novelty_yes", COLORS["green"], "move beyond", dashed=True)
edge("ed3", "geometry_contract", "claim", COLORS["green"], "shared contract")

# Decision to stage modules
edge("ed_s1", "rq", "s1_train", COLORS["purple"], "semantic stage", points=[(1635, 360)])
edge("ed_s2", "rq", "s2_input", COLORS["purple"], "visual stage", points=[(1635, 430), (2640, 430)])

# Stage 1 flow
edge("es11", "s1_train", "s1_stats", COLORS["teal"])
edge("es12", "s1_stats", "s1_prompt_structure", COLORS["orange"])
edge("es13", "s1_vars", "s1_equation", COLORS["orange"])
edge("es14", "s1_prompt_structure", "s1_equation", COLORS["purple"])
edge("es15", "s1_equation", "s1_text_encoder", COLORS["blue"], points=[(2360, 725), (1870, 725)])
edge("es16", "s1_text_encoder", "s1_loss", COLORS["pink"])
edge("es17", "s1_loss", "s1_anchors", COLORS["green"])

# Stage 1 to Stage 2
edge("es1s2", "s1_anchors", "s2_fusion", COLORS["green"], "fixed semantic anchors", points=[(2600, 830), (3040, 830)])

# Stage 2 flow
edge("es21", "s2_input", "s2_clip", COLORS["blue"])
edge("es22", "s2_input", "s2_quality", COLORS["teal"], points=[(2935, 405), (3410, 405)])
edge("es23", "s2_clip", "s2_anat", COLORS["orange"], points=[(3070, 420), (2890, 420)])
edge("es24", "s2_clip", "s2_free", COLORS["purple"], points=[(3090, 430), (3290, 430)])
edge("es25", "s2_quality", "s2_hybrid", COLORS["teal"], "q_r")
edge("es26", "s2_anat", "s2_hybrid", COLORS["orange"])
edge("es27", "s2_free", "s2_hybrid", COLORS["purple"])
edge("es28", "s2_hybrid", "s2_regions", COLORS["blue"])
edge("es29", "s2_regions", "s2_fusion", COLORS["orange"])
edge("es210", "s2_fusion", "s2_logits", COLORS["green"])

# Output flow
edge("eo1", "s2_logits", "o_class", COLORS["blue"], points=[(3650, 1120), (1650, 1120), (1650, 1400)])
edge("eo2", "s2_regions", "o_region", COLORS["purple"], points=[(2660, 1060), (2660, 1170), (2140, 1170)])
edge("eo3", "s2_quality", "o_ext", COLORS["pink"], points=[(3600, 420), (3650, 420), (3650, 1170), (2430, 1170)])
edge("eo4", "o_class", "o_dirichlet", COLORS["teal"])
edge("eo5", "o_region", "o_rule", COLORS["purple"])
edge("eo6", "o_ext", "o_rule", COLORS["pink"])

# Evaluation flow
edge("ee1", "o_class", "e_primary", COLORS["green"], points=[(2650, 1280)])
edge("ee2", "o_ext", "e_rel", COLORS["pink"], points=[(2630, 1495), (3140, 1495)])
edge("ee3", "o_rule", "e_robust", COLORS["blue"], points=[(2620, 1770), (3620, 1770), (3620, 1280)])
edge("ee4", "e_ablation", "e_gate", COLORS["teal"], "decision")
edge("ee5", "e_eff", "e_gate", COLORS["teal"])

# Roadmap flow
for i in range(5):
    edge(f"er{i}", f"r{i}", f"r{i+1}", COLORS["navy"], width=2.8)


xml_bytes = ET.tostring(mxfile, encoding="utf-8")
pretty = minidom.parseString(xml_bytes).toprettyxml(indent="    ", encoding="UTF-8")
OUTPUT.write_bytes(pretty)
print(OUTPUT)
