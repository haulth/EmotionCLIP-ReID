from pathlib import Path
from xml.dom import minidom
from xml.etree import ElementTree as ET


OUTPUT = Path(__file__).with_name("emotionclip_anatomy_research_gap_map.drawio")

C = {
    "navy": "#0F2742",
    "slate": "#475569",
    "red": "#B42318",
    "red_fill": "#FFF1F2",
    "green": "#15803D",
    "green_fill": "#DCFCE7",
    "blue": "#2563EB",
    "blue_fill": "#EAF2FF",
    "orange": "#D97706",
    "orange_fill": "#FEF3C7",
    "purple": "#6D28D9",
    "purple_fill": "#F3E8FF",
    "pink": "#DB2777",
    "pink_fill": "#FCE7F3",
    "teal": "#00827C",
    "teal_fill": "#E8F7F4",
    "gray": "#64748B",
    "gray_fill": "#F8FAFC",
}


def box(fill, stroke, size=18, align="left", dashed=False):
    return (
        "rounded=1;arcSize=10;whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth=2.7;"
        f"fontFamily=Arial;fontColor=#111827;fontSize={size};spacing=14;"
        f"align={align};verticalAlign=middle;"
        + ("dashed=1;dashPattern=8 6;" if dashed else "")
    )


def text_style(size=18, color="#111827", bold=False, align="left"):
    return (
        "text;whiteSpace=wrap;html=1;strokeColor=none;fillColor=none;"
        f"fontFamily=Arial;fontColor={color};fontSize={size};"
        f"align={align};verticalAlign=middle;"
        + ("fontStyle=1;" if bold else "")
    )


def edge_style(color, dashed=False):
    return (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
        f"strokeColor={color};strokeWidth=2.8;endArrow=block;endFill=1;"
        "fontFamily=Arial;fontSize=16;fontColor=#111827;labelBackgroundColor=#FFFFFF;"
        + ("dashed=1;dashPattern=8 6;" if dashed else "")
    )


mxfile = ET.Element("mxfile", {"host": "app.diagrams.net", "agent": "Codex", "version": "30.2.6"})
diagram = ET.SubElement(mxfile, "diagram", {"id": "emotionclip-final-plan", "name": "Final Plan Before to After"})
model = ET.SubElement(
    diagram,
    "mxGraphModel",
    {
        "dx": "1800",
        "dy": "1200",
        "grid": "1",
        "gridSize": "10",
        "guides": "1",
        "tooltips": "1",
        "connect": "1",
        "arrows": "1",
        "fold": "1",
        "page": "1",
        "pageScale": "1",
        "pageWidth": "3300",
        "pageHeight": "2520",
        "math": "0",
        "shadow": "0",
    },
)
root = ET.SubElement(model, "root")
ET.SubElement(root, "mxCell", {"id": "0"})
ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})


def vertex(cell_id, value, x, y, w, h, style):
    cell = ET.SubElement(
        root,
        "mxCell",
        {"id": cell_id, "value": value, "style": style, "vertex": "1", "parent": "1"},
    )
    ET.SubElement(
        cell,
        "mxGeometry",
        {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"},
    )


def edge(cell_id, source, target, color, value="", dashed=False):
    cell = ET.SubElement(
        root,
        "mxCell",
        {
            "id": cell_id,
            "value": value,
            "style": edge_style(color, dashed),
            "edge": "1",
            "parent": "1",
            "source": source,
            "target": target,
        },
    )
    ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})


# Header
vertex(
    "title",
    "<b>EmotionCLIP-ReID — Kế hoạch phát triển mô hình phiên bản chốt</b>",
    70,
    25,
    2500,
    55,
    text_style(38, C["navy"], True),
)
vertex(
    "subtitle",
    "Bốn thay đổi có thể kiểm chứng độc lập: data contract → Stage 2 dual stream → Stage 1 geometry prompt → uncertainty/ambiguity outputs",
    75,
    82,
    2700,
    38,
    text_style(20, C["slate"]),
)
vertex(
    "rq",
    "<b>CÂU HỎI NGHIÊN CỨU</b><br>Liệu phân bố hình học cấp lớp có thể cải thiện emotion prompts ở Stage 1, và hình học cấp ảnh có thể hướng dẫn CLIP patch routing ở Stage 2, trong khi mô hình tự bỏ qua landmark không đáng tin do pose, crop, occlusion hoặc detector failure?<br><br><b>Nguyên tắc:</b> landmark vừa là spatial prior, vừa là geometry feature; cả hai chỉ được sử dụng theo reliability từng vùng và luôn có visual-only fallback.",
    75,
    140,
    3150,
    150,
    box(C["purple_fill"], C["purple"], 20, "center"),
)

# Locked foundations
vertex(
    "foundation",
    "<b>FOUNDATION GIỮ CỐ ĐỊNH KHI ABLATE</b>   sealed train/val/test · immutable run/provenance · original CLIP frozen (last_blocks=0 baseline) · branch temperature + simplex fusion · decoupled probability/strength",
    75,
    315,
    3150,
    70,
    box(C["gray_fill"], C["gray"], 17, "center", True),
)

# Headers
vertex("h_scope", "<b>KHỐI CẦN SỬA</b>", 85, 415, 300, 42, text_style(19, C["navy"], True, "center"))
vertex("h_before", "<b>TỪ — HIỆN TẠI</b>", 440, 415, 700, 42, text_style(20, C["red"], True, "center"))
vertex("h_after", "<b>SANG — PHIÊN BẢN CHỐT</b>", 1215, 415, 1120, 42, text_style(20, C["green"], True, "center"))
vertex("h_test", "<b>GIỮ LẠI KHI</b>", 2410, 415, 720, 42, text_style(20, C["teal"], True, "center"))

rows = [
    {
        "id": "geometry",
        "num": "01",
        "name": "DATA CONTRACT<br><b>SHARED GEOMETRY</b>",
        "y": 480,
        "h": 280,
        "color": C["teal"],
        "fill": C["teal_fill"],
        "before": "<b>Không có geometry artifact chuẩn</b><br>Batch chỉ mang image/label; raw coordinates dễ bị dùng sai.<br><br>Missing landmark có nguy cơ bị điền tọa độ giả; flip/crop không bảo đảm đồng bộ landmark; chưa đo detector jitter hay signal-to-jitter.",
        "after": "<b>Một representation dùng chung cho hai stage</b><br>g(x) = [g<sub>upper</sub>, g<sub>middle</sub>, g<sub>lower</sub>]<br><br>UPPER: brow + left/right eyes · MIDDLE: nose + cheeks · LOWER: mouth + chin/jaw.<br>Mỗi điểm: l<sub>i</sub>=(x<sub>i</sub>,y<sub>i</sub>,v<sub>i</sub>,c<sub>i</sub>,u<sub>i</sub>). Mỗi feature: value + valid mask + uncertainty.<br>Encoder nhận [g<sub>r</sub>⊙m<sub>r</sub>, m<sub>r</sub>, u<sub>r</sub>]. Không mirror/fill điểm giả; transform-aware artifact; ưu tiên 3D mesh khi pose lớn.",
        "test": "<b>Geometry audit trên RAF-DB</b><br>• detection/missing rate<br>• pose distribution<br>• augmentation jitter<br>• signal-to-jitter từng feature<br>• visible-side behavior<br><br>q<sub>r</sub>=0 phải tắt anatomy sạch sẽ.<br><font color='#6D28D9'><b>Đối chiếu [R4–R7]</b></font>",
    },
    {
        "id": "stage2",
        "num": "02",
        "name": "STAGE 2<br><b>DUAL STREAM</b>",
        "y": 800,
        "h": 330,
        "color": C["blue"],
        "fill": C["blue_fill"],
        "before": "<b>Một local stream dùng hard Top-K</b><br>Patch chọn thuần theo image–text similarity; không biết vùng, visibility hoặc landmark quality.<br><br>Không có geometry feature mô tả “vùng đó biến dạng thế nào”; landmark failure có thể trở thành single point of failure.",
        "after": "<b>Visual-primary + reliability-gated geometry residual</b><br><b>Visual stream:</b> A<sub>r,p</sub> từ observed landmarks với w<sub>i</sub>=v<sub>i</sub>c<sub>i</sub>exp(−u<sub>i</sub>/τ); F<sub>r,p</sub> là free attention; M*<sub>r,p</sub>=q<sub>r</sub>A<sub>r,p</sub>+(1−q<sub>r</sub>)F<sub>r,p</sub>.<br><b>Geometry stream:</b> h<sup>geom</sup><sub>r</sub>=G<sub>r</sub>([g<sub>r</sub>⊙m<sub>r</sub>,m<sub>r</sub>,u<sub>r</sub>]).<br><b>Gated residual:</b> h<sub>r</sub>=LN[h<sup>vis</sup><sub>r</sub>+γ<sub>r</sub>q<sub>r</sub>tanh(h<sup>geom</sup><sub>r</sub>)].<br>q<sub>r</sub>=0 ⇒ content-only free attention; geometry projector zero-init.",
        "test": "<b>S0→S6 và SF</b><br>S2→S3: fallback có ích?<br>S3→S4: geometry thêm gì ngoài localization?<br>SF→S5: reliability gate có hơn landmark injection luôn bật?<br><br>Giữ khi tốt hơn trên pose/blur/occlusion, worst-class F1 và deletion/insertion.<br><font color='#6D28D9'><b>Đối chiếu [R3–R7]</b></font>",
    },
    {
        "id": "stage1",
        "num": "03",
        "name": "STAGE 1<br><b>CLASS PROMPT</b>",
        "y": 1170,
        "h": 290,
        "color": C["orange"],
        "fill": C["orange_fill"],
        "before": "<b>Class-only prompt học trong một bước</b><br>“A photo of a face showing [X1][X2][X3][X4] expression of {emotion}.”<br><br>Base context có thể hấp thụ mọi tín hiệu, nên khó xác định geometry residual thực sự đóng góp hay chỉ thêm parameters.",
        "after": "<b>4 role tokens + training 1A/1B</b><br>“A photo of a face with [GLOBAL] [UPPER] [MIDDLE] [LOWER] showing a {emotion} expression.”<br><br>μ<sub>y,r</sub>=median(g<sub>r</sub>|y); s<sub>y,r</sub>=1.4826 MAD(g<sub>r</sub>|y), fit train only.<br>C*<sub>y,r</sub>=C<sup>base</sup><sub>y,r</sub>+γ<sub>r</sub>q<sub>y,r</sub>tanh P<sub>r</sub>([μ<sub>y,r</sub>,log(s<sub>y,r</sub>+ε)]).<br><b>1A:</b> học base prompt. <b>1B:</b> freeze base, chỉ học projector/gate. Output vẫn là một fixed text descriptor/class.",
        "test": "<b>P0→P4 + PR/PS</b><br>P1 role tokens<br>P2 + median<br>P3 + MAD<br>P4 + quality gate<br>PR random geometry<br>PS shuffled class geometry<br><br>Chỉ triển khai sau khi S3/S4 chứng minh geometry hữu ích.<br><font color='#6D28D9'><b>Đối chiếu [R1–R4]</b></font>",
    },
    {
        "id": "outputs",
        "num": "04",
        "name": "OUTPUT + LOSS<br><b>RELIABILITY</b>",
        "y": 1500,
        "h": 300,
        "color": C["pink"],
        "fill": C["pink_fill"],
        "before": "<b>Một uncertainty scalar</b><br>EDL cũ suy evidence từ fused-logit scale/offset và không phân biệt class confusion, regional disagreement và input corruption.<br><br>Raw region disagreement có thể tăng giả khi landmark thiếu hoặc ảnh bị che.",
        "after": "<b>Ba output tách biệt + validity</b><br>A<sub>class</sub>=H(p<sub>final</sub>) · A<sub>region</sub>=JSD<sub>q</sub>(p<sub>upper</sub>,p<sub>middle</sub>,p<sub>lower</sub>) · U<sub>ext</sub>=K/S(x).<br><b>region_disagreement_valid=false</b> khi tổng region quality không đủ.<br>Reliability head nhận global visual, mean/min quality, valid-region ratio, pose/crop quality; không nhận A<sub>region</sub> hay raw region logits ở v1.<br><br>L<sub>2</sub>=CE<sub>final</sub>+βL<sub>align</sub>+λ<sub>rel</sub>L<sub>reliability</sub>+λ<sub>inside</sub>L<sub>routing</sub>. Không tối ưu trực tiếp A<sub>region</sub>.",
        "test": "<b>Claim ambiguity chỉ khi</b><br>• A<sub>region</sub> tương quan human label disagreement<br>• corruption chủ yếu tăng U<sub>ext</sub>, không tăng giả A<sub>region</sub><br>• NLL/Brier/E-AURC và error/OOD AUROC cải thiện<br>• common-logit offset invariant<br><br>FERPlus kiểm tra annotator entropy.<br><font color='#6D28D9'><b>Đối chiếu [R8–R11]</b></font>",
    },
]

for row in rows:
    y, h, rid = row["y"], row["h"], row["id"]
    vertex(
        f"{rid}_scope",
        f"<font style='font-size:34px'><b>{row['num']}</b></font><br>{row['name']}",
        85,
        y,
        300,
        h,
        box(row["fill"], row["color"], 18, "center"),
    )
    vertex(f"{rid}_before", row["before"], 440, y, 700, h, box(C["red_fill"], C["red"], 16, "left"))
    vertex(f"{rid}_after", row["after"], 1215, y, 1120, h, box(row["fill"], row["color"], 16, "left"))
    vertex(f"{rid}_test", row["test"], 2410, y, 720, h, box(C["teal_fill"], C["teal"], 16, "left"))
    edge(f"{rid}_delta", f"{rid}_before", f"{rid}_after", row["color"], "SỬA")
    edge(f"{rid}_evidence", f"{rid}_after", f"{rid}_test", C["teal"], "KIỂM CHỨNG")

# Implementation order from the final plan
vertex("roadmap_title", "<b>Thứ tự triển khai — Stage 2 trước, Stage 1 sau</b>", 85, 1845, 1500, 42, text_style(23, C["navy"], True))
roadmap = [
    ("m0", "<b>1 · Geometry audit</b><br>RAF-DB<br>missing · pose · jitter · SJR", 85, C["teal_fill"], C["teal"]),
    ("m1", "<b>2 · Artifact + loader</b><br>mask/uncertainty<br>transform-aware", 520, C["gray_fill"], C["gray"]),
    ("m2", "<b>3 · Stage 2 S1–S3</b><br>free → anatomy<br>→ hybrid routing", 955, C["blue_fill"], C["blue"]),
    ("m3", "<b>4 · Stage 2 S4</b><br>geometry stream<br>gated residual", 1390, C["blue_fill"], C["blue"]),
    ("m4", "<b>5 · Reliability S5–S6</b><br>corruption training<br>valid disagreement", 1825, C["pink_fill"], C["pink"]),
    ("m5", "<b>6 · Stage 1 P1–P4</b><br>chỉ khi geometry<br>đã có causal gain", 2260, C["orange_fill"], C["orange"]),
    ("m6", "<b>7 · Integrate</b><br>≥3 seeds<br>RAF-DB · FER2013 · FERPlus", 2695, C["green_fill"], C["green"]),
]
for pid, label, x, fill, stroke in roadmap:
    vertex(pid, label, x, 1910, 385, 145, box(fill, stroke, 15, "center"))
for i in range(len(roadmap) - 1):
    edge(f"road_{i}", roadmap[i][0], roadmap[i + 1][0], C["navy"])

# References
vertex("refs_title", "<b>Bài báo đối chiếu trực tiếp</b>", 85, 2100, 900, 40, text_style(23, C["navy"], True))
refs = [
    (
        "ref_prompt",
        "<b>PROMPT / VLM FER</b><br><b>[R1]</b> CLIP-ReID: Exploiting Vision-Language Model for Image Re-identification without Concrete Text Labels — AAAI 2023.<br><b>[R2]</b> CEPrompt: Cross-Modal Emotion-Aware Prompting for Facial Expression Recognition — TCSVT 2024.<br><b>[R3]</b> Multimodal Prompt Alignment for Facial Expression Recognition — ICCV 2025.<br><b>[R4]</b> FineCLIPER: Multi-modal Fine-grained CLIP for Dynamic Facial Expression Recognition with AdaptERs — ACM MM 2024.",
        85,
        C["orange_fill"],
        C["orange"],
    ),
    (
        "ref_landmark",
        "<b>LANDMARK / REGION TOKENS</b><br><b>[R4]</b> FineCLIPER — ACM MM 2024.<br><b>[R5]</b> LA-Net: Landmark-Aware Learning for Reliable Facial Expression Recognition under Label Noise — ICCV 2023.<br><b>[R6]</b> Face-LLaVA: Facial Expression and Attribute Understanding through Instruction Tuning — WACV 2026.<br><b>[R7]</b> Face-mask-aware Facial Expression Recognition based on Face Parsing and Vision Transformer — PRL 2022.<br><br><b>Gap:</b> missing-aware geometry + reliability gate + visual fallback.",
        870,
        C["blue_fill"],
        C["blue"],
    ),
    (
        "ref_unc",
        "<b>UNCERTAINTY / AMBIGUITY</b><br><b>[R8]</b> Evidential Deep Learning to Quantify Classification Uncertainty — NeurIPS 2018.<br><b>[R9]</b> Are Uncertainty Quantification Capabilities of Evidential Deep Learning a Mirage? — NeurIPS 2024.<br><b>[R10]</b> On Calibration of Modern Neural Networks — ICML 2017.<br><b>[R11]</b> Rethinking the Ambiguity in Facial Expression Recognition — Pattern Recognition 2026.<br><br><b>Gap:</b> reliable regional disagreement tách khỏi input unreliability.",
        1655,
        C["pink_fill"],
        C["pink"],
    ),
    (
        "ref_claim",
        "<b>CHỐT ĐÓNG GÓP</b><br><br><b>Stage 1:</b> class geometry distribution cập nhật fixed emotion text descriptors.<br><br><b>Stage 2:</b> instance geometry vừa xác định patch cần nhìn, vừa mô tả biến dạng bằng gated residual.<br><br><b>Safety:</b> visibility và uncertainty từng điểm điều khiển geometry; khi thiếu/sai, mô hình trở về content-only visual attention.",
        2440,
        C["purple_fill"],
        C["purple"],
    ),
]
for rid, label, x, fill, stroke in refs:
    vertex(rid, label, x, 2150, 735, 285, box(fill, stroke, 14, "left"))

xml = ET.tostring(mxfile, encoding="utf-8")
OUTPUT.write_bytes(minidom.parseString(xml).toprettyxml(indent="    ", encoding="UTF-8"))
print(OUTPUT)
