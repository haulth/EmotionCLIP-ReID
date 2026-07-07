from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "docs" / "report" / "fig"
OUT_DRAWIO = FIG_DIR / "emotionclip_reid_proposal_pipeline.drawio"

PAGE_W = 3200
PAGE_H = 1850

NAVY = "10243E"
INK = "111827"
SLATE = "475569"
MUTED = "64748B"
LINE = "CBD5E1"
WHITE = "FFFFFF"

BLUE = "2563EB"
BLUE_LIGHT = "EAF2FF"
TEAL = "00827C"
TEAL_DARK = "006B66"
TEAL_LIGHT = "E8F7F4"
GREEN = "16A34A"
GREEN_LIGHT = "DCFCE7"
AMBER = "D97706"
AMBER_LIGHT = "FEF3C7"
PINK = "DB2777"
PINK_LIGHT = "FCE7F3"
GRAY = "475569"
GRAY_LIGHT = "F1F5F9"
VIOLET = "6D28D9"
VIOLET_LIGHT = "F3E8FF"
RED = "B42318"
RED_LIGHT = "FFF1F2"


def esc(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def html(value: str) -> str:
    return esc(value.replace("\n", "<br>"))


def cell(
    cid: str,
    value: str,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    fill: str = WHITE,
    stroke: str = LINE,
    font_color: str = INK,
    font_size: int = 24,
    stroke_width: float = 2.4,
    rounded: bool = True,
    align: str = "center",
    valign: str = "middle",
    spacing: int = 12,
    extra: str = "",
) -> str:
    rounded_style = "rounded=1;arcSize=10;" if rounded else "rounded=0;"
    style = (
        f"{rounded_style}whiteSpace=wrap;html=1;fillColor=#{fill};strokeColor=#{stroke};"
        f"strokeWidth={stroke_width};fontFamily=Arial;fontColor=#{font_color};fontSize={font_size};"
        f"spacing={spacing};align={align};verticalAlign={valign};{extra}"
    )
    return (
        f'<mxCell id="{cid}" value="{html(value)}" style="{style}" vertex="1" parent="1">'
        f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
    )


def label(
    cid: str,
    value: str,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    color: str = INK,
    size: int = 24,
    bold: bool = False,
    align: str = "left",
) -> str:
    text = f"<b>{value}</b>" if bold else value
    style = (
        f"text;whiteSpace=wrap;html=1;strokeColor=none;fillColor=none;fontFamily=Arial;"
        f"fontColor=#{color};fontSize={size};align={align};verticalAlign=middle;"
    )
    return (
        f'<mxCell id="{cid}" value="{html(text)}" style="{style}" vertex="1" parent="1">'
        f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
    )


def box(
    cid: str,
    title: str,
    body: str,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    fill: str,
    stroke: str,
    tag: tuple[str, str] | None = None,
    note: str = "",
    font_size: int = 23,
    align: str = "center",
) -> list[str]:
    value = f"<b>{title}</b><br>{body}"
    if note:
        value += f"<br><font style='font-size: 15px; color: #{MUTED};'>{note}</font>"
    cells = [
        cell(
            cid,
            value,
            x,
            y,
            w,
            h,
            fill=fill,
            stroke=stroke,
            font_size=font_size,
            stroke_width=3,
            align=align,
            spacing=16,
        )
    ]
    if tag:
        tag_text, tag_color = tag
        cells.append(
            cell(
                f"{cid}_tag",
                f"<b>{tag_text}</b>",
                x + w - 118,
                y + 16,
                96,
                30,
                fill=tag_color,
                stroke=tag_color,
                font_color=WHITE,
                font_size=17,
                stroke_width=1,
                spacing=3,
            )
        )
    return cells


def edge(
    cid: str,
    source: str,
    target: str,
    color: str,
    *,
    width: float = 3,
    dashed: bool = False,
    points: list[tuple[int, int]] | None = None,
) -> str:
    dashed_style = "dashed=1;dashPattern=8 8;" if dashed else ""
    style = (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
        f"strokeColor=#{color};strokeWidth={width};endArrow=block;endFill=1;{dashed_style}"
    )
    if points:
        pts = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in points)
        geom = f'<mxGeometry relative="1" as="geometry"><Array as="points">{pts}</Array></mxGeometry>'
    else:
        geom = '<mxGeometry relative="1" as="geometry"/>'
    return f'<mxCell id="{cid}" value="" style="{style}" edge="1" parent="1" source="{source}" target="{target}">{geom}</mxCell>'


def line(cid: str, x1: int, y1: int, x2: int, y2: int, color: str = LINE, width: float = 2) -> str:
    style = f"endArrow=none;html=1;rounded=0;strokeColor=#{color};strokeWidth={width};"
    return (
        f'<mxCell id="{cid}" value="" style="{style}" edge="1" parent="1">'
        f'<mxGeometry width="50" height="50" relative="1" as="geometry">'
        f'<mxPoint x="{x1}" y="{y1}" as="sourcePoint"/><mxPoint x="{x2}" y="{y2}" as="targetPoint"/>'
        f"</mxGeometry></mxCell>"
    )


def stage_header(cid: str, vi: str, en: str, x: int, y: int, w: int, color: str) -> str:
    return cell(
        cid,
        f"<b>{vi}</b><br><font style='font-size: 26px;'>({en})</font>",
        x,
        y,
        w,
        92,
        fill=WHITE,
        stroke=WHITE,
        font_color=INK,
        font_size=34,
        stroke_width=0,
        spacing=4,
    )


def build_cells() -> list[str]:
    c: list[str] = []
    c.append(label("title", "EmotionCLIP-ReID: phương pháp hai giai đoạn và đóng góp kiểm chứng", 70, 35, 2500, 70, color=NAVY, size=48, bold=True))
    c.append(label("subtitle", "Một chuyển đổi có kiểm soát từ CLIP-ReID sang FER: học emotion descriptors, cố định chúng làm ứng viên semantic anchors, rồi kiểm chứng bằng fine-tuning và ablation.", 75, 105, 2530, 50, color=SLATE, size=24))
    c.append(cell("claim_guard", "<b>Nguyên tắc công bố:</b> sơ đồ chỉ nêu giả thuyết/cơ chế; đóng góp chỉ được khẳng định khi có ablation và metric tương ứng.", 75, 155, 2420, 35, fill=GRAY_LIGHT, stroke=LINE, font_color=INK, font_size=18, stroke_width=1.3, spacing=3))
    for i, (text, color) in enumerate([("FROZEN/REUSE", BLUE), ("MODIFIED", AMBER), ("NEW", GREEN), ("EVIDENCE", PINK), ("OPTIONAL", VIOLET)]):
        c.append(cell(f"legend_{i}", f"<b>{text}</b>", 2475 + i * 138, 58, 125, 30, fill=color, stroke=color, font_color=WHITE, font_size=14, stroke_width=1, spacing=2))

    # Main stage panels.
    c.append(cell("stage1_panel", "", 60, 210, 1500, 1130, fill=WHITE, stroke=BLUE, stroke_width=3))
    c.append(cell("stage2_panel", "", 1640, 210, 1500, 1130, fill=WHITE, stroke=TEAL, stroke_width=3))
    c.append(line("divider", 1600, 230, 1600, 1325, INK, 2.5))
    c.append(stage_header("stage1_header", "Giai đoạn 1: Học descriptor cảm xúc", "learn candidate emotion descriptors T", 150, 230, 1320, BLUE))
    c.append(stage_header("stage2_header", "Giai đoạn 2: Kiểm chứng anchor bằng fine-tuning", "adapter fine-tuning + global/local alignment + calibration", 1700, 230, 1370, TEAL))

    # Contribution markers used inside the method.
    marker_style = "ellipse;"
    for cid, text, x, y, color in [
        ("c1a", "C1", 92, 370, AMBER),
        ("c2a", "C2", 458, 650, GREEN),
        ("c2b", "C2", 1190, 905, GREEN),
        ("c3a", "C3", 2238, 320, GREEN),
        ("c4a", "C4", 2425, 595, GREEN),
        ("c5a", "C5", 2792, 875, GREEN),
    ]:
        c.append(cell(cid, f"<b>{text}</b>", x, y, 48, 48, fill=color, stroke=color, font_color=WHITE, font_size=18, stroke_width=1, extra=marker_style, spacing=2))

    # Stage 1: visual branch, descriptor branch, optimization.
    c += box("s1_images", "FER batch", "face image x<br>emotion label y", 135, 375, 220, 120, fill=GRAY_LIGHT, stroke=GRAY, tag=("MOD", AMBER), font_size=20)
    c += box("s1_img_encoder", "Frozen CLIP image encoder", "ViT-B/16<br>no gradient update", 420, 335, 305, 190, fill=BLUE_LIGHT, stroke=BLUE, tag=("REUSE", BLUE), font_size=22)
    c += box("s1_img_feat", "Image feature", "z_v = f_img(x)", 800, 385, 230, 95, fill=GRAY_LIGHT, stroke=GRAY, font_size=22)
    c += box("s1_labels", "Emotion classes", "7-class FER labels<br>anger ... neutral", 135, 690, 255, 125, fill=TEAL_LIGHT, stroke=TEAL, tag=("MOD", AMBER), font_size=20)
    c += box("s1_au_optional", "AU prior", "optional only; use when AU/pseudo-AU is reliable", 135, 870, 300, 105, fill=VIOLET_LIGHT, stroke=VIOLET, tag=("OPT", VIOLET), font_size=19)
    c += box("s1_prompt", "Emotion Prompt Learner", "template + learnable [X] tokens<br>class-specific context", 500, 655, 430, 165, fill=GREEN_LIGHT, stroke=GREEN, tag=("NEW", GREEN), note="updates only prompt tokens", font_size=21)
    c += box("s1_text_encoder", "Frozen CLIP text encoder", "text transformer + projection", 1005, 660, 310, 140, fill=BLUE_LIGHT, stroke=BLUE, tag=("REUSE", BLUE), font_size=21)
    c += box("s1_descriptors", "Candidate descriptors T", "T = {z_t^c}<br>one vector per emotion", 1175, 905, 310, 135, fill=AMBER_LIGHT, stroke=AMBER, tag=("NEW", GREEN), font_size=21)
    c += box("s1_loss", "Descriptor learning objective", "L_s1 = CE(τ · z_v T^T, y)<br>image/text encoders frozen", 690, 525, 540, 120, fill=WHITE, stroke=INK, tag=("MOD", AMBER), font_size=21)
    c += box("s1_output", "Stage-1 output", "fixed candidate descriptors T<br>checkpoint for Stage 2", 995, 1110, 430, 120, fill=AMBER_LIGHT, stroke=AMBER, tag=("FIXED", BLUE), font_size=21)
    c += box("s1_validation", "Stage-1 evidence required", "descriptor ablation: class-name prompt vs learned T; retrieval/UMAP; confusion and per-class F1.", 140, 1155, 760, 120, fill=PINK_LIGHT, stroke=PINK, tag=("EVID", PINK), font_size=20)

    c.append(edge("e_s1_img_enc", "s1_images", "s1_img_encoder", BLUE, width=3.2))
    c.append(edge("e_s1_img_feat", "s1_img_encoder", "s1_img_feat", BLUE, width=3.2))
    c.append(edge("e_s1_feat_loss", "s1_img_feat", "s1_loss", GRAY, width=3.0))
    c.append(edge("e_s1_labels_prompt", "s1_labels", "s1_prompt", TEAL, width=3.0))
    c.append(edge("e_s1_au_prompt", "s1_au_optional", "s1_prompt", VIOLET, width=2.4, dashed=True))
    c.append(edge("e_s1_prompt_text", "s1_prompt", "s1_text_encoder", GREEN, width=3.2))
    c.append(edge("e_s1_text_loss", "s1_text_encoder", "s1_loss", GRAY, width=3.0))
    c.append(edge("e_s1_text_desc", "s1_text_encoder", "s1_descriptors", AMBER, width=3.2))
    c.append(edge("e_s1_loss_prompt", "s1_loss", "s1_prompt", AMBER, width=2.6, dashed=True, points=[(690, 735)]))
    c.append(edge("e_s1_desc_output", "s1_descriptors", "s1_output", AMBER, width=3.2))

    # Stage 2: visual adaptation, anchoring, decision, calibration.
    c += box("s2_images", "FER train/val batch", "face image x<br>emotion label y", 1695, 380, 225, 120, fill=GRAY_LIGHT, stroke=GRAY, tag=("MOD", AMBER), font_size=20)
    c += box("s2_preproc", "Face-safe preprocessing", "resize/center-fit<br>light augmentation", 1960, 365, 255, 145, fill=BLUE_LIGHT, stroke=BLUE, tag=("NEW", GREEN), font_size=20)
    c += box("s2_encoder", "CLIP ViT + ExpressionAdapter", "adapters in transformer blocks<br>train adapters + selected last blocks", 2265, 330, 430, 205, fill=GREEN_LIGHT, stroke=GREEN, tag=("NEW", GREEN), note="parameter-efficient adaptation", font_size=21)
    c += box("s2_visual", "Visual features", "global z_g<br>patch tokens P", 2760, 385, 305, 120, fill=AMBER_LIGHT, stroke=AMBER, tag=("MOD", AMBER), font_size=21)
    c += box("s2_fixed_t", "Fixed T from Stage 1", "candidate semantic anchors<br>no gradient update", 1695, 905, 320, 145, fill=AMBER_LIGHT, stroke=AMBER, tag=("FIXED", BLUE), font_size=21)
    c += box("s2_cls", "Classifier branch", "s_cls = W z_g", 2095, 625, 270, 95, fill=WHITE, stroke=GRAY, tag=("MOD", AMBER), font_size=21)
    c += box("s2_global", "Global image-text alignment", "s_g = τ · z_g T^T", 2425, 610, 320, 105, fill=TEAL_LIGHT, stroke=TEAL, tag=("NEW", GREEN), font_size=21)
    c += box("s2_local", "Local patch-text alignment", "s_l = τ · mean TopK(P T^T)<br>captures small facial regions", 2425, 765, 320, 135, fill=TEAL_LIGHT, stroke=TEAL, tag=("NEW", GREEN), font_size=20)
    c += box("s2_fusion", "Fused logits", "s = w_cls s_cls + w_g s_g + w_l s_l", 2810, 635, 285, 145, fill=AMBER_LIGHT, stroke=AMBER, tag=("MOD", AMBER), font_size=20)
    c += box("s2_evidence", "Evidential head", "e = softplus(s), α=e+1<br>p_c=α_c/Σα, u=K/Σα", 2810, 900, 285, 165, fill=PINK_LIGHT, stroke=PINK, tag=("NEW", GREEN), font_size=20)
    c += box("s2_loss", "Training objective", "L = CE(s,y) + β CE(s_g+s_l,y) + λ L_unc<br>L_unc: Dirichlet NLL + annealed KL", 2075, 955, 560, 155, fill=WHITE, stroke=INK, tag=("MOD", AMBER), font_size=20)
    c += box("s2_eval", "Evaluation protocol", "macro-F1, balanced acc, per-class F1<br>ECE, uncertainty-risk AUC<br>report ablation per contribution", 2080, 1165, 820, 120, fill=PINK_LIGHT, stroke=PINK, tag=("EVID", PINK), font_size=20)
    c += box("s2_guard", "Interpretation guardrail", "Uncertainty is a calibration signal, not proof of robustness. Robustness requires subset/ablation evidence.", 1700, 1250, 500, 70, fill=RED_LIGHT, stroke=RED, tag=("CHECK", RED), font_size=18)

    c.append(edge("e_s2_img_pre", "s2_images", "s2_preproc", BLUE, width=3.2))
    c.append(edge("e_s2_pre_enc", "s2_preproc", "s2_encoder", GREEN, width=3.2))
    c.append(edge("e_s2_enc_vis", "s2_encoder", "s2_visual", AMBER, width=3.2))
    c.append(edge("e_s2_vis_cls", "s2_visual", "s2_cls", GRAY, width=2.8, points=[(2910, 575), (2230, 575)]))
    c.append(edge("e_s2_vis_global", "s2_visual", "s2_global", TEAL, width=3.0))
    c.append(edge("e_s2_vis_local", "s2_visual", "s2_local", TEAL, width=3.0, points=[(2910, 750)]))
    c.append(edge("e_s2_t_global", "s2_fixed_t", "s2_global", AMBER, width=2.8, points=[(2240, 975), (2240, 660)]))
    c.append(edge("e_s2_t_local", "s2_fixed_t", "s2_local", AMBER, width=2.8, points=[(2280, 1010), (2280, 830)]))
    c.append(edge("e_s2_cls_fusion", "s2_cls", "s2_fusion", AMBER, width=2.8))
    c.append(edge("e_s2_global_fusion", "s2_global", "s2_fusion", AMBER, width=2.8))
    c.append(edge("e_s2_local_fusion", "s2_local", "s2_fusion", AMBER, width=2.8))
    c.append(edge("e_s2_fusion_evidence", "s2_fusion", "s2_evidence", PINK, width=3.0))
    c.append(edge("e_s2_fusion_loss", "s2_fusion", "s2_loss", GRAY, width=2.8, points=[(2940, 840), (2355, 840)]))
    c.append(edge("e_s2_evid_loss", "s2_evidence", "s2_loss", PINK, width=2.8))
    c.append(edge("e_s2_loss_eval", "s2_loss", "s2_eval", GRAY, width=2.8))
    c.append(edge("e_s2_evid_eval", "s2_evidence", "s2_eval", PINK, width=2.8))
    c.append(edge("cross_stage_T", "s1_output", "s2_fixed_t", AMBER, width=4, points=[(1560, 1170), (1640, 970)]))

    # Contribution matrix.
    c.append(cell("contrib_panel", "", 60, 1390, 3080, 380, fill=GRAY_LIGHT, stroke=LINE, stroke_width=2.4))
    c.append(label("contrib_title", "Đóng góp khoa học và bằng chứng bắt buộc", 105, 1410, 1280, 45, color=NAVY, size=32, bold=True))
    contribs = [
        ("C1", "ReID -> FER protocol", "Manifest, face-safe preprocessing, FER metrics.", "So với CLIP-ReID/ReID metrics.", AMBER),
        ("C2", "Emotion descriptor anchoring", "Learn T in Stage 1, freeze T in Stage 2.", "Class prompt, no-fixed-T, no-I2T.", GREEN),
        ("C3", "Expression-aware adaptation", "Adapters + selected last ViT blocks.", "No-adapter; trainable params; val gap.", GREEN),
        ("C4", "Global-local semantic alignment", "z_g T^T + TopK(P T^T).", "No-local; no-global; per-class F1.", GREEN),
        ("C5", "Evidential calibration", "p(y|x), confidence, uncertainty u(x).", "No-unc; ECE; risk AUC.", GREEN),
    ]
    for i, (code, title, mechanism, evidence, color) in enumerate(contribs):
        x = 105 + i * 595
        c.append(cell(f"contrib_marker_{code}", f"<b>{code}</b>", x, 1470, 56, 56, fill=color, stroke=color, font_color=WHITE, font_size=18, stroke_width=1, extra="ellipse;", spacing=2))
        c += box(
            f"contrib_{code}",
            title,
            f"<b>Cơ chế:</b> {mechanism}<br><b>Kiểm chứng:</b> {evidence}",
            x + 72,
            1460,
            500,
            155,
            fill=WHITE,
            stroke=color,
            font_size=18,
            align="left",
        )
    c.append(cell("publication_note", "<b>Không tuyên bố quá mức:</b> AU là optional; descriptor semantics và calibration chỉ là đóng góp khi ablation/metric ủng hộ.", 105, 1640, 2920, 80, fill=WHITE, stroke=RED, font_color=INK, font_size=20, stroke_width=2.5, spacing=10))

    return c


def write_drawio() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"                {item}" for item in build_cells())
    xml = f"""<mxfile host="app.diagrams.net" agent="Codex" version="24.7.17">
    <diagram id="emotionclip-reid-scientific-pipeline" name="Scientific Pipeline">
        <mxGraphModel dx="1800" dy="1050" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{PAGE_W}" pageHeight="{PAGE_H}" math="0" shadow="0">
            <root>
                <mxCell id="0"/>
                <mxCell id="1" parent="0"/>
{body}
            </root>
        </mxGraphModel>
    </diagram>
</mxfile>
"""
    OUT_DRAWIO.write_text(xml, encoding="utf-8")
    print(OUT_DRAWIO)


if __name__ == "__main__":
    write_drawio()
