"""Notebook-safe subprocess progress streaming helpers."""

from __future__ import annotations

import html
import re
from typing import IO, List, Optional

try:
    from IPython.display import HTML, display
except ImportError:  # pragma: no cover - only used outside notebooks
    HTML = None
    display = None


_PROGRESS_RE = re.compile(
    r"^(Stage(?:1(?:-[A-Za-z0-9_]+)?|2)\s+\d+/\d+:).*?\|\s*(\d+)/(\d+)"
)


def _is_progress_update(text: str) -> bool:
    return bool(_PROGRESS_RE.search(text))


def _progress_html(text: str):
    match = _PROGRESS_RE.search(text)
    if HTML is None or match is None:
        return text
    current = int(match.group(2))
    total = max(int(match.group(3)), 1)
    percent = min(100.0, current * 100.0 / total)
    label = html.escape(match.group(1).strip())
    detail = html.escape(text.strip())
    return HTML(
        "<div style='font-family:monospace;max-width:980px'>"
        f"<div style='margin-bottom:4px'>{label} {current}/{total} ({percent:.1f}%)</div>"
        "<div style='height:10px;background:#e5e7eb;border-radius:4px;overflow:hidden'>"
        f"<div style='height:100%;width:{percent:.1f}%;background:#2563eb'></div>"
        "</div>"
        f"<div style='white-space:pre-wrap;margin-top:4px'>{detail}</div>"
        "</div>"
    )


def stream_process_output(process, log_handle: Optional[IO[str]] = None) -> List[str]:
    """Print subprocess output while rendering tqdm carriage-return updates once."""

    progress_display = display(HTML(""), display_id=True) if display is not None and HTML is not None else None
    chunks: List[str] = []
    buffer: List[str] = []

    def emit(text: str, end: str = "\n") -> None:
        print(text, end=end, flush=True)
        if log_handle is not None:
            log_handle.write(text + end)
            log_handle.flush()

    def handle(text: str, end: str = "") -> None:
        if not text:
            return
        chunks.append(text + end)
        if _is_progress_update(text):
            if progress_display is not None:
                progress_display.update(_progress_html(text))
        else:
            emit(text, end=end)

    assert process.stdout is not None
    for char in iter(lambda: process.stdout.read(1), ""):
        if char == "\r":
            handle("".join(buffer), end="\r")
            buffer.clear()
        elif char == "\n":
            handle("".join(buffer), end="\n")
            buffer.clear()
        else:
            buffer.append(char)
    handle("".join(buffer))
    return chunks
