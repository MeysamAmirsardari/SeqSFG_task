"""What the participant sees: one full-screen canvas and one form page, nothing else.

Design rules: a calm dark background (no glare in a booth), two large boxes for the two
sequences (left = first, right = second, matching the keys), the box lights up exactly while its
sequence plays, one short line of text at a time, a thin progress bar, and feedback as a symbol
plus a word (never colour alone). All sizes scale with the screen height.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen

C = {
    "bg": "#0E1116", "panel": "#171C23", "edge": "#2A313B", "active": "#18355F",
    "active_edge": "#5B9BFF", "text": "#E9EDF2", "muted": "#8F9AA8", "accent": "#5B9BFF",
    "good": "#35C27F", "bad": "#F0784A", "warn": "#F2B84B", "high": "#5B9BFF", "low": "#33C3B0",
}


def col(name: str, alpha: int = 255) -> QColor:
    c = QColor(C.get(name, name))
    c.setAlpha(alpha)
    return c


@dataclass
class View:
    kind: str = "blank"                  # blank | message | trial | rest | end
    title: str = ""
    body: str = ""
    footer: str = ""
    figure: str = ""
    data: dict = field(default_factory=dict)
    label: str = ""
    progress: Optional[float] = None
    panels: List[str] = field(default_factory=lambda: ["idle", "idle"])  # idle|active|good|bad
    captions: List[str] = field(default_factory=lambda: ["", ""])
    prompt: str = ""
    prompt_style: str = "muted"          # muted | text | good | bad | warn
    overlay: Optional[Tuple[str, str, str]] = None   # (style, title, body)


def _font(px: float, weight=QFont.Weight.Normal) -> QFont:
    f = QFont()
    f.setPixelSize(max(9, int(round(px))))
    f.setWeight(weight)
    return f


class Stage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = View()
        self.setAutoFillBackground(False)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def show_view(self, v: View) -> None:
        self.view = v
        self.update()

    # -- painting ------------------------------------------------------------------------------
    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        w, h = self.width(), self.height()
        u = h / 100.0
        p.fillRect(self.rect(), col("bg"))
        v = self.view
        if v.kind in ("message", "rest", "end"):
            self._message(p, w, h, u, v)
        elif v.kind == "trial":
            self._trial(p, w, h, u, v)
        if v.progress is not None:
            self._progress(p, w, h, u, v.progress)
        if v.overlay:
            self._overlay(p, w, h, u, *v.overlay)
        p.end()

    def _text(self, p, rect: QRectF, text: str, px: float, color: str,
              weight=QFont.Weight.Normal, align=Qt.AlignmentFlag.AlignLeft) -> float:
        p.setFont(_font(px, weight))
        p.setPen(col(color))
        flags = int(align | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap)
        br = p.boundingRect(rect, flags, text)
        p.drawText(rect, flags, text)
        return br.height()

    def _message(self, p, w, h, u, v: View) -> None:
        cw = min(w * 0.62, 118 * u)
        x0 = (w - cw) / 2.0
        y = h * 0.16
        if v.title:
            y += self._text(p, QRectF(x0, y, cw, 20 * u), v.title, 4.4 * u, "text",
                            QFont.Weight.Medium) + 3.0 * u
        if v.body:
            y += self._text(p, QRectF(x0, y, cw, 45 * u), v.body, 2.6 * u, "text") + 4.0 * u
        if v.figure:
            self._figure(p, QRectF(x0, y, cw, min(26 * u, h * 0.80 - y)), u, v.figure, v.data)
        if v.footer:
            self._text(p, QRectF(0, h * 0.86, w, 5 * u), v.footer, 2.1 * u, "muted",
                       align=Qt.AlignmentFlag.AlignHCenter)

    # -- the two boxes -------------------------------------------------------------------------
    def panel_rects(self, w, h, u) -> List[QRectF]:
        pw, ph, gap = min(w * 0.27, 46 * u), 34 * u, min(w * 0.07, 12 * u)
        y = h * 0.47 - ph / 2.0
        x1 = w / 2.0 - gap / 2.0 - pw
        return [QRectF(x1, y, pw, ph), QRectF(w / 2.0 + gap / 2.0, y, pw, ph)]

    def _trial(self, p, w, h, u, v: View) -> None:
        if v.label:
            self._text(p, QRectF(4 * u, 3.5 * u, w * 0.5, 4 * u), v.label, 1.9 * u, "muted")
        rects = self.panel_rects(w, h, u)
        hints = ["←  or  1", "2  or  →"]
        for i, (r, state) in enumerate(zip(rects, v.panels)):
            fill, edge, width = col("panel"), col("edge"), 0.25 * u
            if state == "active":
                fill, edge, width = col("active"), col("active_edge"), 0.45 * u
                halo = QRectF(r.adjusted(-1.2 * u, -1.2 * u, 1.2 * u, 1.2 * u))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(col("active_edge", 46))
                p.drawRoundedRect(halo, 3.2 * u, 3.2 * u)
            elif state in ("good", "bad"):
                edge, width = col(state), 0.45 * u
                fill = col(state, 38)
            p.setPen(QPen(edge, width))
            p.setBrush(fill)
            p.drawRoundedRect(r, 2.4 * u, 2.4 * u)
            num_color = "text" if state != "idle" else "muted"
            self._text(p, QRectF(r.x(), r.y() + r.height() * 0.17, r.width(), 16 * u), str(i + 1),
                       13 * u, num_color, QFont.Weight.Medium, Qt.AlignmentFlag.AlignHCenter)
            self._text(p, QRectF(r.x(), r.y() + r.height() * 0.74, r.width(), 5 * u), hints[i],
                       2.2 * u, "muted", align=Qt.AlignmentFlag.AlignHCenter)
            if v.captions[i]:
                self._text(p, QRectF(r.x() - 4 * u, r.bottom() + 2.2 * u, r.width() + 8 * u, 5 * u),
                           v.captions[i], 2.4 * u, "accent", QFont.Weight.Medium,
                           Qt.AlignmentFlag.AlignHCenter)
        if v.prompt:
            y = rects[0].bottom() + 8.5 * u
            self._text(p, QRectF(0, y, w, 6 * u), v.prompt, 3.0 * u, v.prompt_style,
                       QFont.Weight.Medium if v.prompt_style != "muted" else QFont.Weight.Normal,
                       Qt.AlignmentFlag.AlignHCenter)

    def _progress(self, p, w, h, u, frac: float) -> None:
        bw, bh = min(w * 0.36, 60 * u), 0.55 * u
        r = QRectF((w - bw) / 2.0, h - 5.5 * u, bw, bh)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(col("edge"))
        p.drawRoundedRect(r, bh / 2, bh / 2)
        f = max(0.0, min(1.0, float(frac)))
        if f > 0:
            p.setBrush(col("accent", 200))
            p.drawRoundedRect(QRectF(r.x(), r.y(), bw * f, bh), bh / 2, bh / 2)

    def _overlay(self, p, w, h, u, style: str, title: str, body: str) -> None:
        p.fillRect(self.rect(), QColor(0, 0, 0, 185))
        cw, ch = min(w * 0.5, 80 * u), 30 * u
        r = QRectF((w - cw) / 2.0, (h - ch) / 2.0, cw, ch)
        p.setPen(QPen(col("warn" if style == "alert" else "edge"), 0.3 * u))
        p.setBrush(col("panel"))
        p.drawRoundedRect(r, 2.4 * u, 2.4 * u)
        inner = r.adjusted(4 * u, 4 * u, -4 * u, -3 * u)
        th = self._text(p, QRectF(inner.x(), inner.y(), inner.width(), 8 * u), title, 3.6 * u,
                        "warn" if style == "alert" else "text", QFont.Weight.Medium)
        self._text(p, QRectF(inner.x(), inner.y() + th + 2.5 * u, inner.width(), 18 * u), body,
                   2.6 * u, "text")

    # -- illustrations ------------------------------------------------------------------------
    def _figure(self, p, r: QRectF, u: float, name: str, data: dict) -> None:
        if name == "sequences":
            self._fig_sequences(p, r, u)
        elif name == "keys":
            self._fig_keys(p, r, u)
        elif name == "blocks":
            self._fig_blocks(p, r, u, int(data.get("blocks", 6)), int(data.get("done", 0)))

    def _fig_sequences(self, p, r: QRectF, u: float) -> None:
        colw = (r.width() - 6 * u) / 2.0
        for j, (label, off) in enumerate((("1", 0.0), ("2", 1.0))):
            x0 = r.x() + j * (colw + 6 * u)
            box = QRectF(x0, r.y(), colw, r.height() * 0.66)
            p.setPen(QPen(col("edge"), 0.2 * u))
            p.setBrush(col("panel"))
            p.drawRoundedRect(box, 1.6 * u, 1.6 * u)
            n, pad = 6, 3 * u
            slot = (box.width() - 2 * pad) / (n + 0.6)
            bar = slot * 0.5
            yh, yl = box.y() + box.height() * 0.34, box.y() + box.height() * 0.64
            for k in range(n):
                xb = box.x() + pad + k * slot
                xa = xb
                if k == n - 1 and off:
                    xb -= bar * 0.34
                    xa += bar * 0.34
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(col("high"))
                p.drawRoundedRect(QRectF(xb, yh, bar, 1.3 * u), 0.5 * u, 0.5 * u)
                p.setBrush(col("low"))
                p.drawRoundedRect(QRectF(xa, yl, bar, 1.3 * u), 0.5 * u, 0.5 * u)
            ex = box.x() + pad + (n - 1) * slot - 0.9 * u
            p.setPen(QPen(col("accent"), 0.22 * u, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(QRectF(ex, yh - 1.6 * u, bar + 1.8 * u, yl - yh + 4.4 * u), u, u)
            self._text(p, QRectF(box.x() + 1.6 * u, box.y() + 1.2 * u, 6 * u, 4 * u), label,
                       2.4 * u, "muted", QFont.Weight.Medium)
            cap = "ends together" if not off else "ends out of sync"
            self._text(p, QRectF(x0, box.bottom() + 1.4 * u, colw, 4 * u), cap, 2.2 * u,
                       "accent" if off else "muted", align=Qt.AlignmentFlag.AlignHCenter)

    def _keycap(self, p, x, y, u, text, wide=1.0) -> QRectF:
        r = QRectF(x, y, 7.5 * u * wide, 7.5 * u)
        p.setPen(QPen(col("muted"), 0.25 * u))
        p.setBrush(col("panel"))
        p.drawRoundedRect(r, 1.3 * u, 1.3 * u)
        self._text(p, QRectF(r.x(), r.y() + 1.6 * u, r.width(), 5 * u), text, 3.2 * u, "text",
                   QFont.Weight.Medium, Qt.AlignmentFlag.AlignHCenter)
        return r

    def _fig_keys(self, p, r: QRectF, u: float) -> None:
        items = [("←", "1", "first sequence"), ("→", "2", "second sequence"),
                 ("P", None, "pause")]
        x = r.x() + 2 * u
        for big, alt, label in items:
            k = self._keycap(p, x, r.y() + 2 * u, u, big)
            x2 = k.right() + 1.2 * u
            if alt:
                self._text(p, QRectF(x2, r.y() + 3.8 * u, 4 * u, 4 * u), "or", 2.0 * u, "muted")
                k = self._keycap(p, x2 + 3.4 * u, r.y() + 2 * u, u, alt)
            self._text(p, QRectF(x, r.y() + 11 * u, k.right() - x + 2 * u, 4 * u), label,
                       2.1 * u, "muted")
            x = k.right() + 7 * u

    def _fig_blocks(self, p, r: QRectF, u: float, n: int, done: int) -> None:
        d, gap = 3.2 * u, 2.2 * u
        total = n * d + (n - 1) * gap
        x = r.x() + (r.width() - total) / 2.0
        y = r.y() + 3 * u
        for i in range(n):
            c = QRectF(x + i * (d + gap), y, d, d)
            if i < done:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(col("accent"))
            elif i == done:
                p.setPen(QPen(col("accent"), 0.35 * u))
                p.setBrush(Qt.BrushStyle.NoBrush)
            else:
                p.setPen(QPen(col("edge"), 0.35 * u))
                p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(c)


# ----------------------------------------------------------------------------------------------
# the form page (questionnaire and debrief)
# ----------------------------------------------------------------------------------------------
FORM_CSS = f"""
QWidget#form {{ background: {C['bg']}; }}
QLabel {{ color: {C['text']}; }}
QLabel[role="title"] {{ font-weight: 500; }}
QLabel[role="muted"] {{ color: {C['muted']}; }}
QLabel[role="error"] {{ color: {C['bad']}; }}
QLineEdit, QSpinBox, QComboBox, QPlainTextEdit {{
  background: {C['panel']}; color: {C['text']}; border: 1px solid {C['edge']};
  border-radius: 8px; padding: 6px 10px; selection-background-color: {C['active']}; }}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {{
  border: 1px solid {C['active_edge']}; }}
QComboBox QAbstractItemView {{ background: {C['panel']}; color: {C['text']};
  selection-background-color: {C['active']}; }}
"""


class FormPage(QtWidgets.QWidget):
    submitted = QtCore.Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("form")
        self.setStyleSheet(FORM_CSS)
        self.widgets = {}
        self.fields: Sequence[dict] = ()

    def build(self, title: str, body: str, fields: Sequence[dict], height: int) -> None:
        old = self.layout()
        if old is not None:
            QtWidgets.QWidget().setLayout(old)       # re-parent and drop the old layout
        u = height / 100.0
        self.fields, self.widgets = fields, {}
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, int(8 * u), 0, int(6 * u))
        col_ = QtWidgets.QWidget()
        col_.setMaximumWidth(int(120 * u))
        lay = QtWidgets.QVBoxLayout(col_)
        t = QtWidgets.QLabel(title)
        t.setProperty("role", "title")
        t.setFont(_font(4.0 * u, QFont.Weight.Medium))
        b = QtWidgets.QLabel(body)
        b.setProperty("role", "muted")
        b.setWordWrap(True)
        b.setFont(_font(2.2 * u))
        lay.addWidget(t)
        lay.addWidget(b)
        lay.addSpacing(int(2 * u))
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setVerticalSpacing(int(1.6 * u))
        form.setHorizontalSpacing(int(3 * u))
        fnt = _font(2.5 * u)
        for f in fields:
            lab = QtWidgets.QLabel(f["label"])
            lab.setFont(fnt)
            lab.setWordWrap(True)
            if f["type"] == "int":
                wdg = QtWidgets.QSpinBox()
                wdg.setRange(int(f.get("min", 0)) - 1, int(f.get("max", 120)))
                wdg.setSpecialValueText("—")
                wdg.setValue(wdg.minimum())
            elif f["type"] == "choice":
                wdg = QtWidgets.QComboBox()
                wdg.addItem("Select…")
                wdg.addItems(list(f["options"]))
            elif f["type"] == "longtext":
                wdg = QtWidgets.QPlainTextEdit()
                wdg.setFixedHeight(int(9 * u))
                wdg.setTabChangesFocus(True)
            else:
                wdg = QtWidgets.QLineEdit()
            wdg.setFont(fnt)
            wdg.setMinimumWidth(int(36 * u))
            self.widgets[f["name"]] = wdg
            form.addRow(lab, wdg)
        lay.addLayout(form)
        self.error = QtWidgets.QLabel("")
        self.error.setProperty("role", "error")
        self.error.setFont(fnt)
        hint = QtWidgets.QLabel("Press Enter to continue")
        hint.setProperty("role", "muted")
        hint.setFont(_font(2.0 * u))
        lay.addSpacing(int(2 * u))
        lay.addWidget(self.error)
        lay.addWidget(hint)
        lay.addStretch(1)
        outer.addWidget(col_, 0, Qt.AlignmentFlag.AlignHCenter)
        first = self.widgets[fields[0]["name"]] if fields else None
        if first is not None:
            QtCore.QTimer.singleShot(0, first.setFocus)

    def values(self) -> Tuple[dict, List[str]]:
        vals, missing = {}, []
        for f in self.fields:
            wdg = self.widgets[f["name"]]
            if f["type"] == "int":
                v = None if wdg.value() == wdg.minimum() else int(wdg.value())
            elif f["type"] == "choice":
                v = None if wdg.currentIndex() == 0 else wdg.currentText()
            elif f["type"] == "longtext":
                v = wdg.toPlainText().strip()
            else:
                v = wdg.text().strip()
            if (v is None or v == "") and not f.get("optional"):
                missing.append(f["label"])
            vals[f["name"]] = v
        return vals, missing

    def try_submit(self) -> bool:
        vals, missing = self.values()
        if missing:
            self.error.setText("Please answer: " + "; ".join(missing))
            return False
        self.error.setText("")
        self.submitted.emit(vals)
        return True

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not isinstance(
                self.focusWidget(), QtWidgets.QPlainTextEdit):
            self.try_submit()
            return
        super().keyPressEvent(e)
