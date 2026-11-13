"""Render every kind of screen to PNG without a display (for review, the README and the tests)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from .text import QUESTIONNAIRE, T


def render_previews(out_dir: Path, size=(1440, 900)) -> List[Path]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from .app import Window
    from .ui import View

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    win = Window()
    win.resize(*size)
    win.show()
    shots = {
        "01_instructions": View(kind="message", title=T["intro2_title"], body=T["intro2_body"],
                                figure="sequences", footer=T["space_hint"]),
        "02_keys": View(kind="message", title=T["intro3_title"], body=T["intro3_body"],
                        figure="keys", footer=T["space_hint"]),
        "03_trial_seq1": View(kind="trial", label="Part 2 of 6", progress=0.31,
                              panels=["active", "idle"], prompt=T["prompt_listen"]),
        "04_trial_seq2": View(kind="trial", label="Part 2 of 6", progress=0.31,
                              panels=["idle", "active"], prompt=T["prompt_listen"]),
        "05_trial_answer": View(kind="trial", label="Part 2 of 6", progress=0.31,
                                prompt=T["prompt_respond"], prompt_style="text"),
        "06_feedback_correct": View(kind="trial", label="Part 2 of 6", progress=0.31,
                                    panels=["idle", "good"], prompt="✓  " + T["correct"],
                                    prompt_style="good"),
        "07_feedback_wrong": View(kind="trial", label="Part 2 of 6", progress=0.31,
                                  panels=["bad", "idle"], prompt="✗  " + T["wrong"],
                                  prompt_style="bad"),
        "08_demo": View(kind="trial", label="Example", panels=["idle", "idle"],
                        captions=[T["caption_sync"], T["caption_async"]]),
        "09_rest": View(kind="rest", title=T["rest_title"].format(i=3, n=6),
                        body=T["rest_long_body"], figure="blocks", data={"blocks": 6, "done": 3},
                        footer=T["space_hint"]),
        "10_pause": View(kind="trial", label="Part 2 of 6", progress=0.31,
                         overlay=("pause", T["paused_title"], T["paused_body"])),
        "11_alert": View(kind="trial", label="Part 2 of 6", progress=0.31,
                         overlay=("alert", "Please call the experimenter",
                                  "The system volume changed from 50 to 44. Set it back to 50."
                                  "\n\nExperimenter: fix it, then press Space.")),
    }
    paths = []
    for name, v in shots.items():
        win.stack.setCurrentIndex(0)
        win.stage.show_view(v)
        app.processEvents()
        p = out_dir / f"{name}.png"
        win.grab().save(str(p))
        paths.append(p)
    win.form.build(T["about_title"], T["about_body"], QUESTIONNAIRE, size[1])
    win.stack.setCurrentIndex(1)
    app.processEvents()
    p = out_dir / "12_form.png"
    win.grab().save(str(p))
    paths.append(p)
    win.close()
    return paths
