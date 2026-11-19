"""The session as a script (a generator), shared by the app and the simulator.

Order: questionnaire, instructions, labelled examples, practice with feedback, then the blocks.
Every delay is tested in `blocks_per_delay` blocks of `trials_per_block` trials (52, as Micheyl et
al.'s blocks of 51-52): the first round in an order drawn from the seed, the second round in the
reverse order, so every delay sits equally early and late on average. Half the trials of every
block carry the change, in an order drawn from the seed. A trial interrupted (pause, audio fault)
is played again with a fresh seed attempt. Logging and resume use tcmr's writer.
"""
from __future__ import annotations

from typing import Dict, Generator, List, Optional

import numpy as np

from tcmr.session import Play, Result, Screen, Writer  # noqa: F401  (Writer re-exported)
from tsync.text import QUESTIONNAIRE

from .config import Config
from .stimulus import build, sha256

FIELDS = ["row", "phase", "index", "attempt", "status", "reason", "block", "block_trial",
          "delay_ms", "levels", "target_present", "target_index", "decoys", "response",
          "said_yes", "correct", "rt_ms", "seed", "sha256", "onset_error_ms", "underflows",
          "volume", "muted", "wall", "t_s"]

T = {
    "intro1": ("Listening for a change",
               "You will hear a fast sequence of low and high tones, about two seconds long.\n\n"
               "The low and high tones go up and down in loudness TOGETHER: when a high tone is a "
               "bit louder, so is the low tone that goes with it."),
    "intro1_independent": ("Listening for a change",
               "You will hear a fast sequence of low and high tones, about two seconds long.\n\n"
               "Listen to the HIGH tones and try to ignore the low ones: the low tones change in "
               "loudness at random, and that does not count."),
    "intro2": ("Your task",
               "In half of the sequences, ONE high tone gets louder ON ITS OWN: its low partner "
               "does not. After each sequence, say whether that happened.\n\n"
               "← or 1 = No       → or 2 = Yes\n\n"
               "You will see whether you were right. When you are not sure, guess. "
               "Press P at any time to pause."),
    "intro2_independent": ("Your task",
               "In half of the sequences, ONE high tone is louder than the other high tones. "
               "After each sequence, say whether that happened.\n\n"
               "← or 1 = No       → or 2 = Yes\n\n"
               "You will see whether you were right. When you are not sure, guess. "
               "Press P at any time to pause."),
    "examples": ("Four examples", "Two with the change and two without. Each is labelled after it "
                 "plays."),
    "practice": ("Practice", "{n} practice trials with feedback."),
    "block": ("Part {k} of {n}", "{trials} sequences. Press Space when you are ready."),
    "end": ("Thank you!", "That is the end. Please let the experimenter know."),
    "resume": ("Welcome back", "We will continue where you left off."),
}


def block_plan(cfg: Config, seed: int) -> List[dict]:
    """Every block: its delay and which of its trials carry the change, fixed by the seed."""
    rng = np.random.default_rng([int(seed), 21])
    first = [cfg.delays_ms[i] for i in rng.permutation(len(cfg.delays_ms))]
    order: List[float] = []
    for r in range(cfg.blocks_per_delay):
        order += first if r % 2 == 0 else first[::-1]
    out = []
    for b, d in enumerate(order):
        n = cfg.trials_per_block
        present = np.array([1] * (n // 2) + [0] * (n - n // 2))
        present = present[np.random.default_rng([int(seed), 22, b]).permutation(n)]
        out.append({"block": b, "delay_ms": float(d), "present": [int(x) for x in present]})
    return out


def trial_seconds(cfg: Config, overhead_s: float = 2.3) -> float:
    return (cfg.pre_ms + cfg.sequence_ms(float(np.mean(cfg.delays_ms))) + cfg.post_ms) / 1000 + overhead_s


class Experiment:
    def __init__(self, cfg: Config, writer, seed: int, amplitude=None,
                 resume_rows: Optional[List[dict]] = None):
        self.cfg, self.w, self.seed, self.amp = cfg, writer, int(seed), amplitude
        self.blocks = block_plan(cfg, self.seed)
        self.done = 0                                  # main trials completed
        self.resumed = resume_rows is not None
        if resume_rows:
            self.done = sum(1 for r in resume_rows if r["phase"] == "main" and r["status"] == "ok")

    def _play(self, phase, delay, present, sd, *, mode="yesno", label="", progress=None,
              caption="", block=-1, target_db=None):
        tr, buf = build(self.cfg, delay, present, sd, phase=phase, block=block, amplitude=self.amp,
                        target_db=target_db)
        return Play(tr, buf, sha256(buf), mode, mode == "yesno" and self.cfg.feedback, label,
                    progress, caption)

    def _log(self, play: Play, res: Result, index, attempt, block_trial=None):
        tr = play.trial
        said_yes = None if res.response is None else res.response == 2
        correct = (said_yes == tr.target_present) if (play.mode == "yesno" and res.status == "ok"
                                                      and said_yes is not None) else None
        a, s = res.audio, res.system
        row = {"phase": tr.phase, "index": index, "attempt": attempt, "status": res.status,
               "reason": res.reason, "block": tr.block, "block_trial": block_trial,
               "delay_ms": tr.delay_ms, "levels": tr.levels, "target_present": int(tr.target_present),
               "target_index": tr.target_index, "decoys": " ".join(map(str, tr.decoys)),
               "response": res.response, "said_yes": said_yes, "correct": correct,
               "rt_ms": res.rt_ms, "seed": "-".join(map(str, tr.seed)), "sha256": play.sha,
               "onset_error_ms": a.get("onset_error_ms"), "underflows": a.get("underflows"),
               "volume": s.get("output_volume"), "muted": s.get("muted")}
        from dataclasses import asdict
        self.w.trial(row, {"trial": tr.summary(), "result": asdict(res), "mode": play.mode})
        return correct

    def _until_ok(self, make, log):
        attempt = 0
        while True:
            play = make(attempt)
            res = yield play
            if res.status != "ok":
                log(play, res, attempt)
                self.w.event("repeat", reason=res.reason)
                attempt += 1
                continue
            return play, res, attempt

    def script(self) -> Generator:
        cfg = self.cfg
        ind = cfg.levels == "independent"
        total = cfg.n_trials
        if not self.resumed:
            res = yield Screen("about", "A few questions first", "Press Enter when you are done.",
                               ("return",), "form", QUESTIONNAIRE)
            self.w.meta["participant_answers"] = (res or {}).get("values")
            self.w.save()
            yield Screen("intro1", *T["intro1_independent" if ind else "intro1"], min_s=1.5)
            yield Screen("intro2", *T["intro2_independent" if ind else "intro2"], min_s=1.5)
            yield Screen("examples", *T["examples"])
            for i, (d, present) in enumerate(cfg.examples):
                cap = "That one HAD the change" if present else "That one had NO change"
                make = lambda a, i=i, d=d, p=present, c=cap: self._play(
                    "example", d, bool(p), (self.seed, 1, i, a), mode="listen", label="Example",
                    caption=c)
                play, res, att = yield from self._until_ok(make, lambda p_, r, a: self._log(p_, r, i, a))
                self._log(play, res, i, att)
            n_prac = sum(it[1] for it in cfg.practice)
            yield Screen("practice", T["practice"][0], T["practice"][1].format(n=n_prac))
            j = 0
            rng = np.random.default_rng([self.seed, 2])
            for item in cfg.practice:
                d, n = item[0], item[1]
                tdb = float(item[2]) if len(item) == 3 else None
                present = rng.permutation([1] * (n // 2) + [0] * (n - n // 2))
                for p_ in present:
                    make = lambda a, j=j, d=d, p_=p_, tdb=tdb: self._play(
                        "practice", d, bool(p_), (self.seed, 2, j, a), label=f"Practice {j + 1} of {n_prac}",
                        progress=(j + 1) / n_prac, target_db=tdb)
                    play, res, att = yield from self._until_ok(make, lambda pl, r, a, j=j: self._log(pl, r, j, a))
                    self._log(play, res, j, att)
                    j += 1
        else:
            yield Screen("resume", *T["resume"])
        n_blocks = len(self.blocks)
        while self.done < total:
            b, k = divmod(self.done, cfg.trials_per_block)
            blk = self.blocks[b]
            if k == 0 or self.resumed:
                self.resumed = False
                yield Screen("block", T["block"][0].format(k=b + 1, n=n_blocks),
                             T["block"][1].format(trials=cfg.trials_per_block - k),
                             min_s=3.0 if b else 0.0, progress=self.done / total)
            i = self.done
            make = lambda a, b=b, k=k, blk=blk: self._play(
                "main", blk["delay_ms"], bool(blk["present"][k]), (self.seed, 6, b, k, a),
                label=f"Part {b + 1} of {n_blocks}", progress=i / total, block=b)
            play, res, att = yield from self._until_ok(make, lambda pl, r, a, k=k: self._log(pl, r, i, a, k))
            self._log(play, res, i, att, k)
            self.done += 1
        self.w.close("complete")
        yield Screen("end", *T["end"], ("escape",), "end")
