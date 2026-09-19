"""The temporal-coherence model of Elhilali et al. (2009), reduced to two channels.

This is the *hypothesis generator*. The behavioural experiment in this package exists to
test a curve, and the curve comes from here: Figure 8B of

    Elhilali M, Ma L, Micheyl C, Oxenham AJ, Shamma SA (2009).
    Temporal coherence in the perceptual organization and cortical representation of
    auditory scenes. Neuron 61:317-329.

which plots the ratio of the second to the first singular value of the channel coherence
matrix as the onset delay between two tone sequences is swept from fully alternating
(dT = 100%) to synchronous (dT = 0%). The ratio is a *segregation* index: near 1 the matrix
has rank 2 and the model predicts two streams, near 0 it has rank 1 and predicts one.

Why re-implement it instead of reading the published bars off the figure
-------------------------------------------------------------------------
Because the published curve is for the published stimulus (300 and 952 Hz, 75 ms tones) and
we need the prediction for *ours*. A reviewer is entitled to ask what the model says about the
sounds the listener actually heard, at the exact asynchronies the experiment used, and to get
a number rather than a ruler held against a bitmap. `predicted_curve` produces that number,
and `reproduce_figure8` checks the implementation against the two values the paper states in
its own text (0.93 alternating, 0.01 synchronous) before any of it is believed.

The pipeline, following the paper's Methods
-------------------------------------------
1. Each channel's envelope is convolved with a bank of "rate filters"

       h(t; w, q) = w * g(w t) cos q  +  w * ghat(w t) sin q,      g(t) = t^2 e^(-3.5t) sin(2 pi t)

   where ghat is the Hilbert transform of g. w (characteristic rate) runs over
   [2 4 8 16 32] Hz and q (characteristic phase) over [0, 2 pi) in steps of pi/3. These are
   bandpass in modulation frequency, which is what makes the analysis about *co-modulation*
   rather than about co-activation: a channel that is simply on contributes nothing.

2. The filtered responses are cross-correlated and summed over the whole bank,

       c_ij = sum_w sum_q  <R_i(.;w,q), R_j(.;w,q)>

   giving a symmetric channel-by-channel coherence matrix C.

3. C is decomposed. lambda2/lambda1 is the output.

One thing the paper leaves implicit, and what we did about it
-------------------------------------------------------------
Taken as bare arithmetic, step 2 does not reproduce the paper's own numbers. Two perfectly
alternating channels are ANTI-correlated, not uncorrelated: their envelopes are complementary,
every odd harmonic of the sequence rate flips sign at a half-period lag, and the signed inner
product comes out near -1. Feeding that into step 3 gives lambda2/lambda1 = 0.16 for the
alternating case, where the paper reports 0.93, and a curve with a hump in the middle rather
than the monotone one in Figure 8B.

The paper says what it means instead: the off-diagonal entries "are at zero for the alternating
sequence", the operation is called "coincidence", and it is performed by "coincidence
detectors". A coincidence detector fires when both inputs are active at once and is silent
otherwise, which is a rectified product; and cortical responses, which the model is meant to
be about, do not go negative. Half-wave rectifying R before the product gives exactly that,
and reproduces both values the paper states:

    reading                     dT=100%   dT=0%     monotone
    signed product (literal)      0.156     0.000    no
    analytic magnitude            0.041     0.000    no
    half-wave rectified           0.893     0.000    YES
    published                     0.93      0.01     yes

So `rectify="halfwave"` is the default. This is an inference about an under-specified method,
not a quotation, and it is flagged as one everywhere it matters. `reproduce_figure8` runs all
three readings so the choice can be re-checked rather than taken on trust, and
`prediction_band` shows what else would have to change to move the answer.

The design constraint this model imposed
-----------------------------------------
Sweeping dT is only a clean ordered axis when the tone fills half the period. With a shorter
tone -- a gap between successive tones in a channel -- the predicted curve is NOT monotone: it
peaks near dT = 75%, where the two channels interdigitate most thoroughly, and comes back down
at full alternation. `duty_cycle_scan` measures that. It is the reason `tone_ms` defaults to
exactly half of `soa_ms` in `tcoh.config`, and the reason a validator refuses anything else
without an explicit acknowledgement.

With two channels the decomposition is closed-form and worth writing down, because it says
exactly what the index means:

    C = [[c11, c12], [c12, c22]],  and if c11 = c22 = c  (our case, by symmetry)
    lambda+- = c +- |c12|   so   lambda2/lambda1 = (1 - r)/(1 + r),   r = |c12| / c

so the index is a monotone re-expression of the normalised cross-channel correlation r.
r = 1 (perfectly co-modulated) gives 0; r = 0 (independent) gives 1. `coherence_matrix`
returns r alongside the ratio so the report can quote whichever is clearer.

What this reduction assumes, stated plainly so it can be argued with
--------------------------------------------------------------------
The paper's stage 1 takes an auditory spectrogram. We hand it the *analytic envelopes of the
two tones*, one per channel. That is exact only if the two tones excite disjoint sets of
cochlear channels and the peripheral transformation is envelope-preserving. For pure tones
separated by 15 semitones (a factor of 2.38, roughly 8 ERBs apart at 1 kHz) the first is a
good approximation and `channel_crosstalk` quantifies how good. The second ignores
compression and adaptation. `predicted_curve(compression=...)` re-runs the sweep through a
power-law compression so the report can show the curve is not an artefact of that choice.

Nothing here touches the listener. The model's only job is to state the prediction before the
data exist; `tcoh.analysis` then tests the observed curve against it and against the
alternatives in `tcoh.predictors`, which are deliberately not this model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

RATES_HZ: Tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 32.0)
PHASES: Tuple[float, ...] = tuple(k * math.pi / 3.0 for k in range(6))
PUBLISHED = {"alternating": 0.93, "synchronous": 0.01}
# The two values Elhilali et al. state in the text for Figure 8: "The ratio of the
# second-to-first singular values (l2/l1) equals 0.93" (alternating) and "the ratio l2/l1 is
# equal to 0.01" (synchronous). They are the only numbers in the paper precise enough to
# check an implementation against, so they are what `reproduce_figure8` checks.


# ----------------------------------------------------------------------------
# the rate filter bank
# ----------------------------------------------------------------------------
def seed_function(t: np.ndarray) -> np.ndarray:
    """g(t) = t^2 exp(-3.5 t) sin(2 pi t), zero for t < 0 (the filter is causal)."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    m = t >= 0.0
    out[m] = t[m] ** 2 * np.exp(-3.5 * t[m]) * np.sin(2.0 * np.pi * t[m])
    return out


def _analytic_pair(tau: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """g and its Hilbert transform on the causal support, padded so the transform is clean.

    scipy's hilbert assumes the signal is periodic; g decays to nothing well before the end of
    the support, but it starts abruptly at 0, so the array is zero-padded on both sides and the
    interior is taken back. Without the pad the wrap-around ripple leaks into the first few
    milliseconds of every filter, which is exactly where the onset of a tone lands.
    """
    from scipy.signal import hilbert
    g = seed_function(tau)
    pad = g.size
    padded = np.concatenate([np.zeros(pad), g, np.zeros(pad)])
    return g, -np.imag(hilbert(padded))[pad:pad + g.size]


def rate_filter(rate_hz: float, phase: float, fs: float, span: float = 4.0) -> np.ndarray:
    """One h(t; w, q), sampled at fs Hz.

    `span` is the length of the causal support in units of the filter's own period; g(t) has
    decayed by more than 60 dB from its peak by t = 4, so 4 periods is the whole filter.
    """
    n = max(2, int(round(span * fs / rate_hz)))
    tau = np.arange(n) / fs * rate_hz              # time in units of the filter's period
    g, ghat = _analytic_pair(tau)
    return rate_hz * (g * math.cos(phase) + ghat * math.sin(phase))


def filter_bank(fs: float, rates_hz: Sequence[float] = RATES_HZ,
                phases: Sequence[float] = PHASES) -> List[Tuple[float, float, np.ndarray]]:
    """(rate, phase, impulse response) for every filter, built once and reused across a sweep."""
    return [(w, q, rate_filter(w, q, fs)) for w in rates_hz for q in phases]


# ----------------------------------------------------------------------------
# the coherence matrix
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Coherence:
    ratio: float                 # lambda2 / lambda1   -- the published index
    correlation: float           # r = |c12| / sqrt(c11 c22)  -- the same thing, unsquashed
    eigenvalues: Tuple[float, ...]
    matrix: Tuple[Tuple[float, ...], ...]
    n_significant: int           # eigenvalues above `significance`, i.e. the predicted stream count

    def as_dict(self) -> dict:
        return {"ratio": self.ratio, "correlation": self.correlation,
                "eigenvalues": list(self.eigenvalues), "n_significant": self.n_significant}


def coherence_matrix(envelopes: np.ndarray, fs: float, bank=None,
                     significance: float = 0.2, trim_s: float = 0.0,
                     rectify: str = "halfwave") -> Coherence:
    """Run the two-stage analysis on an array of channel envelopes, shape (n_channels, n_samples).

    `rectify` is the reading of step 2: "halfwave" (the default, coincidence detection, the
    only one that reproduces the paper's numbers), "none" (the literal signed product), or
    "magnitude" (the analytic envelope of the response). See the module docstring.

    `trim_s` drops the leading seconds before the inner products are taken. The paper excludes
    the first 0.6 s of each neural response "so as to avoid adaptation effects"; the equivalent
    concern here is the filter bank's own start-up transient, which is as long as the slowest
    filter (0.5 s at 2 Hz) and is the same in every condition but adds a constant to every
    inner product. The default keeps everything, and the verification report shows the sweep
    is unchanged by trimming.
    """
    env = np.atleast_2d(np.asarray(envelopes, dtype=float))
    n_ch = env.shape[0]
    if bank is None:
        bank = filter_bank(fs)
    skip = int(round(trim_s * fs))
    c = np.zeros((n_ch, n_ch))
    for _, _, h in bank:
        # 'full' then truncate to the input length keeps the response causal and the same
        # length as the envelope, so channels stay sample-aligned with each other.
        r = np.stack([np.convolve(env[i], h)[: env.shape[1]] for i in range(n_ch)])
        r = _rectify(r, rectify)
        if skip:
            r = r[:, skip:]
        c += r @ r.T
    return _decompose(c, significance)


def _rectify(r: np.ndarray, mode: str) -> np.ndarray:
    if mode == "halfwave":
        return np.maximum(r, 0.0)
    if mode == "magnitude":
        from scipy.signal import hilbert
        return np.abs(hilbert(r, axis=-1))
    if mode == "none":
        return r
    raise ValueError(f"unknown rectify mode {mode!r}")


def _decompose(c: np.ndarray, significance: float) -> Coherence:
    ev = np.linalg.eigvalsh(c)[::-1]                 # symmetric, so eigen == singular values
    ev = np.clip(ev, 0.0, None)
    lam1 = float(ev[0])
    ratio = float(ev[1] / lam1) if lam1 > 0 and ev.size > 1 else 0.0
    d = np.sqrt(np.clip(np.diag(c), 1e-300, None))
    corr = float(abs(c[0, 1]) / (d[0] * d[1])) if c.shape[0] > 1 else 1.0
    n_sig = int(np.sum(ev >= significance * lam1)) if lam1 > 0 else 0
    return Coherence(ratio, corr, tuple(float(x) for x in ev),
                     tuple(tuple(float(x) for x in row) for row in c), n_sig)


# ----------------------------------------------------------------------------
# envelopes of a two-tone sequence
# ----------------------------------------------------------------------------
def sequence_envelope(onsets_ms: Sequence[float], tone_ms: float, ramp_ms: float,
                      total_ms: float, fs: float) -> np.ndarray:
    """The raised-cosine-gated envelope of one channel: a tone at each onset, nothing between.

    This is the envelope of the *tone*, not of the waveform: the model's first stage sees a
    spectrogram, whose value in an active channel is the tone's amplitude envelope.
    """
    n = int(round(total_ms * 1e-3 * fs))
    env = np.zeros(n)
    body = _gate(tone_ms, ramp_ms, fs)
    for t in onsets_ms:
        i = int(round(t * 1e-3 * fs))
        j = min(n, i + body.size)
        if i < n and j > i:
            env[i:j] += body[: j - i]
    return env


def _gate(tone_ms: float, ramp_ms: float, fs: float) -> np.ndarray:
    n = max(1, int(round(tone_ms * 1e-3 * fs)))
    r = min(int(round(ramp_ms * 1e-3 * fs)), n // 2)
    w = np.ones(n)
    if r > 0:
        up = 0.5 * (1.0 - np.cos(np.pi * np.arange(r) / r))
        w[:r], w[n - r:] = up, up[::-1]
    return w


def two_tone_envelopes(lag_ms: float, tone_ms: float = 75.0, soa_ms: float = 150.0,
                       n_tones: int = 6, ramp_ms: float = 10.0, fs: float = 1000.0,
                       lead_ms: float = 0.0, compression: float = 1.0) -> Tuple[np.ndarray, float]:
    """Two isochronous channels, the second lagging the first by `lag_ms`. Returns (2, n) and total ms.

    `compression` applies env ** compression, a stand-in for cochlear compression (0.3 is the
    usual exponent). It is off by default because the paper's stage 1 has none.
    """
    total = lead_ms + (n_tones - 1) * soa_ms + tone_ms + max(lag_ms, 0.0) + soa_ms
    a = sequence_envelope([lead_ms + k * soa_ms for k in range(n_tones)], tone_ms, ramp_ms, total, fs)
    b = sequence_envelope([lead_ms + lag_ms + k * soa_ms for k in range(n_tones)], tone_ms, ramp_ms, total, fs)
    env = np.stack([a, b])
    if compression != 1.0:
        env = env ** compression
    return env, total


# ----------------------------------------------------------------------------
# the prediction
# ----------------------------------------------------------------------------
def lag_ms_from_pct(pct: float, soa_ms: float) -> float:
    """dT% is the onset delay as a percentage of HALF the period: 100% is exact alternation."""
    return float(pct) / 100.0 * soa_ms / 2.0


def pct_from_lag_ms(lag_ms: float, soa_ms: float) -> float:
    return 200.0 * float(lag_ms) / soa_ms


def predicted_curve(pcts: Sequence[float], tone_ms: float = 75.0, soa_ms: float = 150.0,
                    n_tones: int = 6, ramp_ms: float = 10.0, fs: float = 1000.0,
                    compression: float = 1.0, trim_s: float = 0.0, rectify: str = "halfwave",
                    rates_hz: Sequence[float] = RATES_HZ,
                    normalise_filters: bool = False) -> Dict[float, Coherence]:
    """lambda2/lambda1 at each dT%, for a stimulus with these parameters. The hypothesis, as numbers."""
    bank = filter_bank(fs, rates_hz)
    if normalise_filters:
        bank = [(w, q, h / (np.linalg.norm(h) or 1.0)) for w, q, h in bank]
    out = {}
    for p in pcts:
        env, _ = two_tone_envelopes(lag_ms_from_pct(p, soa_ms), tone_ms, soa_ms, n_tones,
                                    ramp_ms, fs, compression=compression)
        out[float(p)] = coherence_matrix(env, fs, bank=bank, trim_s=trim_s, rectify=rectify)
    return out


# the choices the paper does not pin down. Each is a defensible reading of the Methods; the
# prediction is only worth pre-registering to the extent it survives all of them.
VARIANTS: Dict[str, dict] = {
    "default": {},
    "energy_normalised_filters": {"normalise_filters": True},
    "rates_2_16": {"rates_hz": (2.0, 4.0, 8.0, 16.0)},
    "rates_4_32": {"rates_hz": (4.0, 8.0, 16.0, 32.0)},
    "cochlear_compression": {"compression": 0.3},
    "long_sequence": {"n_tones": 16},
    "fine_sampling": {"fs": 2000.0},
    "no_startup_transient": {"trim_s": 0.5},
}


def prediction_band(pcts: Sequence[float], tone_ms: float = 75.0, soa_ms: float = 150.0,
                    n_tones: int = 6, **kw) -> dict:
    """The predicted curve under every variant in VARIANTS, and what survives all of them.

    A pre-registered prediction that holds only for one arbitrary setting of an
    under-specified filter bank is not a prediction. This returns the spread, so the
    hypothesis can be stated at the strength the model actually supports: the ordering and
    the endpoints are common to every variant, the exact intermediate heights are not.
    """
    base = dict(tone_ms=tone_ms, soa_ms=soa_ms, n_tones=n_tones, **kw)
    curves = {}
    for name, over in VARIANTS.items():
        c = predicted_curve(pcts, **{**base, **over})
        curves[name] = [c[float(p)].ratio for p in pcts]
    arr = np.array([curves[k] for k in curves])
    return {"pcts": [float(p) for p in pcts], "curves": curves,
            "lo": arr.min(0).tolist(), "hi": arr.max(0).tolist(), "mid": arr.mean(0).tolist(),
            "max_spread": float((arr.max(0) - arr.min(0)).max()),
            "all_monotone": bool(all(np.all(np.diff(v) >= -1e-9) for v in arr)),
            "all_agree_on_order": _order_agreement(arr),
            # the same two claims, with a reversal counted only when it is big enough to be a
            # claim -- plus the raw number, so the reader judges rather than trusts the flag
            "max_decrease": float(max(0.0, -np.diff(arr, axis=1).min())),
            "all_monotone_within_tol": bool(-np.diff(arr, axis=1).min() <= MONOTONE_TOLERANCE),
            "all_agree_on_order_within_tol": _order_agreement_within(arr, MONOTONE_TOLERANCE),
            "monotone_tolerance": MONOTONE_TOLERANCE}


def _order_agreement(arr: np.ndarray) -> bool:
    """Do all variants rank the dT levels the same way? The ordinal claim, checked."""
    ranks = np.argsort(np.argsort(arr, axis=1), axis=1)
    return bool(np.all(ranks == ranks[0]))


MONOTONE_TOLERANCE = 0.01
# A reversal smaller than this is not a prediction of a reversal. The strict flag stays
# available and `max_decrease` is always reported, so nothing is hidden -- but sampling dT
# finely enough to put two levels 0.02 apart in the index will eventually make some variant
# step down by a rounding error, and calling that "the model predicts a non-monotone curve"
# would be false. The tolerance is fixed here, once, at a fifth of the typical spread BETWEEN
# variants, and never tuned to whatever a configuration happens to need.


def _order_agreement_within(arr: np.ndarray, tol: float) -> bool:
    """Do all variants agree on the order of every pair they separate by more than `tol`?"""
    n = arr.shape[1]
    for i in range(n):
        for j in range(i + 1, n):
            d = arr[:, j] - arr[:, i]
            if np.all(np.abs(d) <= tol):
                continue                      # no variant claims these two differ
            signs = {int(np.sign(x)) for x in d if abs(x) > tol}
            if len(signs) > 1:
                return False
    return True


def duty_cycle_scan(duties: Sequence[float] = (0.25, 0.375, 0.5, 0.625, 0.75),
                    pcts: Sequence[float] = (0.0, 12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 87.5, 100.0),
                    soa_ms: float = 150.0, n_tones: int = 6, fs: float = 1000.0) -> dict:
    """Is dT a monotone axis at this duty cycle? For most of them it is not.

    The experiment needs dT to be an ordered variable: if the model's own index is not monotone
    in dT, a monotone behavioural result would confirm nothing and a non-monotone one could not
    be interpreted. Only tone = soa/2 passes, which is why the config insists on it.
    """
    out = {}
    for duty in duties:
        c = predicted_curve(pcts, tone_ms=duty * soa_ms, soa_ms=soa_ms, n_tones=n_tones, fs=fs)
        v = np.array([c[float(p)].ratio for p in pcts])
        out[float(duty)] = {"ratios": v.tolist(), "monotone": bool(np.all(np.diff(v) >= -1e-9)),
                            "argmax_pct": float(pcts[int(np.argmax(v))])}
    return out


def reproduce_figure8(fs: float = 1000.0) -> dict:
    """Run the paper's own Figure 8 stimulus and compare with the two values it reports.

    300 Hz and 952 Hz, both tones 75 ms, onset delay swept 0 to 100%. The frequencies do not
    enter this reduction (each tone is one channel), so what is being checked is the timing
    geometry and the filter bank, which is what could be wrong.
    """
    pcts = [100.0, 87.5, 75.0, 62.5, 50.0, 37.5, 25.0, 12.5, 0.0]
    readings = {}
    for mode in ("halfwave", "none", "magnitude"):
        curve = predicted_curve(pcts, tone_ms=75.0, soa_ms=150.0, n_tones=8, fs=fs, rectify=mode)
        ratios = [curve[p].ratio for p in pcts]
        readings[mode] = {
            "ratios": ratios, "alternating": ratios[0], "synchronous": ratios[-1],
            "monotone": bool(np.all(np.diff(ratios[::-1]) >= -1e-9)),
            "alternating_err": abs(ratios[0] - PUBLISHED["alternating"]),
            "synchronous_err": abs(ratios[-1] - PUBLISHED["synchronous"])}
    best = min(readings, key=lambda m: readings[m]["alternating_err"] + readings[m]["synchronous_err"])
    return {"pcts": pcts, "published": dict(PUBLISHED), "readings": readings, "best": best,
            **readings["halfwave"]}


def channel_crosstalk(f_a_hz: float, f_b_hz: float) -> dict:
    """How separate are two pure tones on the cochlea, in the units the reduction needs.

    The reduction treats each tone as owning a channel. That is safe when the tones are far
    enough apart that neither is in the other's auditory filter. Reported as the separation in
    ERBs (Glasberg & Moore 1990) and as the attenuation of one tone in a rounded-exponential
    filter centred on the other, which is the number that says how much envelope of B leaks
    into channel A.
    """
    def erb(f):                       # Glasberg & Moore (1990), Hz
        return 24.7 * (4.37 * f / 1000.0 + 1.0)
    def erb_number(f):
        return 21.4 * math.log10(4.37 * f / 1000.0 + 1.0)
    lo, hi = min(f_a_hz, f_b_hz), max(f_a_hz, f_b_hz)
    n_erb = erb_number(hi) - erb_number(lo)

    def roex_db(centre, other):
        # roex(p) skirt of the filter centred on `centre`, evaluated at `other`.
        p = 4.0 * centre / erb(centre)
        g = abs(other - centre) / centre
        w = (1.0 + p * g) * math.exp(-p * g)
        return -10.0 * math.log10(max(w, 1e-300))

    # Both directions, and the smaller of the two is the one that matters: it is the largest
    # amount by which one tone's envelope contaminates the other tone's channel, which is the
    # assumption the two-channel reduction rests on. The roex skirt is fitted near the peak and
    # extrapolates to absurd numbers far out, so the figure is capped where it stops meaning
    # anything and labelled as a floor rather than an estimate.
    db = min(roex_db(lo, hi), roex_db(hi, lo))
    capped = db > 60.0
    return {"f_low_hz": lo, "f_high_hz": hi, "semitones": 12.0 * math.log2(hi / lo),
            "octaves": math.log2(hi / lo), "erbs": n_erb,
            "roex_attenuation_db": min(db, 60.0), "roex_is_floor": capped}


def harmonic_proximity(f_a_hz: float, f_b_hz: float, max_term: int = 6) -> dict:
    """The nearest LOW-order frequency ratio and how far the pair sits from it.

    Two simultaneous pure tones in a simple ratio fuse partly through harmonicity, which would
    be a grouping cue this experiment is not manipulating and does not want. The relevant
    ratios are the low-order ones -- 1:1, 2:1, 3:2, 3:1, 4:3, 5:2 -- because those are the ones
    at which a pair is heard as a single harmonic event; 19:8 is arithmetically close to
    plenty of intervals and perceptually close to none, so terms above `max_term` are not
    searched. Mistuning of more than a few percent is enough for the cue to be weak. The number
    is reported rather than assumed, and `tcoh.verify` flags it when it is small.
    """
    r = max(f_a_hz, f_b_hz) / min(f_a_hz, f_b_hz)
    best = None
    for q in range(1, max_term + 1):
        for pnum in range(q, max_term + 1):
            if math.gcd(pnum, q) != 1:
                continue
            cand = pnum / q
            err = abs(r - cand) / cand
            if best is None or err < best["mistuning"]:
                best = {"ratio": f"{pnum}:{q}", "value": cand, "mistuning": err, "order": pnum + q}
    best["mistuning_pct"] = 100.0 * best["mistuning"]
    best["frequency_ratio"] = r
    best["max_term"] = max_term
    return best
