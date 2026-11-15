"""Simulations behind literature/assessment_2026-11-12.md. Run from the repo root:
    .venv/bin/python literature/assessment_2026-11-12_sim.py
"""
import sys, math
sys.path.insert(0, ".")
import numpy as np
from scipy import stats
from tcoh.model import coherence_matrix
rng = np.random.default_rng(1)

# ---------- A. the published coherence analysis on a 6-component repeating figure
FS, K, R, PERIOD, TONE, RAMP = 1000.0, 6, 8, 125.0, 50.0, 10.0
def env(onsets, total):
    e = np.zeros(total); w = int(TONE); r = int(RAMP)
    sh = np.ones(w); sh[:r] = np.linspace(0, 1, r); sh[-r:] = np.linspace(1, 0, r)
    for o in onsets:
        i = int(round(o)); e[i:i + w] += sh[: max(0, min(w, total - i))]
    return e
def stim(kind, dT):
    total = int(300 + R * PERIOD + 400); base = 300 + np.arange(R) * PERIOD
    lags = (np.arange(K) - (K - 1) / 2) * dT
    E = []
    perm = rng.permutation(K)
    per_rep = [rng.permutation(K) for _ in range(R)]
    for k in range(K):
        if kind == "ordered":    on = base + lags[k]
        elif kind == "scrambled": on = base + lags[perm[k]]
        elif kind == "redrawn":   on = base + np.array([lags[per_rep[r][k]] for r in range(R)])
        elif kind == "anchor":    on = base + rng.uniform(-PERIOD / 2, PERIOD / 2, R)
        E.append(env(on, total))
    return np.stack(E)
def summary(kind, dT, n=40):
    l1, off = [], []
    for _ in range(n if kind != "ordered" else 1):
        c = coherence_matrix(stim(kind, dT), FS)
        M = None
        for name in ("matrix", "c", "C", "coherence"):
            if hasattr(c, name): M = np.asarray(getattr(c, name)); break
        if M is None:
            M = np.asarray([v for v in vars(c).values() if isinstance(v, np.ndarray) and v.ndim == 2][0])
        w = np.sort(np.linalg.eigvalsh(M))[::-1]
        d = np.sqrt(np.diag(M)); N = M / np.outer(d, d)
        l1.append(w[0] / w.sum()); off.append(N[~np.eye(K, dtype=bool)].mean())
    return np.mean(l1), np.mean(off)
print("A. coherence analysis, 6 components x 8 repeats, 50-ms tones every 125 ms")
print("   arrangement   step  spread   lambda1 share   mean pairwise coherence")
print(f"   sync             0       0   {summary('ordered', 0)[0]:.3f}           {summary('ordered', 0)[1]:.3f}")
for dT in (8.0, 16.0):
    for kind in ("ordered", "scrambled", "redrawn"):
        a, b = summary(kind, dT)
        print(f"   {kind:10s}   {dT:4.0f}   {dT*(K-1):5.0f}   {a:.3f}           {b:.3f}")
a, b = summary("anchor", 0); print(f"   random phase     -    full   {a:.3f}           {b:.3f}")

# ---------- B. precision of a 36-trial 3AFC 2-down-1-up track (tcmr rule: 8/4/2 dB)
def track(theta, sigma, n=36, start=75.0, lapse=0.02):
    L, steps, rev, run, last, levels = start, [8.0, 4.0, 2.0], 0, 0, 0, []
    revL = []
    for t in range(n):
        p = 1/3 + (2/3 - lapse) * stats.norm.cdf((L - theta) / sigma)
        ok = rng.random() < p
        step = steps[0] if rev < 2 else steps[1] if rev < 4 else steps[2]
        if ok:
            run += 1
            if run == 2:
                run = 0
                if last == +1: rev += 1; revL.append(L)
                L -= step; last = -1
        else:
            run = 0
            if last == -1: rev += 1; revL.append(L)
            L += step; last = +1
    use = revL[4:] if len(revL) > 5 else revL[-2:]
    return np.mean(use) if use else L
print("\nB. one 36-trial track (3AFC, 2-down-1-up, steps 8/4/2 dB): SD of the threshold estimate")
sd = {}
for sigma in (2.0, 4.0):
    est = np.array([track(58.0, sigma) for _ in range(3000)])
    sd[sigma] = est.std(); print(f"   slope sigma {sigma:.0f} dB: SD {est.std():.2f} dB (bias {est.mean()-58:+.2f})")
def power_paired(effect, se_within, sd_between, n=8, sims=20000):
    d = effect + rng.normal(0, sd_between, (sims, n)) + rng.normal(0, se_within, (sims, n))
    t = d.mean(1) / (d.std(1, ddof=1) / math.sqrt(n))
    return float(np.mean(np.abs(t) > stats.t.ppf(0.975, n - 1)))
print("   power, 8 listeners, two-sided paired t, for a contrast between two conditions with 2 tracks each")
for sigma in (2.0, 4.0):
    se = sd[sigma] * math.sqrt(2) / math.sqrt(2)          # difference of two 2-track means
    print(f"   sigma {sigma:.0f} dB: per-listener SE of the contrast {se:.2f} dB -> power for 1.5 / 3 dB effects: "
          f"{power_paired(1.5, se, 1.5):.2f} / {power_paired(3.0, se, 1.5):.2f}   (between-listener SD of the effect 1.5 dB)")

# ---------- C. 2IFC constant stimuli, 60 trials a condition, 8 listeners
def power_pc(p1, p2, n_trials, n=8, sd_listener=0.06, sims=20000):
    eff = (p1 - p2) + rng.normal(0, sd_listener, (sims, n))
    base = np.clip(p2 + rng.normal(0, 0.05, (sims, n)), 0.52, 0.97)
    a = rng.binomial(n_trials, np.clip(base + eff, 0.5, 0.99)) / n_trials
    b = rng.binomial(n_trials, base) / n_trials
    d = a - b
    t = d.mean(1) / (d.std(1, ddof=1) / math.sqrt(n))
    return float(np.mean(np.abs(t) > stats.t.ppf(0.975, n - 1)))
print("\nC. 2IFC detection, constant stimuli, 8 listeners (listener SD of the effect 0.06)")
for n_tr in (60, 120):
    print(f"   {n_tr} trials per condition: power for a difference of 0.08 / 0.12 / 0.18 in proportion correct (0.70 baseline): "
          + " / ".join(f"{power_pc(0.70 + e, 0.70, n_tr):.2f}" for e in (0.08, 0.12, 0.18)))
print("   false-positive rate at zero effect, 60 trials:", round(power_pc(0.75, 0.75, 60), 3))

# ---------- D. share of the coherence benefit left, by spread
print("\nD. coherence benefit left (sync = 100%, random-phase floor = 0%)")
s0 = summary("ordered", 0)[0]; fl = summary("anchor", 0, n=120)[0]
for dT in (4.0, 8.0, 12.0, 16.0):
    o = summary("ordered", dT)[0]; r = summary("redrawn", dT, n=80)[0]
    print(f"   spread {dT*(K-1):3.0f} ms: ordered = scrambled {100*(o-fl)/(s0-fl):4.0f}%   redrawn {100*(r-fl)/(s0-fl):4.0f}%")
