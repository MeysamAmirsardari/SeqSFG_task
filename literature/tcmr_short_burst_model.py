import numpy as np
rng = np.random.default_rng(1)
fs = 1000.0
K = np.arange(-10, 11)
PM = 2.0 * K.size

def window(D, ramp):
    t = (np.arange(int(round(D * fs))) + 0.5) / fs
    w = np.ones_like(t)
    on, off = t < ramp, t > D - ramp
    w[on] = np.sin(0.5 * np.pi * t[on] / ramp) ** 2
    w[off] = np.sin(0.5 * np.pi * (D - t[off]) / ramp) ** 2
    return t, w

def thr(p, snrs):
    p = np.array(p)
    i = int(np.argmax(p >= 0.707))
    return float(np.interp(0.707, [p[i - 1], p[i]], [snrs[i - 1], snrs[i]]))

def run(D, ramp, normalise, n=12000, sig_e=1.0, sig_c=(1.0, 2.0, 3.0)):
    t, w = window(D, ramp)
    E = np.exp(2j * np.pi * np.outer(K, t))
    C = rng.normal(size=(n * 3, K.size)) + 1j * rng.normal(size=(n * 3, K.size))
    z = C @ E
    if normalise:
        z /= np.sqrt(np.mean(np.abs(z) ** 2, axis=1, keepdims=True) / PM)
    z = z.reshape(n, 3, -1)
    w2 = w ** 2
    Em = (np.abs(z) ** 2 * w2).sum(-1)
    sd = float(np.std(10 * np.log10(Em)))
    phi = rng.uniform(0, 2 * np.pi, n)
    snrs = np.arange(-26, 18.1, 1.0)
    ne, nc = rng.normal(size=(n, 3)), rng.normal(size=(n, 3))
    pR, pC = [], {s: [] for s in sig_c}
    for snr in snrs:
        s = np.sqrt(PM * 10 ** (snr / 10))
        Es = (np.abs(z[:, 0, :] + s * np.exp(1j * phi)[:, None]) ** 2 * w2).sum(-1)
        L = 10 * np.log10(np.column_stack([Es, Em[:, 1], Em[:, 2]]))
        pR.append(np.mean((L + sig_e * ne).argmax(1) == 0))
        d = np.column_stack([10 * np.log10(Es / Em[:, 0]), np.zeros(n), np.zeros(n)])
        for sc in sig_c:
            pC[sc].append(np.mean((d + sc * nc).argmax(1) == 0))
    return sd, thr(pR, snrs), {sc: thr(pC[sc], snrs) for sc in sig_c}

print("burst  ramp  noise      level_SD  thr_alone  thr_compare(s=1,2,3 dB)   CMR(s=1,2,3)")
for D, r in ((0.1875, 0.02), (0.1, 0.01), (0.06, 0.01), (0.05, 0.01), (0.04, 0.01), (0.03, 0.01)):
    for norm in (False, True):
        sd, tR, tC = run(D, r, norm)
        print(f"{D*1000:6.1f} {r*1000:4.0f}  {'equalised' if norm else 'natural  '}  {sd:6.2f}   {tR:7.1f}    "
              + " ".join(f"{tC[s]:6.1f}" for s in tC) + "     " + " ".join(f"{tR - tC[s]:5.1f}" for s in tC))

def coherence(J, D=0.05, ramp=0.01, T=0.2, nb=6000, coherent=False):
    t, w = window(D, ramp)
    n = int(nb * T * fs) + 600
    a = np.abs(rng.normal(size=nb) + 1j * rng.normal(size=nb))
    j0 = rng.uniform(-J, J, nb)
    env = []
    for b in range(2):
        j = j0 if coherent else rng.uniform(-J, J, nb)
        e = np.zeros(n)
        on = np.round((0.2 + np.arange(nb) * T + j) * fs).astype(int)
        for k, i in enumerate(on):
            e[i:i + w.size] += a[k] * w
        env.append(e)
    return float(np.corrcoef(env)[0, 1])

print("\njitter  mean|async|  P(pair no overlap)  mean overlap  env corr (independent)   (D = 50 ms, 5 Hz)")
D = 50.0
for J in (0, 10, 20, 30, 40, 50, 65):
    d = np.abs(rng.uniform(-J, J, 400000) - rng.uniform(-J, J, 400000))
    print(f"+-{J:3d}   {d.mean():6.1f} ms     {np.mean(d >= D):5.2f}             {np.mean(np.clip(1 - d / D, 0, 1)):5.2f}        {coherence(J / 1000):5.2f}")
print("coherent jitter +-65: env corr", round(coherence(0.065, coherent=True), 2))
