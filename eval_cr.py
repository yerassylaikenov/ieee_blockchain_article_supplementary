#!/usr/bin/env python3
"""
Camera-ready evaluation for "Anchor, Don't Store".

Corrects the endorsement model reviewed at submission. The gateway sends each
transaction to exactly k endorsing peers chosen uniformly among n = 3, the
client waits for all k responses, and each peer therefore carries a share
k/n of the offered load. Latency and queueing now refer to the same path.

Two policies are reported: OutOf(1,3) and OutOf(2,3).
Seed 20260910, 60,000 replications per configuration.
"""
import json
from math import factorial

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20260910
RUNS = 60_000

P = {
    "n_peers": 3, "peer_servers": 4,
    "endorse_median": 0.025, "endorse_sigma": 0.45,
    "validate_median": 0.018, "validate_sigma": 0.35,
    "key_n": 5, "key_m": 3, "key_servers": 2,
    "key_median": 0.015, "key_sigma": 0.50,
    "fetch_median": 0.060, "fetch_sigma": 0.55,
    "unwrap_median": 0.004, "unwrap_sigma": 0.30,
    "shred_median": 0.040, "shred_sigma": 0.40,
    "think_time": 4.0,
    "rho_guard": 0.98,
}


def lognorm(rng, med, sig, size):
    return np.exp(np.log(med) + sig * rng.standard_normal(size))


def erlang_c_wait(lam, mu, c):
    """Mean queueing delay of an M/M/c station, None when rho >= guard."""
    a = lam / mu
    rho = a / c
    if rho >= P["rho_guard"]:
        return None
    s = sum(a ** j / factorial(j) for j in range(c))
    tail = a ** c / (factorial(c) * (1 - rho))
    return (tail / (s + tail)) / (c * mu - lam)


def read_path(clinicians, k, seed=SEED):
    rng = np.random.default_rng(seed + 1000 * k + clinicians)
    lam = clinicians / P["think_time"]
    per_peer = lam * k / P["n_peers"]
    w_e = erlang_c_wait(per_peer, 1 / P["endorse_median"], P["peer_servers"])
    per_cust = lam / P["key_n"]
    w_k = erlang_c_wait(per_cust, 1 / P["key_median"], P["key_servers"])
    if w_e is None or w_k is None:
        return None, per_peer
    endorse = lognorm(rng, P["endorse_median"], P["endorse_sigma"],
                      (RUNS, k)).max(axis=1) + w_e
    validate = lognorm(rng, P["validate_median"], P["validate_sigma"], RUNS)
    shares = lognorm(rng, P["key_median"], P["key_sigma"],
                     (RUNS, P["key_n"])) + w_k
    shares.sort(axis=1)
    keyq = shares[:, P["key_m"] - 1]
    unwrap = lognorm(rng, P["unwrap_median"], P["unwrap_sigma"], RUNS)
    fetch = lognorm(rng, P["fetch_median"], P["fetch_sigma"], RUNS)
    return endorse + validate + keyq + unwrap + fetch, per_peer


def erasure(m, n, seed=SEED):
    rng = np.random.default_rng(seed + 100 * m + n)
    s = lognorm(rng, P["key_median"], P["key_sigma"], (RUNS, n))
    s.sort(axis=1)
    return s[:, m - 1] + lognorm(rng, P["shred_median"], P["shred_sigma"],
                                 (RUNS, n)).max(axis=1)


def linkage(salted, n=1500, sigma_day=1.5, keep=0.85, buckets=6,
            seed=SEED):
    """Two institutions hold records for the same n patients.  The adversary
    holds the post-shred ledger and, for each record at A, ranks B-records."""
    rng = np.random.default_rng(seed + (1 if salted else 2))
    rid = rng.integers(0, 2 ** 62, n)
    h = rng.integers(0, 2 ** 62, n)
    if salted:
        cA = rng.integers(0, 2 ** 62, n)
        cB = rng.integers(0, 2 ** 62, n)
    else:
        cA = (rid * 1_000_003 + h) % (2 ** 62)
        cB = cA.copy()
    day = rng.uniform(0, 365, n)
    dA = day + rng.normal(0, sigma_day, n)
    dB = day + rng.normal(0, sigma_day, n)
    b = rng.integers(0, buckets, n)
    bA = np.where(rng.random(n) < keep, b, rng.integers(0, buckets, n))
    bB = np.where(rng.random(n) < keep, b, rng.integers(0, buckets, n))
    hits = 0
    for i in range(n):
        exact = np.where(cB == cA[i])[0]
        if exact.size:
            hits += int(exact[0] == i)
            continue
        score = -np.abs(dB - dA[i]) / 3.0 + (bB == bA[i]) * 0.5
        hits += int(int(np.argmax(score)) == i)
    return hits / n, 1.0 / n


FOOT = {
    "commitment_bytes": 120, "grant_bytes": 300, "read_log_bytes": 180,
    "tombstone_bytes": 140,
    "grants_per_encounter": 5, "reads_per_encounter": 12,
    "root_bytes": 32, "epoch_seconds": 600,
    # baseline used only for the upper curve of Fig. 4
    "payload_on_ledger_bytes": 250_000,
}


def footprint(encounters, tombstones=0):
    """Section VII-A: S = n_e*s_c + n_g*s_g + n_r*s_r + n_t*s_t."""
    f = FOOT
    consent = (encounters * f["commitment_bytes"]
               + encounters * f["grants_per_encounter"] * f["grant_bytes"]
               + encounters * f["reads_per_encounter"] * f["read_log_bytes"]
               + tombstones * f["tombstone_bytes"])
    roots = 31_536_000 // f["epoch_seconds"]
    anchor = roots * f["root_bytes"]
    payload = encounters * f["payload_on_ledger_bytes"]
    return consent, anchor, roots, payload


def saturation(k):
    cap = P["peer_servers"] / P["endorse_median"]
    lam = P["rho_guard"] * cap * P["n_peers"] / k
    return lam * P["think_time"]


def main():
    conc = [100, 250, 500, 750, 900, 1000, 1250, 1500, 1700, 1850]
    out = {"seed": SEED, "runs": RUNS, "params": P, "latency": {},
           "saturation": {}}
    for k in (1, 2):
        out["saturation"][k] = saturation(k)
        rows = {}
        for c in conc:
            r, load = read_path(c, k)
            rows[c] = None if r is None else {
                "median": float(np.median(r)) * 1e3,
                "p95": float(np.percentile(r, 95)) * 1e3,
                "per_peer_load": load}
        out["latency"][k] = rows
        print(f"\nOutOf({k},3): saturation at ~{saturation(k):.0f} clinicians")
        for c, v in rows.items():
            print(f"  {c:5d}: " + ("saturated" if v is None else
                  f"median {v['median']:6.1f} ms  p95 {v['p95']:6.1f} ms  "
                  f"per-peer {v['per_peer_load']:6.1f} tx/s"))

    out["erasure"] = {}
    for m, n in ((2, 3), (3, 5), (4, 7), (5, 9)):
        e = erasure(m, n)
        out["erasure"][f"{m}of{n}"] = {"median": float(np.median(e)) * 1e3,
                                       "p95": float(np.percentile(e, 95)) * 1e3}
    print("\nerasure:", {k: (round(v['median'], 1), round(v['p95'], 1))
                         for k, v in out["erasure"].items()})

    salt, chance = linkage(True)
    bare, _ = linkage(False)
    sens = {}
    for sd in (0.5, 1.5, 3.0, 7.0):
        sens[sd] = linkage(True, sigma_day=sd)[0]
    out["linkage"] = {"salted": salt, "bare": bare, "chance": chance,
                      "timing_noise_sensitivity": sens}
    print(f"linkage: bare {bare:.3f}  salted {salt:.3f}  chance {chance:.5f}")
    print("  timing-noise sensitivity:", {k: round(v, 3) for k, v in sens.items()})

    consent, anchor, roots, payload = footprint(1_000_000)
    out["footprint"] = {"params": FOOT, "encounters": 1_000_000,
                        "consent_bytes": consent, "anchor_bytes": anchor,
                        "anchor_roots": roots, "payload_bytes": payload}
    print(f"footprint @1M encounters: consent plane {consent/1e9:.2f} GB "
          f"({consent/2**30:.2f} GiB), anchor {roots:,} roots = "
          f"{anchor:,} B ({anchor/2**20:.2f} MiB)")

    json.dump(out, open("eval_cr.json", "w"), indent=1, default=str)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.grid": True, "grid.alpha": .3,
                         "legend.frameon": False, "figure.dpi": 220,
                         "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    style = {1: ("#1B6CA8", "OutOf(1,3)"), 2: ("#C8102E", "OutOf(2,3)")}
    for k, (col, lab) in style.items():
        xs = [c for c in conc if out["latency"][k][c]]
        ax.plot(xs, [out["latency"][k][c]["median"] for c in xs], ls="--",
                lw=1.4, color=col, marker="o", ms=3.5, label=f"{lab} median")
        ax.plot(xs, [out["latency"][k][c]["p95"] for c in xs], ls="-",
                lw=1.9, color=col, marker="o", ms=3.5, label=f"{lab} p95")
        ax.axvline(saturation(k), color=col, lw=0.9, ls=":", alpha=0.8)
    ax.axhline(400, color="#37474F", lw=1.2, ls=(0, (5, 3)))
    ax.text(110, 412, "400 ms target", fontsize=7.5, color="#37474F")
    ax.set_xlabel("concurrent clinicians")
    ax.set_ylabel("read-path latency, ms")
    ax.set_ylim(100, 500)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.savefig("fig5_cr.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    vals = [bare, salt, chance]
    labs = ["bare content hash", "salted commitment\n(this work)",
            "random guessing"]
    bars = ax.bar(labs, vals, color=["#C8102E", "#0F9D58", "#90A4AE"],
                  width=0.55)
    ax.set_ylabel("top-1 re-identification rate"); ax.set_ylim(0, 1.08)
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.02,
                f"{v:.3f}" if v >= 1e-3 else f"{v:.2e}", ha="center",
                fontsize=8.5)
    fig.savefig("fig7_cr.png"); plt.close(fig)
    # Fig. 4 -- analytical footprint
    enc = np.logspace(4, 7, 50)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(enc, [footprint(e)[3] for e in enc], color="#A52A2A", lw=2.2,
            label="payload written on ledger")
    ax.plot(enc, [footprint(e)[0] for e in enc], color="#1B5E8C", lw=2.2,
            ls=(0, (6, 3)), label="consent plane (this work)")
    ax.plot(enc, [footprint(e)[1] for e in enc], color="#B8860B", lw=2.6,
            ls=(0, (2, 2)), label="anchor plane (this work)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e4, 1e7); ax.set_ylim(1e4, 1e13)
    ax.set_xlabel("annual encounters")
    ax.set_ylabel("annual ledger growth (bytes)")
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    fig.savefig("fig4_cr.png"); plt.close(fig)

    # Fig. 6 -- erasure completion by quorum
    q = ["2of3", "3of5", "4of7", "5of9"]
    labs = ["2-of-3", "3-of-5", "4-of-7", "5-of-9"]
    med = [out["erasure"][k]["median"] for k in q]
    p95 = [out["erasure"][k]["p95"] for k in q]
    x = np.arange(len(q)); w = 0.36
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.bar(x - w / 2, med, w, color="#1B6CA8", label="median")
    ax.bar(x + w / 2, p95, w, color="#F2A03D", label="p95")
    ax.axhline(2000, color="#C8102E", lw=1.3, ls=(0, (5, 3)))
    ax.text(-0.35, 2200, "2 s design target", fontsize=7.5, color="#C8102E")
    for xi, v in zip(x - w / 2, med):
        ax.text(xi, v * 1.08, f"{v:.0f}", ha="center", fontsize=7)
    for xi, v in zip(x + w / 2, p95):
        ax.text(xi, v * 1.08, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_yscale("log"); ax.set_ylim(30, 4000)
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_xlabel("key-plane quorum"); ax.set_ylabel("erasure completion, ms")
    ax.legend(loc="upper right")
    fig.savefig("fig6_cr.png"); plt.close(fig)
    print("figures written: fig4_cr.png fig5_cr.png fig6_cr.png fig7_cr.png")


PUBLISHED = {   # value printed in the camera-ready paper -> where to read it
    "Sec VII-C  OutOf(1,3) p95 @500  (ms)": (219, lambda o: o["latency"][1][500]["p95"]),
    "Sec VII-C  OutOf(2,3) p95 @500  (ms)": (228, lambda o: o["latency"][2][500]["p95"]),
    "Sec VII-C  OutOf(1,3) p95 @1850 (ms)": (391, lambda o: o["latency"][1][1850]["p95"]),
    "Sec VII-C  OutOf(2,3) p95 @900  (ms)": (314, lambda o: o["latency"][2][900]["p95"]),
    "Sec VII-C  saturation k=1 (clinicians)": (1882, lambda o: o["saturation"][1]),
    "Sec VII-C  saturation k=2 (clinicians)": (941, lambda o: o["saturation"][2]),
    "Sec VII-D  erasure 3-of-5 median (ms)": (78.7, lambda o: o["erasure"]["3of5"]["median"]),
    "Sec VII-D  erasure 3-of-5 p95 (ms)": (117.4, lambda o: o["erasure"]["3of5"]["p95"]),
    "Sec VII-E  salted top-1": (0.179, lambda o: o["linkage"]["salted"]),
    "Sec VII-E  bare-hash top-1": (1.000, lambda o: o["linkage"]["bare"]),
    "Sec VII-E  sensitivity sigma=7 d": (0.045, lambda o: o["linkage"]["timing_noise_sensitivity"][7.0]),
    "Sec VII-A  anchor roots per year": (52560, lambda o: o["footprint"]["anchor_roots"]),
}


def check(out):
    """Compare each published value with this run, rounded as printed."""
    ok = True
    print("\nself-check against the paper:")
    for name, (want, get) in PUBLISHED.items():
        got = get(out)
        digits = len(str(want).split(".")[1]) if "." in str(want) else 0
        good = round(got, digits) == round(want, digits)
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name:40s} paper {want}  run {got:.4g}")
    print("ALL PASS" if ok else "MISMATCH - see FAIL lines")
    return ok


if __name__ == "__main__":
    import sys
    main()
    if "--check" in sys.argv:
        o = json.load(open("eval_cr.json"))
        o["latency"] = {int(k): {int(c): v for c, v in r.items()}
                        for k, r in o["latency"].items()}
        o["saturation"] = {int(k): v for k, v in o["saturation"].items()}
        o["linkage"]["timing_noise_sensitivity"] = {
            float(k): v for k, v in o["linkage"]["timing_noise_sensitivity"].items()}
        sys.exit(0 if check(o) else 1)
