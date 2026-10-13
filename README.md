# Supplementary material — "Anchor, Don't Store"

Reproduces every number and figure in Section VII of the camera-ready paper.

## Run

Use a virtual environment; recent Linux distributions and Homebrew Python
refuse a system-wide `pip install` (PEP 668).

    python3 -m venv .venv
    source .venv/bin/activate          # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    python eval_cr.py --check

Runtime is a few seconds. The run writes `eval_cr.json` and four figures, then
compares every value printed in the paper with the run and prints PASS/FAIL.
Exit code 0 means everything matches. Tested with Python 3.12, NumPy 2.4,
Matplotlib 3.10; any Python >= 3.8 with NumPy >= 1.20 should work.

## What is reproduced

| Paper item | Output |
|---|---|
| Sec. VII-A footprint (3.8 GB/yr, 52,560 roots, 1.6 MiB) | console + `eval_cr.json["footprint"]` |
| Fig. 4 analytical footprint | `fig4_cr.png` |
| Sec. VII-C, Fig. 5 read latency and saturation | `fig5_cr.png`, `eval_cr.json["latency"]`, `["saturation"]` |
| Sec. VII-D, Fig. 6 erasure completion | `fig6_cr.png`, `eval_cr.json["erasure"]` |
| Sec. VII-E, Fig. 7 linkage and noise sensitivity | `fig7_cr.png`, `eval_cr.json["linkage"]` |

All results are deterministic: seed 20260910, 60,000 replications per
configuration. These are consequences of a stated model, not testbed
measurements.

## Model

**Endorsement.** Policy OutOf(k, 3): the gateway forwards each transaction to k
peers drawn uniformly from three; the client waits for all k; each peer
carries a share k/3 of the offered load lambda = N/4 (one request every four
seconds per clinician). Each peer is a four-server queue, lognormal service with
median 25 ms. Queueing delay from Erlang-C; a station is saturated at
utilisation 0.98. Hence saturation at N = 0.98 * 160 * 3 * 4 / k, giving 1,882
clinicians for k = 1 and 941 for k = 2.

**Key plane.** Five custodians, two servers each, 15 ms median service; three
shares needed. Never the binding resource below ~2,600 clinicians.

**Linkage.** 1,500 patients, one record each at two institutions. Shared day
~ U(0, 365); each institution adds Gaussian timing noise, sigma = 1.5 days.
Size bucket of six, preserved with probability 0.85. Salted-case score:
-|dt|/3 + 0.5 * [same bucket], top-1 argmax. Chance = 1/1,500. Sensitivity
sweeps sigma over 0.5, 1.5, 3 and 7 days.

**Footprint.** Commitment 120 B, grant 300 B, read-log entry 180 B, tombstone
140 B; five grants and twelve reads per encounter; 32-byte root per ten-minute
epoch. The on-ledger payload baseline in Fig. 4 is 250 kB per encounter.

All parameters live in the `P` and `FOOT` dictionaries at the top of the script.
