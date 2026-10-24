# sweet.tv TV programme recommendation - GSB final exam

Khamzat Yusupov (Юсупов Хамзат), Kaggle `khamzatiusupov`.

## Scored submission and current status

`submission_final.csv` is the original submitted 50/50 personal-frequency/global-popularity RRF model. The student supplied its completed Kaggle screenshot: **private 0.31291**, public 0.32185. Evidence and the exact file digest are under `evidence/`. This is a returned score, not a frozen-leaderboard rank. There was no authenticated API retrieval by the agent.

`submission_khamzat.csv` is a subsequent, locally selected recency/channel-context candidate. It uses the same original hold-out. Its returned private score is **0.31276**, public **0.33163**, documented in `evidence/kaggle_v2_score.png`. This is lower than the original private 0.31291, despite higher local and public scores. Therefore the first submission remains the designated final. The file was renamed by the student from `submission_candidate_v2.csv` without changing its bytes; its recorded SHA-256 matches.

## Reproduce

Accept the competition rules under your own Kaggle account and download the five original files. Data are not redistributed in this repository. Python 3.9.6, NumPy 2.0.2, SciPy 1.13.1 and scikit-learn 1.6.1 were used on macOS arm64. Install `requirements.txt` in a suitable environment.

```sh
export SWEETTV_DATA='/path/to/sweettv-tv-program-recommender'
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python3 src/prepare.py
python3 src/model.py
cp artifacts/validation_results.json artifacts/validation_before_recency.json
python3 src/model.py --final
python3 src/recency.py
python3 src/recency.py --final
python3 src/verify.py
```

The preparation uses streaming CSV reads, but retains channel schedules and interaction aggregates in memory; it is not a constant-memory claim. Scripts take tens of seconds on the measured machine; recorded timings are in their JSON outputs. Input hashes are retained. Network access is not needed for modelling after downloading the files. No future viewing file is used.

## Join, split, fairness

Views and broadcasts are half-open intervals on the same numeric channel. Require `min(view_stop, airing_end) - max(view_start, airing_start) >= 120` seconds. A qualifying match is one event. Multiple matches of the same programme count as repeated events. `duraton` is not programme length; 117 records disagree with the naive wall-clock difference, so interval endpoints are authoritative. Timezone offsets are not supplied and no timezone conversion is invented.

Train ends 2020-05-18 00:00; validation ends 2020-07-27 00:00. Clip crossing intervals. No views in weeks 31-42 enter fitting. Validation uses all 6,363 nonzero IDs in its schedule; final prediction uses 6,636. Repeats are allowed for every method. Labels come only from qualifying nonzero-ID overlaps. Missing IDs are never a programme or a negative label. Final fitting on weeks 11-30 happens only after local model selection.

Evaluate the same 1,246 template users with >=1 eligible labelled hold-out item. Include 8 users cold in the early training window; exclude 14 users with no eligible truth. Every method uses the identical popularity fallback for cold users. Final all-20-week fitting has 2 cold template users. AP@5 divides its rank-weighted hit sum by `min(5, number of relevant items)`. Its mean is MAP@5. Nothing is claimed about an unknown hidden Kaggle label-generation rule.

## Implemented methods and measured results

| Method | Local MAP@5 |
|---|---:|
| Distinct-viewer popularity | 0.77644 |
| Metadata TF-IDF/cosine profile | 0.18376 |
| Own implicit ALS, 32 factors | 0.21629 |
| Personal event frequency | 0.77441 |
| Initial RRF: 50% frequency + 50% popularity | 0.80992 |
| Recency/channel-context refinement | 0.83182 |

Content uses category, three genres, decade, director and actors. Vocabulary/IDF are training-only; future schedule metadata is not used as content features. Profiles are log-count-weighted averages. Own implicit ALS minimizes weighted square loss with `p_ui = 1[n_ui > 0]`, `c_ui = 1 + 20 log(1+n_ui)` and L2 regularization. Two settings were compared: `(factors=32, reg=0.1)` and `(64, 1)`, 12 epochs, seed 20261024. Both required methods are evaluated separately even though their weights were zero in the best initial fusion.

Initial fusion searched 17 explicit RRF settings; failed configurations and the prior percentile-rank attempt remain in the audit files. RRF uses `sum_m weight_m / (60 + rank_m)`, with one-based ranks. Raw model scores are not interchangeable probabilities. Ties ultimately resolve to ascending numeric programme ID. All outputs contain exactly five distinct nonzero scheduled programme IDs, in original template-user order.

The later refinement searched 7/14/28-day half-lives, then 27 fixed RRF mixtures. Decay is accumulated per qualifying event before each fitting cutoff. Past channel affinity additionally uses unlabelled viewing context, with each channel interval capped at six hours. Programme-channel exposure is calculated only from the permitted future schedule, normalized by labelled airtime per channel. The winning weights are 25% global popularity, 25% 28-day personal event frequency and 50% 28-day channel context. Cold users still receive the same global popularity fallback. The refinement was performed after observing the first returned score; its settings were selected using the original hold-out, not hidden labels. Repeated tuning on that hold-out can overfit.

## Verification and evidence limits

`src/verify.py` compares the optimized interval join with an independent brute-force oracle for 240 sampled views, checks every aggregate's time/ID/overlap invariants, checks implicit ALS against a separate dense calculation (max discrepancy 6.7e-16), independently expands AP terms for both examples, verifies the scored CSV and computes conditional paired-user bootstrap intervals. Additional decay and candidate-format checks are in `artifacts/recency_verification.json` and `artifacts/candidate_v2_check.json`.

Examples use the same most-active and least-active warm users across methods, chosen by training event count rather than recommendation hits. They include IDs, titles, categories and observed hold-out matches. A programme ID may merge multiple broadcasts/episode titles. Absence from this platform's log does not establish that a programme was never watched. Label coverage is only 10.76% of early schedule rows. The private/local score gap is real, but its causal source is unverified. No live experiment, user enjoyment or revenue impact is claimed.

AI assistance: Codex generated/debugged code, inspected data, searched references, ran checks and drafted documentation. Student supplied data, account and Kaggle screenshot and performed the upload. Personal student reasoning is not invented. The course submission consists only of the PDF and an original unmodified native session export; the CSV belongs here.

## References actually consulted

- Hu, Koren and Volinsky (2008), *Collaborative Filtering for Implicit Feedback Datasets*: https://yifanhu.net/PUB/cf.pdf
- scikit-learn 1.6.1 `DictVectorizer` / `TfidfTransformer` official documentation.
- Cormack, Clarke and Büttcher (2009), *Reciprocal rank fusion outperforms condorcet and individual rank learning methods*: https://doi.org/10.1145/1571941.1572114
- Competition: https://www.kaggle.com/competitions/sweettv-tv-program-recommender
