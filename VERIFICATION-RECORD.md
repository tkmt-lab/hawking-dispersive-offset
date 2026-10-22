# Verification record: hawking-dispersive-offset 1.0.0

This record carries process and provenance details kept out of the article (version 1.2). The article states the mathematics and the present scope; this file states what was fixed before which result, which computations are independent of which, which AI readings took place, and what the packet checks. The packet was assembled and checked as a local release candidate; publication is a separate step taken by the author.

## Status of the main result

The formula b0 = g'''(0)/(12 kappa) (article Eq. (19)) is the result of a formal asymptotic calculation conditional on the assumptions A1–A3 of Section 7.1. Stage 1 (reduction of the omega -> 0 limit of T_ratio to a zero-frequency problem) is an identity under A1 and A2; Stage 2 (the local expansion) is exact rational algebra under A3. There is no proof of A1–A3, no remainder bound, no Lean formalization and no human expert review. The next coefficient b1 has the same status.

## Computations and their independence

1. **Two scattering methods for the curvature comparison.** A first method integrates a solution basis in x with an adaptive eighth-order integrator; a second solves the boundary-value problem with multi-element Chebyshev collocation (`src/hawking_bvp.py`). The second was written without reading the first one's code and compared only with its output tables. The comparison covers the symmetric tanh family only: 96 table entries (93 numerical, 64 distinct points; 3 above-cutoff), stored in `results/compare_first_impl.json`. The first method's code and tables are not distributed; the comparison rows are. All other profiles use the second method only.
2. **The low-frequency stage-wise checks** (`b0/check_steps.py`, `b0/zero_mode.py`) use the second scattering method and a separate integration of the zero-frequency equation (adaptive DOP853, rtol 1e-12). Agreement of these two is a check of Stage 1 within one code base, not an independent implementation.
3. **The independent numerical check** (`independent/`) is a third scattering code, written from a written problem definition only (reproduced in `independent/README.md`), by a separate AI-assisted session that was instructed not to open the main code, data, article or predictions and records that it did not. It shares no code with `src/` or `b0/`. Its own variants (refined, baseline, DOP853, collocation of the original equation) share its flow and dispersion-root routines.

## What was fixed before which result

- **Curvature, held-out profiles B1, B2.** Coefficient predictions (`results/c_predictions_blind_stage2.json`) were written before their scans.
- **Prediction-first test of b0, families B3, B4** (article Section 7.5). The profiles, the predicted b0, b1 and delta (`b0/blind_predictions.json`), the three success criteria, the cutoffs and basis of criterion 1, the frequencies and omega -> 0 extrapolation bases, the boundary-value discretization and the integrator tolerance were committed to the research repository on 21 October 2026, 22:40 JST, before any zero-frequency or scattering computation for these families. The results first appear in a commit at 22:45 JST; the prediction files are unchanged between the two. The cutoff list actually run (10, 20, 40, 80, 160) was set at run time. Chosen later, with results available: the basis of criterion 3 (`b0/analyze_b0.py`) and the refined b1 fits (`b0/b1_check.py`). A first parameter choice for B3 was rejected before any computation because its right asymptotic velocity was positive. These are prediction-first tests by the same investigators, not blinded experiments.
- **Independent check, flows T05, B4, N1, N2.** The predicted b0 values (`independent/predictions-sealed.json`) were committed on 21 October 2026, 23:46 JST, before the independent computation was requested; they were not given to it. The independent results were committed on 22 October 2026, 00:02 JST, and only then compared with the predictions. N1 and N2 were chosen for this check and appear nowhere else. The public copy of the prediction file differs from the committed one only in two descriptive strings (internal names removed); the numbers are identical. Original SHA256: 4b408c6b0af3210d55cbe86dd1e77e8e00c8bfb67267e74ce5de821129450384 (see `source-derivatives.json`).

## AI readings (not human peer review)

- **Derivation.** A fresh AI session given only the study report and the derivation note returned "correct with gaps": the zero-frequency reduction is consistent under the matching assumptions, and it reproduced the eps^3 coefficient with its own exact rational calculation. Its points (normalization of psi, global connection, travelling-wave normalization, hypotheses, the linear-velocity check, the order at which G0 enters, the argument of g) were addressed in article version 1.
- **Article, version 1.** A second fresh AI session acting as referee returned "acceptable with changes" with 12 items. It reproduced b0, b1 and T(0) through sixth order with exact rational arithmetic and found the sign convention of the Wronskian and the label of the positive-wave-number Airy component inconsistent; this did not change T(0), b0 or b1. Version 1.1 addresses all items (Wronskian convention, b1 printed in Appendix A, the frozen protocol of the prediction-first test, frequency-window sensitivity, credit to Corley–Jacobson and Unruh–Schützhold, definitions).
- **Literature.** Two limited literature checks place the method (Coutant–Weinfurtner zero-frequency procedure) as known and the explicit coefficient for general flows as a "medium, provisional" novelty: Himemoto–Tanaka allow terms of this type with unevaluated coefficients; no earlier statement of the same coefficient was found in the sources examined, but one cannot be excluded. A 2026 paper whose full text was not freely available was not read. The second check (October 2026) recommended citing the 2025 tunneling paper of Del Porro, Liberati and Schneider separately from their 2026 paper; version 1.2 does so (bibliographic data checked on the free arXiv page and at Crossref).

## Article versions

- 0: curvature coefficient c (phase-integral formula vs full scattering over 19 profiles), b0 as a numerical observation.
- 1: formal derivation of b0 (Section 7), stage-wise checks, prediction-first test.
- 1.1: response to the AI referee (above).
- 1.2 (this release): citation of the 2025 tunneling paper; independent numerical check (Section 3 table row and paragraph, Eq. (12)); one pointer sentence in Section 7.5; a clause in the conclusion; version line and AI-use paragraph worded for circulation. All other text, every table and figure, and every number are unchanged from 1.1.

While preparing 1.2 one transcription was corrected before it reached any article: an internal comparison note gave the B4 difference as -5.9e-9, computed from the rounded independent value; from the stored values it is -5.8e-9 (1.1336017720 - 1.1336017778). The article prints -5.8e-9.

## Independent check: numbers and qualifications

Differences of the independent central b0 from the predictions: T05 -4.1e-8, B4 -5.8e-9, N1 +3.1e-8, N2 -1.1e-7. Over 42 extrapolation choices the range contains the prediction for B4, N1, N2; for T05 (exact b0 = 0) the range [-1.24e-6, -5.7e-9] misses 0 by 5.7e-9, less than the largest change of the T05 central value between solver variants (7.9e-9). Ranges are sensitivities of the extrapolation formulas, not confidence intervals. Re-running the 80 central solves with NumPy 2.0.2 / SciPy 1.13.1 (instead of 2.5.4 / 1.18.1) changes T/T_H - 1 by at most 2.5e-12 and the extrapolated b0 by up to 3.7e-8 (N2); the largest difference from a prediction is then 1.4e-7 (N2). The agreement is therefore at the 1e-7 level, which is the floor set by double precision amplified by Lambda^2 and the two extrapolations. It is numerical evidence for Eq. (19) on four flows, not a proof of A1–A3.

## Present packet checks

- `check_packet.py`: closed file list and SHA256SUMS, claims.json evidence fingerprints and fixed scope, article labels/references/citations, version and repository fields, PDF validation record, absence of private paths and addresses (`publication_privacy.py`: file names, raw text, UTF-16 views, PDF text, metadata and streams). It ignores `.git`.
- `reproduce.py` (about 35 s): exact local expansion and g-form (byte-identical output), zero-frequency formula for 7 profiles at all 40 stored cutoffs (identical to the stored values on the preparation machine; tolerance relative 1e-9), two scattering limits, the curvature and b0 analyses on the stored scans (numbers within relative 1e-8 or absolute 1e-10; the absolute bound covers fit residuals and differences from fits, whose last digits differ between platforms by about 3e-12), all TeX tables byte-identical and figure data within the same bounds, the independent comparison, and a replay of the independent code's 80 central solves.
- Every page of the PDF was rendered and inspected (`pdf-validation.json`); no overfull boxes, no undefined references.
- Before release the packet was also scanned with a non-public list of internal identifiers (not distributed) and checked in a fresh git clone with the same commands as the CI workflow.
- No new scattering scan, Lean run or literature search was made for the packet. Bounded privacy scans are not a guarantee against every unknown identifier.

## Files and labels

- Public copies whose text differs from the research originals (comment headers, one environment variable `HIGH_LAM`, the JSON field `b0_inf_frequency_fit`, the exhibit paths of the article source) are listed with both hashes in `source-derivatives.json`. Numbers are unchanged.
- Not distributed: the first scattering method's code and printed tables, derivation notes, the AI reviews, scan logs and research-process records.
- Profile keys: see AGENTS.md and `exhibits/all-profiles.md`.
