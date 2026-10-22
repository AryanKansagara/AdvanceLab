# AdvanceLab: Cash-Advance ML Sandbox

A runnable educational simulation inspired by cash-advance products. It is not affiliated with Bree and does not reproduce Bree's model, rules, branding, or proprietary systems. It has no banking integration, real applicants, or money movement. Developed with AI assistance and verified with automated tests and a reproducible synthetic evaluation.

## Run

Python 3.10+; standard library only. No dependency installation or API keys required:

```sh
git clone https://github.com/AryanKansagara/AdvanceLab.git
cd AdvanceLab
python3 -m unittest -v
python3 advance_lab.py --output demo-output
```

The command creates a synthetic decision ledger and a JSON evaluation report in the chosen output directory. Demo reruns replace only the known synthetic demo IDs.

See [RESULTS.md](RESULTS.md) for the reference metrics and their limitations. GitHub Actions runs the tests and complete demo on Python 3.12.

## Implemented

- Reproducible generation of 2,000 synthetic applications and simulated future repayment outcomes.
- Strict feature schema, finite-number/range validation, and rejection of unexpected fields such as future repayment outcomes.
- Chronological 60/20/20 splits; train-only feature scaling; logistic regression implemented in Python.
- Validation-only approval-threshold selection using explicitly fictional decision costs, then held-out ROC-AUC, Brier score, approval rate, and simulated late-rate evaluation.
- Feature-distribution monitoring using a smoothed population stability index with train-defined bins, a shifted synthetic cohort, and a demo alert threshold.
- SQLite decision records with duplicate-ID protection and approved-to-repaid transition validation.
- Tests for reproducibility, temporal boundaries, schema/label leakage guards, numeric ranges, scaling, metrics, drift, and ledger state.

## Limits and interview talking points

This is an ML/backend simulation, not a full Bree clone or deployed lending product. The chronological index and known data-generating process make the example reproducible; synthetic test results do not establish real-world predictive performance. The model is not probability-calibrated separately, and no real credit-policy, fairness, fraud, affordability, regulatory, or security validation has been conducted. Drift indicates feature-distribution change, not proof of model degradation or fraud. A label-maturation policy and delayed-outcome monitoring would be needed with real repayments.

The train/validation/test boundaries prevent those partitions from directly sharing fitting steps, but they do not substitute for real as-of feature snapshots, repeated-customer splits, or production leakage audits. The ledger records risk and decisions, not personal data, and has no authentication or access-control layer. Do not expose it as a public service.

Next: compare logistic regression with LightGBM, add calibration curves and subgroup analysis, and simulate delayed labels. No LightGBM implementation is included in this version.
