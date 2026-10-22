# Synthetic evaluation results

Reproduced locally on Python 3.12 with the default seed 42, 350 training iterations, and a chronological 60/20/20 split. All results refer exclusively to generated data and a fictional decision-cost model.

| Measurement | Value |
| --- | ---: |
| Train / validation / test rows | 1,200 / 400 / 400 |
| Validation-selected approval threshold | 0.15 |
| Held-out ROC-AUC | 0.74894 |
| Held-out Brier score | 0.19749 |
| Simulated approval rate | 19.5% |
| Simulated late rate among approved cases | 15.38% |
| Shifted buffer-ratio PSI | 2.46626 |
| Demo drift alert | Triggered at illustrative threshold 0.2 |
| Unit tests passed | 8 |

The deliberately shifted population has smaller balance buffers. PSI detects distribution change, not repayment deterioration or fraud. The 10-case ledger demo records eight declined and two repaid synthetic advances; it does not move money.

These metrics do not demonstrate real-world underwriting performance, fairness, profitability, or a ready-to-deploy lending policy. Threshold selection depends on explicitly invented costs. Reproduce with `python advance_lab.py --output demo-output`.
