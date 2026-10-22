"""Synthetic cash-advance ML sandbox. No real customers, bank integration or credit decisions."""
from __future__ import annotations
import argparse
import json
import math
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path

FIELDS = {"monthly_income", "income_variability", "balance_buffer", "overdrafts_90d", "requested_amount"}


def features(application: dict) -> list[float]:
    if set(application) != FIELDS:
        raise ValueError("Missing or unexpected fields; outcome data must never enter scoring")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in application.values()):
        raise ValueError("Features must be finite numbers")
    income = application["monthly_income"]
    variability = application["income_variability"]
    overdrafts = application["overdrafts_90d"]
    amount = application["requested_amount"]
    if income <= 0 or not 0 <= variability <= 1 or overdrafts < 0 or overdrafts != int(overdrafts):
        raise ValueError("Invalid feature range")
    if not 0 < amount <= 500:
        raise ValueError("Sandbox advance amount must be in (0, 500]")
    return [variability, application["balance_buffer"] / income, overdrafts, amount / income]


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-max(-35, min(35, value))))


@dataclass
class Row:
    sequence: int
    application: dict
    late: int  # Synthetic future repayment outcome, excluded from feature extraction.


def generate(count=2000, seed=42, shifted=False) -> list[Row]:
    rng = random.Random(seed)
    rows = []
    for sequence in range(count):
        income = rng.uniform(1500, 6500)
        application = {
            "monthly_income": income,
            "income_variability": rng.uniform(0, 0.8),
            "balance_buffer": rng.uniform(-100, 1800) * (0.25 if shifted else 1),
            "overdrafts_90d": rng.randrange(6),
            "requested_amount": rng.uniform(50, 500),
        }
        x = features(application)
        probability = sigmoid(-2.1 + 2.5*x[0] - 3*x[1] + 0.35*x[2] + 4*x[3])
        rows.append(Row(sequence, application, int(rng.random() < probability)))
    return rows


def chronological_split(rows):
    if len(rows) < 20 or any(a.sequence >= b.sequence for a, b in zip(rows, rows[1:])):
        raise ValueError("Need at least 20 uniquely ordered observations")
    return rows[:int(len(rows)*0.6)], rows[int(len(rows)*0.6):int(len(rows)*0.8)], rows[int(len(rows)*0.8):]


class RiskModel:
    """Logistic regression with train-only scaling and full-batch gradient descent."""
    def fit(self, rows, iterations=350):
        x = [features(row.application) for row in rows]
        self.means = [sum(column)/len(x) for column in zip(*x)]
        self.scales = [max(1e-8, math.sqrt(sum((v-m)**2 for v in column)/len(x)))
                       for column, m in zip(zip(*x), self.means)]
        design = [[1] + self.standardize(values) for values in x]
        self.weights = [0.0]*5
        for _ in range(iterations):
            gradient = [0.0]*5
            for values, row in zip(design, rows):
                error = sigmoid(sum(w*v for w, v in zip(self.weights, values))) - row.late
                for j, value in enumerate(values):
                    gradient[j] += error*value
            self.weights = [w - 0.15*g/len(rows) for w, g in zip(self.weights, gradient)]
        return self

    def standardize(self, values):
        return [(v-m)/s for v, m, s in zip(values, self.means, self.scales)]

    def predict(self, application):
        values = [1] + self.standardize(features(application))
        return sigmoid(sum(w*v for w, v in zip(self.weights, values)))


def choose_threshold(model, validation):
    # Toy costs, not a real lender's economics. Tune on validation only.
    def cost(threshold):
        return sum(5 if model.predict(r.application) < threshold and r.late else
                   1 if model.predict(r.application) >= threshold and not r.late else 0
                   for r in validation)
    return min((i/100 for i in range(5, 96, 5)), key=cost)


def evaluate(model, rows, threshold):
    scores = [model.predict(row.application) for row in rows]
    positives = [s for s, r in zip(scores, rows) if r.late]
    negatives = [s for s, r in zip(scores, rows) if not r.late]
    auc = (sum((p > n) + 0.5*(p == n) for p in positives for n in negatives)
           / (len(positives)*len(negatives))) if positives and negatives else None
    approved = [r for s, r in zip(scores, rows) if s < threshold]
    return {"rows": len(rows), "roc_auc": auc,
            "brier_score": sum((s-r.late)**2 for s, r in zip(scores, rows))/len(rows),
            "approval_rate": len(approved)/len(rows),
            "late_rate_among_approved": sum(r.late for r in approved)/len(approved) if approved else None}


def psi(reference, current, bins=10):
    """Smoothed population stability index using reference-defined quantile boundaries."""
    if not reference or not current:
        raise ValueError("PSI requires two nonempty populations")
    ordered = sorted(reference)
    boundaries = sorted(set(ordered[min(len(ordered)-1, int(len(ordered)*i/bins))]
                            for i in range(1, bins)))
    def distribution(values):
        counts = [0]*(len(boundaries)+1)
        for value in values:
            counts[sum(value > edge for edge in boundaries)] += 1
        return [(n+0.5)/(len(values)+0.5*len(counts)) for n in counts]
    return sum((b-a)*math.log(b/a) for a, b in zip(distribution(reference), distribution(current)))


class Ledger:
    """Local decision/repayment state machine. Never moves money."""
    def __init__(self, path=":memory:"):
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS advances (id TEXT PRIMARY KEY, amount REAL, risk REAL, status TEXT)")

    def decide(self, identifier, application, model, threshold):
        risk = model.predict(application)
        status = "approved" if risk < threshold else "declined"
        with self.db:
            self.db.execute("INSERT INTO advances VALUES (?, ?, ?, ?)",
                            (identifier, application["requested_amount"], risk, status))
        return status

    def repay(self, identifier):
        with self.db:
            result = self.db.execute("UPDATE advances SET status='repaid' WHERE id=? AND status='approved'", (identifier,))
            if result.rowcount != 1:
                raise ValueError("Only an approved outstanding advance can be repaid")

    def close(self):
        self.db.close()


def run(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    train, validation, test = chronological_split(generate())
    model = RiskModel().fit(train)
    threshold = choose_threshold(model, validation)
    shifted = generate(400, seed=99, shifted=True)
    buffer_psi = psi([features(r.application)[1] for r in train], [features(r.application)[1] for r in shifted])
    report = {"data": "synthetic only", "split": [len(train), len(validation), len(test)],
              "threshold_selected_on_validation": threshold,
              "held_out_evaluation": evaluate(model, test, threshold),
              "shifted_buffer_ratio_psi": buffer_psi,
              "drift_alert_demo_threshold": 0.2, "drift_alert": buffer_psi > 0.2}
    ledger = Ledger(output / "demo.sqlite")
    try:
        # Reset only known synthetic demo IDs, so rerunning the example is repeatable.
        with ledger.db:
            ledger.db.execute("DELETE FROM advances WHERE id LIKE 'synthetic-demo-%'")
        for i, row in enumerate(test[:10]):
            status = ledger.decide(f"synthetic-demo-{i}", row.application, model, threshold)
            if status == "approved":
                ledger.repay(f"synthetic-demo-{i}")
        report["ledger_states"] = dict(ledger.db.execute("SELECT status, COUNT(*) FROM advances GROUP BY status"))
    finally:
        ledger.close()
    (output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
