"""Shared training-health diagnostics - used by BOTH train.py (the live
console warning during a real run) and the dashboard API (so a symbol's
training history can be flagged without needing a live process). One
definition, not two, so the two surfaces can't silently drift apart.

Checks several DISTINCT real failure modes, not just one signal - a training
run can go wrong in more ways than overfitting alone:
  - overfitting: train accuracy pulling away from validation, sustained
  - lost_edge: validation has stopped beating a 50% coin-flip, sustained
  - numerical_instability: NaN/Inf loss - an acute, single-epoch problem,
    not something that needs to "sustain" to matter
  - not_learning: training accuracy itself stuck near 50% for a long
    stretch - the model isn't finding any signal at all, a different
    problem from overfitting (which requires training accuracy to be
    doing WELL, just not generalizing)

None of these fire on a single noisy epoch alone (except numerical
instability, which is never noise) - see this project's own history of a
real epoch-4 gap spike that partly recovered over the next two epochs;
flagging on that single epoch would have been a false alarm.
"""
import math

GAP_WARNING_THRESHOLD = 0.10   # 10 percentage points - a real, not noise-level, gap
SUSTAINED_EPOCHS = 3           # require the LAST N epochs' average to clear
                                # a threshold, not one noisy epoch
MIN_EPOCHS_SINCE_BEST = 5      # also require validation to have genuinely
                                # stalled (not just had one worse epoch)
NOT_LEARNING_EPOCHS = 8        # how many epochs of near-coin-flip TRAINING
                                # accuracy before concluding it's stuck, not just slow
NOT_LEARNING_BAND = 0.02       # "near 50%" = within +/-2 points


def annotate_training_health(rows: list[dict]) -> list[dict]:
    """rows: edge_tracker CSV rows in epoch order, each with at least
    'val_dir_acc', 'train_dir_acc', 'epochs_since_best', 'val_loss' (native
    types, not strings). Returns the SAME rows with these fields added:
      - train_val_gap: this epoch's train accuracy minus val accuracy
      - overfitting_risk: bool
      - lost_edge_risk: bool
      - numerical_instability: bool (NaN/Inf loss THIS epoch)
      - not_learning_risk: bool
      - issues: list[str] - human-readable names of whichever fired, for
        display; empty list means nothing detected
    """
    gaps = []
    train_accs = []
    significant_flags = []
    annotated = []

    for row in rows:
        train_acc = float(row["train_dir_acc"])
        val_acc = float(row["val_dir_acc"])
        val_loss = float(row.get("val_loss", 0.0))
        significant = str(row.get("significant_edge", "True")) == "True"

        gap = train_acc - val_acc
        gaps.append(gap)
        train_accs.append(train_acc)
        significant_flags.append(significant)

        recent_gaps = gaps[-SUSTAINED_EPOCHS:]
        sustained_gap = (len(recent_gaps) >= SUSTAINED_EPOCHS
                          and (sum(recent_gaps) / len(recent_gaps)) > GAP_WARNING_THRESHOLD)
        stalled = int(row["epochs_since_best"]) >= MIN_EPOCHS_SINCE_BEST
        overfitting_risk = bool(sustained_gap and stalled)

        recent_significant = significant_flags[-SUSTAINED_EPOCHS:]
        lost_edge_risk = bool(len(recent_significant) >= SUSTAINED_EPOCHS and not any(recent_significant))

        numerical_instability = bool(math.isnan(val_loss) or math.isinf(val_loss)
                                      or math.isnan(train_acc) or math.isinf(train_acc))

        recent_train = train_accs[-NOT_LEARNING_EPOCHS:]
        not_learning_risk = bool(
            len(recent_train) >= NOT_LEARNING_EPOCHS
            and all(abs(t - 0.5) < NOT_LEARNING_BAND for t in recent_train)
        )

        issues = []
        if numerical_instability:
            issues.append("Numerical instability (NaN/Inf loss)")
        if overfitting_risk:
            issues.append("Overfitting (training pulling away from validation)")
        if lost_edge_risk:
            issues.append("Lost statistical edge (validation at/below coin-flip)")
        if not_learning_risk:
            issues.append("Not learning (training accuracy stuck near 50%)")

        row = dict(row)
        row.update({
            "train_val_gap": gap,
            "overfitting_risk": overfitting_risk,
            "lost_edge_risk": lost_edge_risk,
            "numerical_instability": numerical_instability,
            "not_learning_risk": not_learning_risk,
            "issues": issues,
        })
        annotated.append(row)

    return annotated
