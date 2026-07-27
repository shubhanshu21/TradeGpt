"""Unit tests for src/core/training_diagnostics.py - the shared health-check
logic used by BOTH train.py's live console warning and the dashboard API."""
from core.training_diagnostics import annotate_training_health


def _row(train_acc, val_acc, epochs_since_best, val_loss=1.0, significant=True):
    return {"train_dir_acc": train_acc, "val_dir_acc": val_acc,
            "epochs_since_best": epochs_since_best, "val_loss": val_loss,
            "significant_edge": str(significant)}


def test_no_issues_on_healthy_run():
    rows = [_row(0.52, 0.53, 0), _row(0.53, 0.54, 0), _row(0.54, 0.545, 0)]
    annotated = annotate_training_health(rows)
    assert all(not r["issues"] for r in annotated)


def test_single_noisy_epoch_does_not_trigger_overfitting():
    # Real project history: epoch 4 spiked to a large gap, then partly
    # recovered over the next two epochs - a single bad epoch must not
    # trigger a false alarm.
    rows = [_row(0.52, 0.54, 0), _row(0.53, 0.535, 0), _row(0.68, 0.51, 1), _row(0.60, 0.52, 0)]
    annotated = annotate_training_health(rows)
    assert not annotated[-1]["overfitting_risk"]


def test_sustained_growing_gap_with_stalled_validation_triggers_overfitting():
    rows = [
        _row(0.52, 0.53, 0),
        _row(0.60, 0.51, 1),
        _row(0.65, 0.50, 2),
        _row(0.70, 0.49, 3),
        _row(0.75, 0.48, 4),
        _row(0.78, 0.47, 5),
    ]
    annotated = annotate_training_health(rows)
    assert annotated[-1]["overfitting_risk"]
    assert "Overfitting" in annotated[-1]["issues"][0] or any("Overfitting" in i for i in annotated[-1]["issues"])


def test_lost_edge_risk_requires_sustained_insignificance():
    rows = [_row(0.55, 0.54, 0, significant=True),
            _row(0.55, 0.49, 1, significant=False),
            _row(0.55, 0.48, 2, significant=False),
            _row(0.55, 0.47, 3, significant=False)]
    annotated = annotate_training_health(rows)
    assert annotated[-1]["lost_edge_risk"]


def test_numerical_instability_fires_immediately_no_sustain_needed():
    rows = [_row(0.55, 0.54, 0), _row(0.55, 0.54, 0, val_loss=float("nan"))]
    annotated = annotate_training_health(rows)
    assert not annotated[0]["numerical_instability"]
    assert annotated[1]["numerical_instability"]
    assert any("Numerical instability" in i for i in annotated[1]["issues"])


def test_not_learning_risk_detects_stuck_training_accuracy():
    rows = [_row(0.501, 0.50, i) for i in range(10)]
    annotated = annotate_training_health(rows)
    assert annotated[-1]["not_learning_risk"]


def test_not_learning_risk_does_not_fire_when_training_is_progressing():
    rows = [_row(0.50 + i * 0.03, 0.50, i) for i in range(10)]
    annotated = annotate_training_health(rows)
    assert not annotated[-1]["not_learning_risk"]
