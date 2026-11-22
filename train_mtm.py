"""NYAS BrainTECH / MTM: offline LEFT/RIGHT motor imagery classification.

Run in VS Code terminal: python train_mtm.py
For a short download/demo run: python train_mtm.py --quick

Uses public prerecorded EEG, not live mind reading or real robotic control.
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # Save plots as PNGs; no extra GUI setup on Mac
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.datasets import eegbci
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
DATA_FOLDER = ROOT / "data"    # Recordings saved here and reused on later runs
RESULTS_FOLDER = ROOT / "results"
CONFIDENCE_THRESHOLD = 0.75  # A demonstration threshold, NOT a safety guarantee

mne.set_log_level("ERROR")


def load_subject(subject, runs):
    """Download EEG data and create labeled epochs for a single participant.

    For runs 4, 8, 12 ONLY: T1 = imagined left fist; T2 = imagined right fist.
    Output X has shape [trials, EEG channels, time samples].
    """
    edf_paths = eegbci.load_data(
        subject,
        runs,
        path=DATA_FOLDER,
        update_path=False,
        verbose="ERROR",
    )
    arrays, labels = [], []

    for edf in edf_paths:
        raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
        eegbci.standardize(raw)
        raw.pick("eeg")
        raw.filter(8.0, 30.0, method="iir", verbose="ERROR")

        events, annotations = mne.events_from_annotations(raw, verbose="ERROR")
        event_id = {"LEFT": annotations["T1"], "RIGHT": annotations["T2"]}
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=1.0,
            tmax=3.0,
            baseline=None,
            preload=True,
            verbose="ERROR",
        )
        if len(epochs) == 0:
            raise RuntimeError(f"No LEFT/RIGHT epochs for subject {subject} in {edf}")
        x = epochs.get_data()
        y = np.where(epochs.events[:, 2] == event_id["LEFT"], 0, 1)
        arrays.append(x)
        labels.append(y)

    return np.concatenate(arrays), np.concatenate(labels)


def load_group(subjects, runs, group_name):
    data, targets = [], []
    print(f"\nLoading {group_name} subjects: {subjects}", flush=True)
    for subject in subjects:
        print(f"  Subject {subject}: downloading/loading...", flush=True)
        x, y = load_subject(subject, runs)
        print(f"    {len(y)} EEG trials: {sum(y == 0)} LEFT, {sum(y == 1)} RIGHT", flush=True)
        data.append(x)
        targets.append(y)
    return np.concatenate(data), np.concatenate(targets)


def pipeline(classifier):
    # CSP is fitted ONLY on the training set (avoids leakage from test subjects).
    return Pipeline([
        ("csp", CSP(n_components=4, reg="ledoit_wolf", log=True, norm_trace=False)),
        ("classifier", classifier),
    ])


def run(quick=False):
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

    if quick:
        # Smaller exploration with one run per person; do not treat as final study.
        train_subjects, val_subjects, test_subjects = [1, 2], [3], [4]
        runs = [4]
    else:
        # Separate PEOPLE for train / model selection / final evaluation.
        train_subjects, val_subjects, test_subjects = [1, 2, 3, 4, 5], [6], [7, 8]
        runs = [4, 8, 12]

    print("MTM LEFT/RIGHT EEG CLASSIFIER")
    print("Dataset: PhysioNet EEGMMIDB; labels: 0=LEFT, 1=RIGHT")
    print("Mode:", "QUICK (practice run)" if quick else "FULL (initial experiment)")
    print("First run requires internet to download EDF recordings.")

    x_train, y_train = load_group(train_subjects, runs, "TRAIN")
    x_val, y_val = load_group(val_subjects, runs, "VALIDATION")
    x_test, y_test = load_group(test_subjects, runs, "TEST")
    print("\nInput shapes (trials, channels, samples):")
    print("  Training:", x_train.shape)
    print("  Validation:", x_val.shape)
    print("  Test:", x_test.shape)

    models = {
        "Logistic Regression": pipeline(LogisticRegression(max_iter=1000)),
        "LDA": pipeline(LinearDiscriminantAnalysis()),
        "Linear SVM": pipeline(SVC(kernel="linear", probability=True, random_state=42)),
    }
    scores = []
    for name, model in models.items():
        print(f"\nTraining {name}...", flush=True)
        model.fit(x_train, y_train)
        prediction = model.predict(x_val)
        scores.append({
            "model": name,
            "validation_accuracy": float(accuracy_score(y_val, prediction)),
            "validation_balanced_accuracy": float(balanced_accuracy_score(y_val, prediction)),
        })

    model_scores = pd.DataFrame(scores)
    model_scores.to_csv(RESULTS_FOLDER / "model_comparison.csv", index=False)
    print("\nVALIDATION MODEL COMPARISON:")
    print(model_scores.round(3).to_string(index=False))

    best_name = model_scores.sort_values(
        "validation_balanced_accuracy", ascending=False, kind="stable"
    ).iloc[0]["model"]
    best = models[best_name]
    print("\nSelected using VALIDATION subjects:", best_name)

    test_pred = best.predict(x_test)
    test_acc = float(accuracy_score(y_test, test_pred))
    test_bal_acc = float(balanced_accuracy_score(y_test, test_pred))
    print(f"Test accuracy: {test_acc:.1%}")
    print(f"Test balanced accuracy: {test_bal_acc:.1%}")
    print(classification_report(
        y_test, test_pred, labels=[0, 1], target_names=["LEFT", "RIGHT"],
        zero_division=0,
    ))

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, test_pred, labels=[0, 1], display_labels=["LEFT", "RIGHT"],
        ax=ax, cmap="Blues", colorbar=False,
    )
    ax.set_title("MTM: Test participants")
    fig.tight_layout()
    fig.savefig(RESULTS_FOLDER / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    probs = best.predict_proba(x_test)
    predicted_labels = best.classes_[np.argmax(probs, axis=1)]
    confidence = probs.max(axis=1)
    accepted = confidence >= CONFIDENCE_THRESHOLD
    action = np.where(
        accepted,
        np.where(predicted_labels == 0, "LEFT", "RIGHT"),
        "NO ACTION",
    )
    incorrect_all = int(np.sum(predicted_labels != y_test))
    incorrect_accepted = int(np.sum(accepted & (predicted_labels != y_test)))
    coverage = float(np.mean(accepted))
    accepted_accuracy = (
        float(np.mean(predicted_labels[accepted] == y_test[accepted]))
        if accepted.any() else None
    )

    print(f"\nMTM DECISION LAYER (illustrative threshold {CONFIDENCE_THRESHOLD:.2f}):")
    print(f"  Command coverage: {coverage:.1%}")
    print(f"  Incorrect commands without rejection: {incorrect_all}")
    print(f"  Incorrect ACCEPTED commands: {incorrect_accepted}")
    print(f"  No-action decisions: {int(np.sum(~accepted))}")
    print("  Accuracy of accepted predictions:",
          f"{accepted_accuracy:.1%}" if accepted_accuracy is not None else "N/A")
    print("  Note: model confidence may be miscalibrated. These are NOT safety guarantees.")

    decisions = pd.DataFrame({
        "actual": np.where(y_test == 0, "LEFT", "RIGHT"),
        "predicted": np.where(predicted_labels == 0, "LEFT", "RIGHT"),
        "confidence": confidence.round(3),
        "mtm_action": action,
    })
    decisions.to_csv(RESULTS_FOLDER / "example_predictions.csv", index=False)

    # This plot is EXPLORATORY: we do not choose a threshold using test data.
    thresholds = np.arange(0.50, 0.96, 0.05)
    points = []
    for threshold in thresholds:
        selected = confidence >= threshold
        points.append({
            "threshold": round(float(threshold), 2),
            "coverage": float(np.mean(selected)),
            "accepted_accuracy": (
                float(np.mean(predicted_labels[selected] == y_test[selected]))
                if selected.any() else np.nan
            ),
            "incorrect_accepted_commands": int(
                np.sum(selected & (predicted_labels != y_test))
            ),
        })
    tradeoff = pd.DataFrame(points)
    tradeoff.to_csv(RESULTS_FOLDER / "confidence_tradeoff_exploratory.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tradeoff["threshold"], tradeoff["coverage"], marker="o", label="Coverage")
    ax.plot(tradeoff["threshold"], tradeoff["accepted_accuracy"], marker="s", label="Accepted accuracy")
    ax.set(xlabel="Confidence threshold", ylabel="Fraction", ylim=(0, 1.05),
           title="MTM: exploratory accuracy vs coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_FOLDER / "confidence_tradeoff.png", dpi=160)
    plt.close(fig)

    joblib.dump({
        "pipeline": best,
        "class_labels": {0: "LEFT", 1: "RIGHT"},
        "train_subjects": train_subjects,
        "validation_subjects": val_subjects,
        "test_subjects": test_subjects,
        "runs": runs,
        "bandpass_hz": [8.0, 30.0],
        "epoch_tmin_tmax_seconds": [1.0, 3.0],
        "confidence_threshold_demo": CONFIDENCE_THRESHOLD,
    }, RESULTS_FOLDER / "mtm_model.joblib")

    summary = {
        "mode": "quick" if quick else "full",
        "selected_model": best_name,
        "test_accuracy": test_acc,
        "test_balanced_accuracy": test_bal_acc,
        "command_coverage": coverage,
        "accepted_command_accuracy": accepted_accuracy,
        "incorrect_commands_without_abstaining": incorrect_all,
        "incorrect_accepted_commands": incorrect_accepted,
        "threshold": CONFIDENCE_THRESHOLD,
        "important_note": "Experimental offline results only; no guarantee of real-world robotic safety.",
    }
    (RESULTS_FOLDER / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("\nDONE! See saved files in:", RESULTS_FOLDER)
    print("  model_comparison.csv, example_predictions.csv, confusion_matrix.png,")
    print("  confidence_tradeoff.png, confidence_tradeoff_exploratory.csv,")
    print("  summary.json, mtm_model.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MTM EEG LEFT vs RIGHT classifier")
    parser.add_argument(
        "--quick", action="store_true",
        help="Practice with 4 participants and run 4 only (fewer downloads).",
    )
    args = parser.parse_args()
    try:
        run(quick=args.quick)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        print("Check your internet connection and installed packages, then retry.")
        raise
