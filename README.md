# NYAS MTM — EEG motor imagery classifier for VS Code (macOS)

This is a **prerecorded EEG experiment**, not a live EEG headset or a physical robot controller. It predicts whether a participant was instructed to imagine moving the LEFT or RIGHT hand. The saved model is experimental, not medically or physically safe to drive real equipment.

## 1. Open the folder in Visual Studio Code

Unzip `NYAS_MTM_VSCode.zip`. In VS Code choose **File → Open Folder…** and select **NYAS_MTM_VSCode**. Install the **Python** extension from Microsoft if you do not already have it.

## 2. Set up Python on your Mac

Choose **Terminal → New Terminal** (in VS Code). Run these one line at a time:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `python3` is missing, install Python from https://www.python.org/downloads/ and restart VS Code. Python 3.11, 3.12, or 3.13 works with recent versions of these libraries.

Press **Cmd+Shift+P**, run **Python: Select Interpreter**, and choose the one in this folder's `.venv`.

## 3. Train a quick prototype

```bash
python train_mtm.py --quick
```

- Automatically downloads EEG trials from **PhysioNet EEG Motor Movement/Imagery**.
- Uses runs 4 with left/right motor imagery, with separate train, validation, and test subjects.
- Trains **CSP + Logistic Regression, CSP + LDA, CSP + linear SVM**.
- Selects the model by *validation* balanced accuracy and tests it on previously unseen subjects.
- Adds a demonstration **NO ACTION** rule when classifier confidence is below 75%.
- The first download needs internet access; recordings are cached in `data/`.

Quick results are practice only; a small dataset and one split cannot show general performance.

## 4. Run a larger experiment

```bash
python train_mtm.py
```

Trains on subjects 1–5, selects the classifier using participant 6, and evaluates on unseen participants 7–8. Uses motor imagery runs 4, 8, and 12. Downloads additional data automatically.

## 5. Find your results

After a successful run, look in the `results/` folder:

- `model_comparison.csv` — Logistic Regression vs LDA vs SVM, using validation subjects
- `summary.json` — test accuracy, balanced accuracy, coverage, incorrect accepted commands
- `confusion_matrix.png` — LEFT/RIGHT confusion matrix
- `confidence_tradeoff.png` — exploratory coverage vs accepted accuracy
- `confidence_tradeoff_exploratory.csv` — exploratory threshold table (not for choosing threshold on test data)
- `example_predictions.csv` — actual labels, predictions, and example virtual actions
- `mtm_model.joblib` — saved CSP + classifier pipeline

**Research caveats:** EEG classification can be difficult between different participants; results may be near chance. The classifier's reported probabilities may be poorly calibrated. Never claim 75% is a proven safety threshold. Always keep the final test participants separate when selecting models and thresholds. The model file does not include EDF preprocessing, so new recordings need the same channel ordering, 8–30 Hz filtering, and 1–3 s epochs. Only load trusted `.joblib` files.

## If it fails

- `ModuleNotFoundError`: confirm the `.venv` interpreter and rerun `python -m pip install -r requirements.txt`.
- `python: command not found`: run `source .venv/bin/activate` in the new terminal.
- Download or SSL error: check connection and retry; previously downloaded files remain cached.
- Wrong interpretation of labels: this script is **only for runs 4, 8, 12**; other PhysioNet runs use different tasks.

Dataset: https://physionet.org/content/eegmmidb/1.0.0/

MNE documentation: https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html
