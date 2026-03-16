## AI-Driven Antenna Fault Detection & Performance Monitoring

This repository is a **mini end-to-end ML project** for antenna fault detection and antenna performance monitoring. It contains:

- A **dataset** (`antenna_fault.csv`) with antenna geometry/material/performance features and multiple labels.
- **Model training notebooks** to train and evaluate classifiers.
- Pre-trained **saved models** (`.pkl`) for 4 prediction targets.
- A **Tkinter desktop GUI** (`tkinter_interface.py`) that loads the saved models and predicts from user-entered features.

The core idea is to use ML to learn patterns in simulated/measured antenna parameters (e.g., S11, VSWR, gain, efficiency, bandwidth) and then:

- **Detect whether an antenna is Fault vs Normal**
- **Classify the fault type** (separately for WiFi and Bluetooth modes)

---

## Repository structure (what’s inside)

Top-level files in this repo:

- **`antenna_fault.csv`**: the dataset (features + 4 labels).
- **`antennaclassification.ipynb`**: exploratory/training notebook (loads CSV, trains models per target, prints metrics, plots confusion matrices + feature importances).
- **`savepkl.ipynb`**: training + **exports models to `.pkl`** files using `joblib.dump(...)`.
- **`tkinter_interface.py`**: desktop UI to enter features and run predictions using the `.pkl` models.
- **`model_WiFi_Fault.pkl`**, **`model_BT_Fault.pkl`**, **`model_WiFi_Status.pkl`**, **`model_BT_Status.pkl`**: saved scikit-learn models produced by the notebook.

---

## Dataset: `antenna_fault.csv`

### Columns
The CSV contains numeric feature columns and 4 label columns.

**Features** (used by the GUI and by the notebooks):

- `Length`, `Width`, `Height`
- `Permittivity`, `Conductivity`, `epsilon_r`
- `Bend`, `Feed`
- `S11`, `VSWR`
- `Gain`, `Efficiency`, `Bandwidth`

**Targets / labels** (multi-task setup):

- `WiFi Fault` (fault category for WiFi)
- `BT Fault` (fault category for Bluetooth)
- `WiFi Status` (binary: Fault/Normal)
- `BT Status` (binary: Fault/Normal)

### Label classes (as used by the GUI)
The GUI includes fixed class lists (mapping model integer output → label string).

**`WiFi Fault` classes**

- `Bending`
- `Body_Effect`
- `Conductivity_Degradation`
- `Cracks`
- `Humidity_Sweat`
- `No_Fault`
- `Rupture_Coupure`
- `Strong_Flexion`

**`BT Fault` classes**

- `Bending`
- `Body_Effect`
- `Conductivity_Degradation`
- `Coupure`
- `Cracks`
- `Humidity_or_Sweat`
- `No_Fault`
- `Rupture`
- `Strong_Flexion`

**`WiFi Status` / `BT Status` classes**

- `Fault`
- `Normal`

---

## Modeling approach (what the notebooks do)

Both notebooks implement the same general pipeline:

- Load `antenna_fault.csv` with pandas
- Set the input matrix \(X\) to all non-target columns
- For each target in:
  - `WiFi Fault`, `BT Fault`, `WiFi Status`, `BT Status`
  - Encode target labels using `LabelEncoder`
  - Split train/test using `train_test_split(test_size=0.2, random_state=42)`
  - Train a **`RandomForestClassifier(n_estimators=100, random_state=42)`**
  - Evaluate with:
    - `accuracy_score`
    - `classification_report`
    - `confusion_matrix` (visualized with seaborn heatmap)
  - Display feature importances (`model.feature_importances_`)

### Saved artifacts
`savepkl.ipynb` saves a model per target as:

- `model_WiFi_Fault.pkl`
- `model_BT_Fault.pkl`
- `model_WiFi_Status.pkl`
- `model_BT_Status.pkl`

These are loaded by the GUI using `joblib.load(...)`.

---

## Tkinter GUI app: `tkinter_interface.py`

The GUI is a simple desktop app that:

- Lets you choose which **target model** to use:
  - WiFi Fault / BT Fault / WiFi Status / BT Status
- Lets you enter numeric values for the 13 features:
  - `Length`, `Width`, `Height`, `Permittivity`, `Conductivity`, `Bend`, `Feed`, `S11`, `VSWR`, `Gain`, `Efficiency`, `Bandwidth`, `epsilon_r`
- Loads the corresponding `.pkl` model and returns a predicted class name.

### Run the GUI
From the repo folder:

```bash
python tkinter_interface.py
```

Make sure the `.pkl` files are in the **same directory** as `tkinter_interface.py` (the script expects exactly those filenames).

---

## Environment / dependencies
There is no `requirements.txt` in this repo, but the code uses:

- Python 3.x
- `numpy`
- `pandas`
- `scikit-learn`
- `joblib`
- `matplotlib` / `seaborn` (for notebook plots)
- `tkinter` (GUI; included with most standard Python installations on Windows)

Typical install (if needed):

```bash
pip install numpy pandas scikit-learn joblib matplotlib seaborn
```

---

## How to reproduce training (notebooks)

- Open and run:
  - `antennaclassification.ipynb` to train + evaluate + visualize
  - `savepkl.ipynb` to train + export `.pkl` models

Notes:

- The notebooks loop over the 4 targets and train **one model per target**.
- `antennaclassification.ipynb` includes a Kaggle-style path in one cell; `savepkl.ipynb` uses the local file `antenna_fault.csv`.

---

## Project workflow (high-level)

1. **Antenna simulation / measurement** (e.g., CST or similar tools)
2. **Fault injection** (e.g., cracks, bending, rupture, conductivity degradation, humidity/sweat, etc.)
3. **Feature extraction** (S11, impedance-related metrics, gain/efficiency/bandwidth, geometry/material parameters)
4. **Dataset creation** (`antenna_fault.csv`)
5. **Model training** (RandomForest per target)
6. **Deployment** (Tkinter GUI loads `.pkl` models for prediction)

---

## Limitations / assumptions (current repo state)

- The GUI assumes the model outputs match the **hard-coded class order** in `CLASS_MAPPINGS`. If training label order changes, the mapping must be updated.
- The saved `.pkl` models embed preprocessing decisions from training (here: label encoding for the target; features are assumed numeric).
- The repo currently includes models but does not include a pinned dependency file (`requirements.txt`).
