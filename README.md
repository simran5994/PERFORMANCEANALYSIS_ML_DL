````markdown
# Performance Analysis of ML and DL Techniques for Intrusion Detection

This repository contains the code used for the intrusion detection experiments conducted using the ToN-IoT network dataset.

## Getting Started

### 1. Requirements

The code was developed using Python 3.10 and Conda.

Required Python packages:

- tensorflow
- shap
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- openpyxl

Activate the Conda environment:

```bash
conda activate ids
````

If the environment does not already exist, install the required packages in a new Conda environment before running the code.

### 2. Dataset

Place the following dataset file in the project directory:

```text
train_test_network.xlsx
```

The expected structure is:

```text
PERFORMANCEANALYSIS_ML_DL/
│
├── train_test_network.xlsx
│
└── ids/
    ├── run_all.py
    └── ...
```

### 3. Verify the Installation

Navigate to the `ids` directory:

```bash
cd ids
```

Check that TensorFlow and SHAP are installed correctly:

```bash
python -c "import tensorflow, shap; print('BOTH OK')"
```

Expected output:

```text
BOTH OK
```

### 4. Run the Experiment

From the `ids` directory, run:

```bash
python run_all.py > run_final.txt 2>&1
```

On macOS, `caffeinate` can optionally be used to prevent the computer from sleeping during execution:

```bash
caffeinate -i python run_all.py > run_final.txt 2>&1
```

The script runs the complete experimental pipeline and saves the generated results and figures in the `results` directory.

### 5. Check the Output

After the run finishes, check the log using:

```bash
grep -E "COMPLETE|collapsed" run_final.txt
```

A successful run should end with a completion message similar to:

```text
COMPLETE in XX.X minutes
```

During LSTM cross-validation, the following message may also appear:

```text
[LSTM fold 4 collapsed to one class - retraining once]
```

This indicates that the affected fold was automatically retrained once.

The generated results will be available under:

```text
ids/results/
```

including the result tables (`.csv`) and generated figures (`.png`) used for the analysis.

## Expected Result

A successful execution should produce the `results` directory, the generated result files and figures, and a final `COMPLETE` message in `run_final.txt`.
