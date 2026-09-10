# Artificial_Neural_Network

> A Streamlit app that loads the Heart Failure Clinical Records dataset and lets you train a configurable Artificial Neural Network (ANN) while comparing optimizers and visualizing training history and evaluation metrics.

## Project overview

This repository contains a Streamlit application that demonstrates building and training a compact feed-forward Artificial Neural Network for binary classification on the **Heart Failure Clinical Records** dataset (predicting `DEATH_EVENT`). The app focuses on:

* letting you select different optimizers and learning rates,
* tracking training and validation loss/accuracy,
* visualizing training curves and classification reports,
* measuring final test performance and training time.

The app is ideal for experimentation, teaching, and quick optimizer comparisons on a small medical dataset.

## Features

* Upload your `heart_failure_clinical_records_dataset.csv` and preprocess automatically (train/test split + standard scaling).
* Choose optimizer: **Adam, Adagrad, Adamax, Adadelta, RMSprop**.
* Configure learning rate (where applicable) and number of epochs.
* Early stopping with restore-best-weights to avoid overfitting.
* Plots for training/validation loss and accuracy.
* Test set evaluation with accuracy and detailed classification report.

## Model architecture

* Input layer: `input_dim` (depends on dataset features)
* Dense(32, activation=`relu`, kernel_initializer=`he_uniform`)
* Dense(16, activation=`relu`, kernel_initializer=`he_uniform`)
* Dense(8, activation=`relu`, kernel_initializer=`he_uniform`)
* Dropout(0.15)
* Dense(4, activation=`relu`, kernel_initializer=`he_uniform`)
* Dropout(0.30)
* Output: Dense(1, activation=`sigmoid`)

Loss: `binary_crossentropy`

Optimizer: selectable in the UI (configured with the chosen learning rate when applicable).

## Files in this repo

* `Artificial_Neural_Network.py` — Streamlit app (main script) that contains preprocessing, model definition, training loop, plotting and evaluation.
* `Artificial Neural Network Frontend.pdf` — PDF containing app screenshots and explanation (optional demo / documentation).
* (Add) `heart_failure_clinical_records_dataset.csv` — not included here; upload via the Streamlit UI when running locally.

## Installation

1. Create a Python virtual environment (recommended):

2. Install dependencies. Example `requirements.txt` contents:

```
streamlit
pandas
numpy
scikit-learn
matplotlib
tensorflow>=2.10
```

Then install:

```bash
pip install -r requirements.txt
```

## Running the app (Streamlit)

1. Place `Artificial_Neural_Network.py` in a folder and ensure the virtual environment is active.
2. Run the app:

```bash
streamlit run Artificial_Neural_Network.py
```

3. In the app UI, upload `heart_failure_clinical_records_dataset.csv` (the original dataset with `DEATH_EVENT` column). Use sidebar controls to select optimizer, learning rate and epochs. Click **Train Model and Analyze Results**.

## Usage notes

* The app standardizes features using `StandardScaler` and stratifies the train/test split by the target for consistent class balance.
* Early stopping is enabled (patience=20, `min_delta=0.001`) with `validation_split=0.2` during training.
* `Adadelta` uses its default learning rate behavior in the script (set to 1.0 in the code), while other optimizers use the sidebar `learning_rate` input.
* Batch size is fixed at 32 in the app.

## Results & Visualizations

After training, the app shows:

* Training and validation **loss** vs epochs plot.
* Training and validation **accuracy** vs epochs plot.
* Final **test set accuracy** and **classification report** (precision, recall, f1-score for each class).
* Training time in seconds and number of epochs actually executed (due to early stopping).

## Tips

* Try different optimizers and learning rates to observe convergence speed and final validation performance.
* Reduce/increase dropout or change layer sizes to experiment with model capacity.
* For reproducible comparisons, keep `random_state=42` in `train_test_split` and run multiple seeds if you want average performance.

## Troubleshooting

* **`DEATH_EVENT` column missing:** Upload the correct CSV or inspect column names for whitespace/format issues.
* **GPU / TensorFlow-related errors:** Ensure TensorFlow is installed correctly for your platform. If you have a GPU, install `tensorflow` with GPU support and the necessary drivers (CUDA/cuDNN) matching your TensorFlow version.
* **Slow training:** Reduce `epochs`, lower model size, or run on a machine with GPU.


  <img width="1919" height="928" alt="Screenshot (183)" src="https://github.com/user-attachments/assets/0040c083-b025-4551-baed-0e6f41bb005b" />


