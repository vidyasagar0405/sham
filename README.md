# Sham: Go Naive Bayes Spam Classifier

A Naive Bayes text classifier written in Go using only the standard library. The project parses a dataset, trains a mathematical model, and uses binary serialization to save the model for subsequent text classification via a command-line interface.

## Features
* **Standard Library Only:** Implements the core machine learning logic without external dependencies.
* **Mathematical Handling:**
    * Uses **Laplace (Add-1) Smoothing** to handle unseen vocabulary during testing.
    * Uses **Logarithmic Probabilities** to prevent floating-point underflow.
* **Binary Serialization:** Exports the trained model to a `.gob` file (`shamModel.gob`) so the inference tool doesn't need to parse the original dataset or recalculate probabilities on every run.
* **Inference CLI:** A command-line tool that loads the `.gob` file to classify raw strings or text files.
* **Evaluation Metrics:** Calculates a confusion matrix, Accuracy, Precision, Recall, and F1 Score against a 20% validation split during the training phase.

=== Model Evaluation Dashboard ===
Total Tested : 38771

True Positives (Spam caught) : 17165
True Negatives (Ham kept)    : 20182
False Positives (Ham trashed): 321
False Negatives (Spam leaked): 1103

Accuracy  : 0.9633
Precision : 0.9816
Recall    : 0.9396
F1 Score  : 0.9602
==================================


## Dataset
The model is trained on [190k Spam/Ham Email Dataset for Classification](https://www.kaggle.com/datasets/meruvulikith/190k-spam-ham-email-dataset-for-classification) from Kaggle.

Place the extracted dataset in the `./dataset/` directory as `spam_Emails_data.csv` before running the training script.

## Project Structure

The project is divided into two separate programs:

1. `main.go` **(Training):** Reads the CSV, builds the Bag-of-Words dictionaries, applies smoothing, converts frequencies to log probabilities, evaluates the model, and exports `shamModel.gob`.
2. `predict.go` **(Inference):** A CLI tool that loads the `.gob` file and processes terminal input or files to output a prediction.

## Build and Run

```bash
# Clone the repository
git clone [https://github.com/yourusername/sham-classifier.git](https://github.com/yourusername/sham-classifier.git)
cd sham-classifier

# Build the scripts
go build -o train train.go
go build -o predict predict.go

```

### Training

Run the training script to evaluate the data and generate the model binary.

```bash
./train-model

```

### Predictions

Use the `predict` binary to classify new text.

**Classify a raw string:**

```bash
./predict "UR chosen 2 receive a £1000 cash prize! Txt WIN"

```

**Classify specific text files:**

```bash
./predict email_01.txt email_02.txt

```

**Classify an entire directory:**

```bash
./predict ./inbox/*.txt

```

**Specify a custom model:**
Point the CLI to a specific `.gob` file using the `--model` flag.

```bash
./predict --model custom_model.gob ./inbox/*.txt

```
