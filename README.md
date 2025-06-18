# ICD-10 Classification Take-Home

This project demonstrates an end-to-end pipeline for classifying clinical notes into ICD-10 codes using the SynthEHR ICD-10-CM dataset. The core of the solution is a PyTorch-based transformer model, fine-tuned for multi-label classification of medical text.

* I used generated code to scaffold the pipeline.
* To complete this assignment under the time constraints, I had to sample the training and evaluation datasets. 
* After model training, I was able to generate a PR curve and optimal threshold, and then predict codes for a given text string. 

## Project Overview

- **Input:** Free-text clinical notes (from the `user` column)
- **Output:** ICD-10 diagnosis codes (as a list in the `codes` column)
- **Model:** A pretrained transformer (e.g., DistilBERT) is used as the backbone.

## Setup

```bash
git clone https://github.com/th3o6a1d/jt-inv-ph
cd jt-inv-ph
pip install -r requirements.txt
```

## Usage

### 1. Train the Model (with a Small Sample Size)

To quickly test the pipeline or debug, you can train the model on a small subset of the data by specifying a sample size:

```bash
python src/train.py --sample_size 100 --epochs 5
```

Omit `--sample_size` to use the full dataset.

### 2. Evaluate the Model (with a Small Sample Size)

Similarly, you can evaluate the model on a small sample for quick feedback:

```bash
python src/evaluate.py --sample_size 100
```

This will run evaluation on 100 examples. Omit `--sample_size` to evaluate on the full validation set.

The evaluate script will create a folder called debug that will show sigmoid thresholds and PR curve.

### 3. Make Predictions

To predict ICD-10 codes for new clinical notes, use the `predict.py` script. You can pass a text file or a string as input:

You should pass in the threshold obtained after running `evaluate.py`

```bash
python src/predict.py --threshold 0.15 --text "Patient presents with chest pain and shortness of breath."
```

## Notes
- The dataset is loaded directly from Hugging Face Datasets.
- Scripts are designed for clarity and reproducibility, not for full model convergence within the time limit.
- For experimentation or debugging, use the `--sample_size` argument to speed up training and evaluation.
