import torch
from transformers import AutoTokenizer
import joblib
from model import DistilBertForICD10
import numpy as np
import argparse

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(text, model, tokenizer, mlb, threshold):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = outputs['logits']
        sigmoids = torch.sigmoid(logits).cpu().numpy()[0]
    pred_labels = (sigmoids > threshold).astype(int)
    pred_codes = mlb.inverse_transform(np.array([pred_labels]))[0]
    return pred_codes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict ICD codes for input text using the trained model.")
    parser.add_argument('--text', type=str, required=True, help='Text to predict ICD codes for.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for prediction (default: 0.5).')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("model/")
    mlb = joblib.load("model/mlb.joblib")
    num_labels = len(mlb.classes_)
    model = DistilBertForICD10.from_pretrained("model/", num_labels=num_labels)
    model.to(DEVICE)
    model.eval()

    codes = predict(args.text, model, tokenizer, mlb, args.threshold)
    print(f"Predicted ICD codes: {codes}") 