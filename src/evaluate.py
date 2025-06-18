import torch
from transformers import AutoTokenizer
from model import DistilBertForICD10
from data import get_train_test
import joblib
from sklearn.metrics import precision_recall_curve, average_precision_score
from tqdm import tqdm
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

BATCH_SIZE = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', type=int, default=None, help='Number of test samples to evaluate.')
    parser.add_argument('--text', type=str, default=None, help='Text to predict ICD codes for.')
    args = parser.parse_args()

    # Ensure debug directory exists
    debug_dir = os.path.join(os.path.dirname(__file__), '../debug')
    os.makedirs(debug_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("model/")
    mlb = joblib.load("model/mlb.joblib")
    # Check classes loading correctly
    # print("MLB classes:", mlb.classes_)
    num_labels = len(mlb.classes_)
    model = DistilBertForICD10.from_pretrained("model/", num_labels=num_labels)
    model.to(DEVICE)
    model.eval()

    _, _, X_test, y_test, _ = get_train_test(tokenizer, return_test_raw_labels=True)

    # Subsample if requested
    if args.sample_size is not None:
        n = len(y_test)
        sample_size = min(args.sample_size, n)
        indices = np.random.choice(n, sample_size, replace=False)
        X_test = {k: v[indices] for k, v in X_test.items()}
        y_test = [y_test[i] for i in indices]

    # Binarize y_test using loaded mlb
    y_test = mlb.transform(y_test)

    # print("First 5 raw y_test examples:", y_test[:5])
    # print("Transformed y_test shape:", y_test.shape)
    # print("First 5 transformed y_test rows:", y_test[:5])
    # print("Inverse transform of first row:", mlb.inverse_transform(y_test[:1]))
    # print("Number of all-zero rows in y_test:", np.sum(np.sum(y_test, axis=1) == 0))

    test_data = torch.utils.data.TensorDataset(X_test['input_ids'], X_test['attention_mask'])
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)

    sigmoid_outputs = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids, attention_mask = [x.to(DEVICE) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            sigmoids = torch.sigmoid(logits).cpu().numpy()
            sigmoid_outputs.append(sigmoids)
    sigmoid_outputs = np.vstack(sigmoid_outputs)

    # Plot and save sigmoid distribution
    plt.figure()
    plt.hist(sigmoid_outputs.ravel(), bins=50, color='skyblue', edgecolor='black')
    plt.title('Sigmoid Output Distribution')
    plt.xlabel('Sigmoid Output')
    plt.ylabel('Frequency')
    sigmoid_dist_path = os.path.join(debug_dir, 'sigmoid_distribution.png')
    plt.savefig(sigmoid_dist_path)
    plt.close()
    print(f'Saved sigmoid distribution to {sigmoid_dist_path}')

    # Precision-Recall Curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    n_classes = y_test.shape[1]
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], sigmoid_outputs[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], sigmoid_outputs[:, i])
    # Compute micro-average PR curve and AP
    precision_micro, recall_micro, thresholds_micro = precision_recall_curve(y_test.ravel(), sigmoid_outputs.ravel())
    average_precision["micro"] = average_precision_score(y_test, sigmoid_outputs, average="micro")

    plt.figure()
    plt.step(recall_micro, precision_micro, where="post")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve: AP={average_precision["micro"]:0.2f}')
    pr_curve_path = os.path.join(debug_dir, 'precision_recall_curve.png')
    plt.savefig(pr_curve_path)
    plt.close()
    print(f'Saved precision-recall curve to {pr_curve_path}')
    print(f"Micro-averaged Average Precision: {average_precision['micro']:.4f}")

    print("y_test shape:", y_test.shape)
    print("sigmoid_outputs shape:", sigmoid_outputs.shape)
    print("Number of positive labels in y_test:", y_test.sum())

    f1 = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro + 1e-8)
    best_idx = np.argmax(f1)
    best_threshold = thresholds_micro[best_idx]
    print(f"Best threshold by PR curve: {best_threshold:.2f}") 