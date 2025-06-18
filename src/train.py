import torch
from transformers import AutoTokenizer
from torch.optim import AdamW
from tqdm import tqdm
from data import get_train_test, MODEL_NAME
from model import get_model
import argparse

EPOCHS = 10
BATCH_SIZE = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument('--sample-size', type=int, default=100, help='Number of samples to use for training/testing')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train for')
    args = parser.parse_args()
    EPOCHS = args.epochs if args.epochs is not None else EPOCHS

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    X_train, y_train, X_test, y_test, mlb = get_train_test(tokenizer, sample_size=args.sample_size)
    num_labels = y_train.shape[1]
    model = get_model(num_labels)
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    train_data = torch.utils.data.TensorDataset(X_train['input_ids'], X_train['attention_mask'], torch.tensor(y_train, dtype=torch.float32))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    print("Total positive labels in y_train:", y_train.sum())
    print("Total positive labels in y_test:", y_test.sum())
    # print("Per-class positive labels in y_train:", y_train.sum(axis=0))
    # print("Per-class positive labels in y_test:", y_test.sum(axis=0))

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids, attention_mask, labels = [x.to(DEVICE) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {total_loss/len(train_loader):.4f}")

    model.save_pretrained("model/")
    tokenizer.save_pretrained("model/")
    import joblib
    joblib.dump(mlb, "model/mlb.joblib")
    print("Model and tokenizer saved to model/") 
