import datasets
from transformers import AutoTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "FiscaAI/synth-ehr-icd10cm-prompt"

def load_dataset(sample_size=None):
    ds = datasets.load_dataset(DATASET_NAME)
    data = ds['train']
    if sample_size is not None:
        data = data.shuffle(seed=42).select(range(sample_size))
    return data

def preprocess_data(dataset, tokenizer=None, max_length=128, mlb=None, return_raw_labels=False):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    texts = dataset['user']
    codes = dataset['codes']
    if return_raw_labels:
        y = codes
        mlb_out = mlb if mlb is not None else MultiLabelBinarizer().fit(codes)
    else:
        if mlb is None:
            mlb = MultiLabelBinarizer()
            y = mlb.fit_transform(codes)
        else:
            y = mlb.transform(codes)
        mlb_out = mlb
    X = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    return X, y, mlb_out

def get_train_test(tokenizer=None, test_size=0.2, random_state=42, sample_size=None, return_test_raw_labels=False):
    dataset = load_dataset(sample_size=sample_size)
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=test_size, random_state=random_state)
    train_data = dataset.select(train_idx)
    test_data = dataset.select(test_idx)
    X_train, y_train, mlb = preprocess_data(train_data, tokenizer)
    X_test, y_test, _ = preprocess_data(test_data, tokenizer, mlb=mlb, return_raw_labels=return_test_raw_labels)
    return X_train, y_train, X_test, y_test, mlb

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_dataset(sample_size=10)
    print(f"Loaded {len(dataset)} samples.")
    X, y, mlb = preprocess_data(dataset, tokenizer)
    print(f"Tokenized shape: {X['input_ids'].shape}, Labels shape: {y.shape}") 