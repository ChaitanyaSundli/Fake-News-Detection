import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                            precision_recall_curve, roc_auc_score, roc_curve, auc,
                            precision_score, recall_score)
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
import re
import os
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed_value)

def load_data(filename):
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found")

    if filename.endswith('.csv'):
        df = pd.read_csv(filename)
    elif filename.endswith('.xlsx') or filename.endswith('.xls'):
        df = pd.read_excel(filename)
    else:
        raise ValueError("Unsupported file format. Please use CSV or Excel file.")

    required_cols = ['text', 'label']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the dataset")

    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution: \n{df['label'].value_counts(normalize=True)}")

    return df

def preprocess_text(text, remove_stopwords=False):
   
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text) 
    text = re.sub(r'<.*?>', '', text) 
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()  

    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])

    return text

def contains_negation(text):

    text_lower = text.lower()

    negation_words = [
        ' not ', ' never ', ' neither ', ' nor ', ' none ',
        ' isn\'t ', ' aren\'t ', ' wasn\'t ', ' weren\'t ', ' hasn\'t ', ' haven\'t ',
        ' hadn\'t ', ' doesn\'t ', ' don\'t ', ' didn\'t ', ' won\'t ', ' wouldn\'t ',
        ' can\'t ', ' cannot ', ' couldn\'t ', ' shouldn\'t ', ' isnt ', ' arent ',
        ' wasnt ', ' werent ', ' hasnt ', ' havent ', ' hadnt ', ' doesnt ',
        ' dont ', ' didnt ', ' wont ', ' wouldnt ', ' cant ', ' couldnt ',
        ' shouldnt ', ' no longer ', ' not been ', ' not the ',
        ' refuted ', ' debunked ', ' denies ', ' deny '
    ]

    has_negation = any(neg_word in f" {text_lower} " for neg_word in negation_words)

    negation_count = sum(text_lower.count(neg_word.strip()) for neg_word in negation_words)

    return {
        'has_negation': has_negation,
        'negation_count': negation_count
    }

class TextVectorizer:
    def __init__(self, max_features=10000, max_len=200, embedding_dim=100):
        self.max_features = max_features
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.word_index = None
        self.vocab_size = None
        self.embedding_matrix = None
    
    def fit_transform(self, texts):
        
        word_counts = Counter()
        for text in texts:
            word_counts.update(text.split())

        most_common = word_counts.most_common(self.max_features - 2)  
        self.word_index = {'<PAD>': 0, '<UNK>': 1}
        for word, _ in most_common:
            self.word_index[word] = len(self.word_index)

        self.vocab_size = len(self.word_index)

        sequences = []
        for text in texts:
            sequence = [self.word_index.get(word, 1) for word in text.split()] 
            if len(sequence) > self.max_len:
                sequence = sequence[:self.max_len]
            else:
                sequence = sequence + [0] * (self.max_len - len(sequence)) 
            sequences.append(sequence)

        return np.array(sequences)

    def transform(self, texts):
        if self.word_index is None:
            raise ValueError("Vectorizer has not been fitted yet")

        sequences = []
        for text in texts:
            sequence = [self.word_index.get(word, 1) for word in text.split()] 
            if len(sequence) > self.max_len:
                sequence = sequence[:self.max_len]
            else:
                sequence = sequence + [0] * (self.max_len - len(sequence)) 
            sequences.append(sequence)

        return np.array(sequences)

    def load_embeddings(self, embeddings_file=None):
        self.embedding_matrix = np.random.uniform(-0.1, 0.1, size=(self.vocab_size, self.embedding_dim))
        self.embedding_matrix[0] = np.zeros(self.embedding_dim) 

        return self.embedding_matrix
    
    def save_word_index(self, path="word_index.pt"):
        if self.word_index is None:
            raise ValueError("Word index is not built yet. Call fit_transform() first.")
        torch.save(self.word_index, path)

class LSTMClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx, embedding_matrix=None):
        super().__init__()

        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))

        self.lstm = torch.nn.LSTM(embedding_dim,
                                hidden_dim,
                                num_layers=n_layers,
                                bidirectional=bidirectional,
                                dropout=dropout if n_layers > 1 else 0,
                                batch_first=True)

        self.fc = torch.nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, text):

        embedded = self.embedding(text)

        output, (hidden, cell) = self.lstm(embedded)

        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]

        hidden = self.dropout(hidden)

        return self.fc(hidden)

class BiLSTMAttention(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 dropout, pad_idx, embedding_matrix=None):
        super().__init__()

        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))

        self.lstm = torch.nn.LSTM(embedding_dim,
                                hidden_dim,
                                num_layers=n_layers,
                                bidirectional=True,
                                dropout=dropout if n_layers > 1 else 0,
                                batch_first=True)

        self.attention = torch.nn.Linear(hidden_dim * 2, 1)

        self.fc = torch.nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, text):

        embedded = self.embedding(text)

        output, (hidden, cell) = self.lstm(embedded)

        attention_scores = self.attention(output).squeeze(2)
        mask = (text != 0).float()
        attention_scores = attention_scores.masked_fill(mask == 0, -1e10)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), output).squeeze(1)

        context_vector = self.dropout(context_vector)

        return self.fc(context_vector)

class TextCNN(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                dropout, pad_idx, embedding_matrix=None):
        super().__init__()

        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))

        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels=1,
                          out_channels=n_filters,
                          kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = torch.nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        
        embedded = embedded.unsqueeze(1)

        conved = [torch.nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs]


        pooled = [torch.nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))

        return self.fc(cat)


def train_model(model, train_dataloader, validation_dataloader, optimizer, criterion, device, epochs=4, early_stopping_patience=3):
    model.to(device)

   
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    training_stats = []

    for epoch in range(epochs):
        print(f'\n======== Epoch {epoch+1} / {epochs} ========')
        start_time = time.time()

        model.train()
        total_train_loss = 0
        train_progress = tqdm(train_dataloader, desc="Training")

        for batch in train_progress:
            b_input_ids = batch[0].to(device)
            b_labels = batch[1].to(device)

            model.zero_grad()
            outputs = model(b_input_ids)

            loss = criterion(outputs, b_labels)
            total_train_loss += loss.item()
            train_progress.set_postfix({"loss": f"{loss.item():.4f}"})

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_dataloader)

        model.eval()
        total_eval_loss = 0
        all_predictions = []
        all_true_labels = []
        all_probs = []

        val_progress = tqdm(validation_dataloader, desc="Validation")

        for batch in val_progress:
            b_input_ids = batch[0].to(device)
            b_labels = batch[1].to(device)

            with torch.no_grad():
                outputs = model(b_input_ids)
                loss = criterion(outputs, b_labels)

            total_eval_loss += loss.item()

            probs = torch.nn.functional.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(b_labels.cpu().numpy())
            all_probs.extend(probs[:,1].cpu().numpy())

            val_progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_val_loss = total_eval_loss / len(validation_dataloader)
        accuracy = accuracy_score(all_true_labels, all_predictions)

        classification_metrics = classification_report(
            all_true_labels, all_predictions, output_dict=True, zero_division=0
        )

        if len(set(all_true_labels)) == 2:
            roc_auc = roc_auc_score(all_true_labels, all_probs)
        else:
            roc_auc = None

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "./best_fake_news_model.pt")
        else:
            epochs_without_improvement += 1

        epoch_stats = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_accuracy': accuracy,
            'val_precision': classification_metrics['weighted avg']['precision'],
            'val_recall': classification_metrics['weighted avg']['recall'],
            'val_f1': classification_metrics['weighted avg']['f1-score'],
            'roc_auc': roc_auc,
            'time': time.time() - start_time
        }
        training_stats.append(epoch_stats)

        print(f"\nEpoch {epoch+1}:")
        print(f"  Training Loss: {avg_train_loss:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")
        print(f"  Validation Accuracy: {accuracy:.4f}")
        print(f"  Validation F1: {classification_metrics['weighted avg']['f1-score']:.4f}")
        if roc_auc:
            print(f"  ROC-AUC: {roc_auc:.4f}")

        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    model.load_state_dict(torch.load("./best_fake_news_model.pt"))

    plot_training_stats(training_stats)

    return model, training_stats

def plot_training_stats(stats):
    stats_df = pd.DataFrame(stats)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].plot(stats_df['epoch'], stats_df['train_loss'], 'b-o', label='Training')
    axes[0, 0].plot(stats_df['epoch'], stats_df['val_loss'], 'r-o', label='Validation')
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].legend(fontsize=12)
    axes[0, 0].grid(True)

    axes[0, 1].plot(stats_df['epoch'], stats_df['val_accuracy'], 'g-o', label='Accuracy')
    axes[0, 1].plot(stats_df['epoch'], stats_df['val_f1'], 'c-o', label='F1 Score')
    axes[0, 1].set_title('Validation Metrics', fontsize=14)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Score', fontsize=12)
    axes[0, 1].legend(fontsize=12)
    axes[0, 1].grid(True)

    axes[1, 0].plot(stats_df['epoch'], stats_df['val_precision'], 'm-o', label='Precision')
    axes[1, 0].plot(stats_df['epoch'], stats_df['val_recall'], 'y-o', label='Recall')
    axes[1, 0].set_title('Precision and Recall', fontsize=14)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Score', fontsize=12)
    axes[1, 0].legend(fontsize=12)
    axes[1, 0].grid(True)

    if not stats_df['roc_auc'].isnull().all():
        axes[1, 1].plot(stats_df['epoch'], stats_df['roc_auc'], 'r-o', label='ROC-AUC')
        axes[1, 1].set_title('ROC-AUC Score', fontsize=14)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Score', fontsize=12)
        axes[1, 1].legend(fontsize=12)
        axes[1, 1].grid(True)
    else:
        axes[1, 1].set_visible(False)

    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300)
    plt.show()
    plt.close()

def plot_advanced_metrics(true_labels, predictions, raw_scores):

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks([0.5, 1.5], ['Real', 'Fake'], fontsize=12)
    plt.yticks([0.5, 1.5], ['Real', 'Fake'], fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()
    plt.close()

    if len(set(true_labels)) == 2:
        plt.figure(figsize=(10, 8))
        fpr, tpr, thresholds = roc_curve(true_labels, raw_scores)
        auc_score = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300)
        plt.show()  
        plt.close()

        plt.figure(figsize=(10, 8))
        precision, recall, thresholds = precision_recall_curve(true_labels, raw_scores)
        average_precision = np.sum(np.diff(recall) * np.array(precision)[:-1])

        plt.plot(recall, precision, lw=2, label=f'PR curve (AP = {average_precision:.3f})')
        plt.axhline(y=sum(true_labels) / len(true_labels), color='r', linestyle='--',
                   label=f'Baseline (support = {sum(true_labels)/len(true_labels):.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Precision-Recall Curve', fontsize=16)
        plt.legend(loc="lower left", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('precision_recall_curve.png', dpi=300)
        plt.show()  
        plt.close()

        plt.figure(figsize=(12, 8))

        scores_pos = np.array(raw_scores)[np.array(true_labels) == 1]
        scores_neg = np.array(raw_scores)[np.array(true_labels) == 0]

        sns.histplot(scores_pos, color='green', alpha=0.6, bins=30, kde=True,
                    label='Fake News (Positive Class)')
        sns.histplot(scores_neg, color='red', alpha=0.6, bins=30, kde=True,
                    label='Real News (Negative Class)')

        plt.axvline(x=0.5, color='black', linestyle='--', label='Default Threshold (0.5)')
        plt.legend(fontsize=12)
        plt.title('Score Distribution by Class', fontsize=16)
        plt.xlabel('Predicted Probability (Score)', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('score_distribution.png', dpi=300)
        plt.show() 
        plt.close()

def evaluate_model(model, test_dataloader, criterion, device, threshold=0.5):
    model.to(device)
    model.eval()

    predictions = []
    raw_scores = []
    true_labels = []
    total_loss = 0

    print("Evaluating model...")
    eval_progress = tqdm(test_dataloader)

    for batch in eval_progress:
        b_input_ids = batch[0].to(device)
        b_labels = batch[1].to(device)

        with torch.no_grad():
            outputs = model(b_input_ids)
            loss = criterion(outputs, b_labels)
            total_loss += loss.item()

        logits = outputs
        probs = torch.nn.functional.softmax(logits, dim=1)
        raw_scores.extend(probs[:, 1].cpu().numpy())

        batch_predictions = (probs[:, 1] > threshold).int().cpu().numpy()
        predictions.extend(batch_predictions)
        true_labels.extend(b_labels.cpu().numpy())

    avg_loss = total_loss / len(test_dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    conf_matrix = confusion_matrix(true_labels, predictions)
    class_report = classification_report(true_labels, predictions, target_names=['Real', 'Fake'])

    if len(set(true_labels)) == 2:
        roc_auc = roc_auc_score(true_labels, raw_scores)

    print(f"Test Loss: {avg_loss:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

    plot_advanced_metrics(true_labels, predictions, raw_scores)

    precisions, recalls, thresholds = precision_recall_curve(true_labels, raw_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

    print(f"Optimal threshold: {optimal_threshold:.3f}")

    return {
        'test_loss': avg_loss,
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'roc_auc': roc_auc if len(set(true_labels)) == 2 else None,
        'optimal_threshold': optimal_threshold,
        'raw_scores': raw_scores,
        'true_labels': true_labels,
        'predictions': predictions
    }

def predict_text(model, vectorizer, text, device, threshold=70):

    negation_info = contains_negation(text)
    has_negation = negation_info['has_negation']

    processed_text = preprocess_text(text)

    sequence = vectorizer.transform([processed_text])

    input_tensor = torch.tensor(sequence).to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)

    fake_prob = probs[0, 1].item()
    real_prob = probs[0, 0].item()

    if has_negation:
        fake_prob, real_prob = real_prob, fake_prob

    prediction = 'FAKE' if fake_prob >= threshold else 'REAL'

    return {
        'prediction': prediction,
        'fake_probability': fake_prob,
        'real_probability': real_prob,
        'confidence': max(fake_prob, real_prob),
        'negation_detected': has_negation,
        'negation_count': negation_info['negation_count'],
        'original_fake_prob': probs[0, 1].item(),
        'original_real_prob': probs[0, 0].item()
    }

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        dataset_path = input("Enter Dataset Filepath:")
        df = load_data(dataset_path)
        print("Dataset loaded successfully!")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("Preprocessing text data...")
    df['processed_text'] = df['text'].apply(lambda x: preprocess_text(x, remove_stopwords=True))

    X = df['processed_text'].values
    y = df['label'].values

    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.1765, random_state=42, stratify=y_train_temp)

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")

    print("Vectorizing text data...")
    vectorizer = TextVectorizer(max_features=10000, max_len=200, embedding_dim=100)
    X_train_seq = vectorizer.fit_transform(X_train)

    import os
    save_path = os.path.join(os.getcwd(), "word_index.pt")
    vectorizer.save_word_index(save_path)
    print(f"✅ word_index.pt saved to: {save_path}")

    X_val_seq = vectorizer.transform(X_val)
    X_test_seq = vectorizer.transform(X_test)

    embedding_matrix = vectorizer.load_embeddings()

    train_dataset = TensorDataset(torch.tensor(X_train_seq), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val_seq), torch.tensor(y_val))
    test_dataset = TensorDataset(torch.tensor(X_test_seq), torch.tensor(y_test))

    batch_size = 32
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

    vocab_size = vectorizer.vocab_size
    embedding_dim = 100
    hidden_dim = 128
    output_dim = 2  
    n_layers = 2
    bidirectional = True
    dropout = 0.5
    pad_idx = 0

    model_choice = input("Choose model architecture (1: BiLSTM with Attention, 2: TextCNN, 3: Regular LSTM): ")

    if model_choice == '1':
        print("Creating BiLSTM with Attention model...")
        model = BiLSTMAttention(
            vocab_size, embedding_dim, hidden_dim, output_dim,
            n_layers, dropout, pad_idx, embedding_matrix
        )
    elif model_choice == '2':
        print("Creating TextCNN model...")
        filter_sizes = [2, 3, 4, 5]
        n_filters = 100
        model = TextCNN(
            vocab_size, embedding_dim, n_filters, filter_sizes,
            output_dim, dropout, pad_idx, embedding_matrix
        )
    else:
        print("Creating LSTM model...")
        model = LSTMClassifier(
            vocab_size, embedding_dim, hidden_dim, output_dim,
            n_layers, bidirectional, dropout, pad_idx, embedding_matrix
        )

    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    print("Training model...")
    trained_model, training_stats = train_model(
        model, train_dataloader, val_dataloader, optimizer, criterion,
        device, epochs=8, early_stopping_patience=3
    )

    print("Evaluating model on test set...")
    test_results = evaluate_model(trained_model, test_dataloader, criterion, device)

    while True:
        print("\n--- Fake News Detector ---")
        print("Enter a news article text to classify (or 'q' to quit):")
        user_input = input("> ")

        if user_input.lower() == 'q':
            break

        if len(user_input.strip()) < 10:
            print("Please enter a longer text to analyze.")
            continue

        result = predict_text(model=trained_model, vectorizer=vectorizer, text=user_input, device=device, threshold=0.5)

        print("\nPrediction Results:")
        print(f"Classification: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Probability of being FAKE: {result['fake_probability']:.2%}")
        print(f"Probability of being REAL: {result['real_probability']:.2%}")

if __name__ == "__main__":
    main()