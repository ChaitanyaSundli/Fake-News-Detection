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

# Set random seed for reproducibility
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed_value)

# Function to load and preprocess the dataset
def load_data(filename):
    """Load the dataset from a CSV or Excel file"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found")

    # Determine file type and read accordingly
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)
    elif filename.endswith('.xlsx') or filename.endswith('.xls'):
        df = pd.read_excel(filename)
    else:
        raise ValueError("Unsupported file format. Please use CSV or Excel file.")

    # Check if required columns exist
    required_cols = ['text', 'label']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the dataset")

    # Display dataset statistics
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution: \n{df['label'].value_counts(normalize=True)}")

    return df

# Text preprocessing function
def preprocess_text(text, remove_stopwords=False):
    """Clean and preprocess text data"""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace

    # Optionally remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])

    return text

def contains_negation(text):
    """
    Check if text contains negation words and return information about the negation.
    
    Args:
        text (str): The input text to analyze
        
    Returns:
        dict: Dictionary with information about negation
    """
    # Lowercase the text for easier matching
    text_lower = text.lower()
    
    # List of common negation words and phrases
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
    
    # Check if any negation word is in the text
    has_negation = any(neg_word in f" {text_lower} " for neg_word in negation_words)
    
    # Count how many negation words appear
    negation_count = sum(text_lower.count(neg_word.strip()) for neg_word in negation_words)
    
    # Return information about negation
    return {
        'has_negation': has_negation,
        'negation_count': negation_count
    }

# Text vectorization class
class TextVectorizer:
    def __init__(self, max_features=10000, max_len=200, embedding_dim=100):
        self.max_features = max_features
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.word_index = None
        self.vocab_size = None
        self.embedding_matrix = None
        
    def fit_transform(self, texts):
        """Build vocabulary and transform texts to sequences"""
        # Build vocabulary from training data
        word_counts = Counter()
        for text in texts:
            word_counts.update(text.split())
        
        # Keep most common words
        most_common = word_counts.most_common(self.max_features - 2)  # -2 for <PAD> and <UNK>
        self.word_index = {'<PAD>': 0, '<UNK>': 1}
        for word, _ in most_common:
            self.word_index[word] = len(self.word_index)
        
        self.vocab_size = len(self.word_index)
        
        # Transform texts to sequences
        sequences = []
        for text in texts:
            sequence = [self.word_index.get(word, 1) for word in text.split()]  # 1 is <UNK>
            if len(sequence) > self.max_len:
                sequence = sequence[:self.max_len]
            else:
                sequence = sequence + [0] * (self.max_len - len(sequence))  # Pad with zeros
            sequences.append(sequence)
        
        return np.array(sequences)
    
    def transform(self, texts):
        """Transform texts to sequences using fitted vocabulary"""
        if self.word_index is None:
            raise ValueError("Vectorizer has not been fitted yet")
        
        sequences = []
        for text in texts:
            sequence = [self.word_index.get(word, 1) for word in text.split()]  # 1 is <UNK>
            if len(sequence) > self.max_len:
                sequence = sequence[:self.max_len]
            else:
                sequence = sequence + [0] * (self.max_len - len(sequence))  # Pad with zeros
            sequences.append(sequence)
        
        return np.array(sequences)
    
    def load_embeddings(self, embeddings_file=None):
        """Load pre-trained word embeddings (optional)"""
        # For simplicity, we'll create random embeddings
        # In practice, you would load GloVe or Word2Vec embeddings
        self.embedding_matrix = np.random.uniform(-0.1, 0.1, size=(self.vocab_size, self.embedding_dim))
        self.embedding_matrix[0] = np.zeros(self.embedding_dim)  # <PAD> token
        
        return self.embedding_matrix

# Define the LSTM model
class LSTMClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx, embedding_matrix=None):
        super().__init__()
        
        # Embedding layer
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # Initialize with pre-trained embeddings if provided
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        
        # LSTM layer
        self.lstm = torch.nn.LSTM(embedding_dim, 
                                hidden_dim, 
                                num_layers=n_layers, 
                                bidirectional=bidirectional, 
                                dropout=dropout if n_layers > 1 else 0,
                                batch_first=True)
        
        # Linear layer for classification
        self.fc = torch.nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        # Dropout layer
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, text):
        # text shape: [batch size, seq len]
        embedded = self.embedding(text)
        # embedded shape: [batch size, seq len, embedding dim]
        
        output, (hidden, cell) = self.lstm(embedded)
        # output shape: [batch size, seq len, hidden dim * n directions]
        
        if self.lstm.bidirectional:
            # Concatenate the final forward and backward hidden states
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Pass through linear layer
        return self.fc(hidden)

# Define the BiLSTM model with attention
class BiLSTMAttention(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 dropout, pad_idx, embedding_matrix=None):
        super().__init__()
        
        # Embedding layer
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # Initialize with pre-trained embeddings if provided
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        
        # BiLSTM layer
        self.lstm = torch.nn.LSTM(embedding_dim, 
                                hidden_dim, 
                                num_layers=n_layers, 
                                bidirectional=True, 
                                dropout=dropout if n_layers > 1 else 0,
                                batch_first=True)
        
        # Attention layer
        self.attention = torch.nn.Linear(hidden_dim * 2, 1)
        
        # Linear layer for classification
        self.fc = torch.nn.Linear(hidden_dim * 2, output_dim)
        
        # Dropout layer
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, text):
        # text shape: [batch size, seq len]
        embedded = self.embedding(text)
        # embedded shape: [batch size, seq len, embedding dim]
        
        output, (hidden, cell) = self.lstm(embedded)
        # output shape: [batch size, seq len, hidden dim * 2]
        
        # Attention mechanism
        attention_scores = self.attention(output).squeeze(2)
        # Apply mask for padding tokens
        mask = (text != 0).float()
        attention_scores = attention_scores.masked_fill(mask == 0, -1e10)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=1)
        # Apply attention weights
        context_vector = torch.bmm(attention_weights.unsqueeze(1), output).squeeze(1)
        
        # Apply dropout
        context_vector = self.dropout(context_vector)
        
        # Pass through linear layer
        return self.fc(context_vector)

# Define the CNN model for text classification
class TextCNN(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                dropout, pad_idx, embedding_matrix=None):
        super().__init__()
        
        # Embedding layer
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # Initialize with pre-trained embeddings if provided
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        
        # Convolutional layers with different filter sizes
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels=1, 
                          out_channels=n_filters, 
                          kernel_size=(fs, embedding_dim)) 
            for fs in filter_sizes
        ])
        
        # Linear layer for classification
        self.fc = torch.nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        # Dropout layer
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, text):
        # text shape: [batch size, seq len]
        embedded = self.embedding(text)
        # embedded shape: [batch size, seq len, embedding dim]
        
        # Add channel dimension for CNN
        embedded = embedded.unsqueeze(1)
        # embedded shape: [batch size, 1, seq len, embedding dim]
        
        # Apply convolutions and max-pooling
        conved = [torch.nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved_n shape: [batch size, n_filters, seq_len - filter_sizes[n] + 1]
        
        pooled = [torch.nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n shape: [batch size, n_filters]
        
        # Concatenate pooled features
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat shape: [batch size, n_filters * len(filter_sizes)]
        
        # Pass through linear layer
        return self.fc(cat)

# Training function with early stopping
def train_model(model, train_dataloader, validation_dataloader, optimizer, criterion, device, epochs=4, early_stopping_patience=3):
    model.to(device)

    # Initialize tracking variables
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    training_stats = []

    # Training loop
    for epoch in range(epochs):
        print(f"\n======== Epoch {epoch+1} / {epochs} ========", flush=True)

        start_time = time.time()

        # Training phase
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

        # Validation phase
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

        # Calculate overall metrics
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        accuracy = accuracy_score(all_true_labels, all_predictions)
        
        # Compute detailed metrics
        classification_metrics = classification_report(
            all_true_labels, all_predictions, output_dict=True, zero_division=0
        )

        # Calculate ROC-AUC if applicable
        if len(set(all_true_labels)) == 2:
            roc_auc = roc_auc_score(all_true_labels, all_probs)
        else:
            roc_auc = None

        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            # Save the best model
            torch.save(model.state_dict(), "./best_fake_news_model.pt")
        else:
            epochs_without_improvement += 1

        # Store stats
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

        # Print epoch results
        print(f"\nEpoch {epoch+1}:", flush=True)
        print(f"  Training Loss: {avg_train_loss:.4f}", flush=True)
        print(f"  Validation Loss: {avg_val_loss:.4f}", flush=True)
        print(f"  Validation Accuracy: {accuracy:.4f}", flush=True)
        print(f"  Validation F1: {classification_metrics['weighted avg']['f1-score']:.4f}", flush=True)

        if roc_auc:
            print(f"  ROC-AUC: {roc_auc:.4f}", flush=True)

        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs", flush=True)
            break

    # Load the best model
    model.load_state_dict(torch.load("./best_fake_news_model.pt"))

    # Plot training stats
    plot_training_stats(training_stats)

    return model, training_stats

# Function to plot training statistics
def plot_training_stats(stats):
    stats_df = pd.DataFrame(stats)

    # Create subplots with better layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Loss curves
    axes[0, 0].plot(stats_df['epoch'], stats_df['train_loss'], 'b-o', label='Training')
    axes[0, 0].plot(stats_df['epoch'], stats_df['val_loss'], 'r-o', label='Validation')
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].legend(fontsize=12)
    axes[0, 0].grid(True)

    # Plot 2: Accuracy and F1
    axes[0, 1].plot(stats_df['epoch'], stats_df['val_accuracy'], 'g-o', label='Accuracy')
    axes[0, 1].plot(stats_df['epoch'], stats_df['val_f1'], 'c-o', label='F1 Score')
    axes[0, 1].set_title('Validation Metrics', fontsize=14)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Score', fontsize=12)
    axes[0, 1].legend(fontsize=12)
    axes[0, 1].grid(True)

    # Plot 3: Precision and Recall
    axes[1, 0].plot(stats_df['epoch'], stats_df['val_precision'], 'm-o', label='Precision')
    axes[1, 0].plot(stats_df['epoch'], stats_df['val_recall'], 'y-o', label='Recall')
    axes[1, 0].set_title('Precision and Recall', fontsize=14)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Score', fontsize=12)
    axes[1, 0].legend(fontsize=12)
    axes[1, 0].grid(True)

    # Plot 4: ROC-AUC if available
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
    os.makedirs("static", exist_ok=True)
    plt.savefig('static/training_metrics.png', dpi=300)
    plt.show()  # Add this line to display the plot
    plt.close()

# Function to plot advanced metrics
def plot_advanced_metrics(true_labels, predictions, raw_scores):
    """Create advanced metric visualizations"""
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks([0.5, 1.5], ['Real', 'Fake'], fontsize=12)
    plt.yticks([0.5, 1.5], ['Real', 'Fake'], fontsize=12)
    plt.tight_layout()
    os.makedirs("static", exist_ok=True)
    plt.savefig('static/confusion_matrix.png', dpi=300)
    plt.show()  # Add this line to display the plot
    plt.close()
    
    # Plot ROC curve
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
        os.makedirs("static", exist_ok=True)
        plt.savefig('static/roc_curve.png', dpi=300)
        plt.show()  # Add this line to display the plot
        plt.close()
        
        # Plot Precision-Recall curve
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
        os.makedirs("static", exist_ok=True)
        plt.savefig('static/precision_recall_curve.png', dpi=300)
        plt.show()  # Add this line to display the plot
        plt.close()
        
        # Plot score distribution
        plt.figure(figsize=(12, 8))
        
        # Create separate distributions for positive and negative classes
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
        os.makedirs("static", exist_ok=True)
        plt.savefig('static/score_distribution.png', dpi=300)
        plt.show()  # Add this line to display the plot
        plt.close()


# Enhanced evaluation function with advanced visualizations
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

        # Apply threshold for predictions
        batch_predictions = (probs[:, 1] > threshold).int().cpu().numpy()
        predictions.extend(batch_predictions)
        true_labels.extend(b_labels.cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(test_dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    conf_matrix = confusion_matrix(true_labels, predictions)
    class_report = classification_report(true_labels, predictions, target_names=['Real', 'Fake'])

    # Calculate ROC-AUC
    if len(set(true_labels)) == 2:
        roc_auc = roc_auc_score(true_labels, raw_scores)

    print(f"Test Loss: {avg_loss:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

    # Create enhanced visualizations
    plot_advanced_metrics(true_labels, predictions, raw_scores)

    # Find optimal threshold using precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(true_labels, raw_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

    print(f"Optimal threshold: {optimal_threshold:.3f}")

    # Perform cost-benefit analysis
    cost_benefit_analysis(true_labels, raw_scores)

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

# Cost-Benefit Analysis function
def cost_benefit_analysis(true_labels, scores):
    """
    Perform cost-benefit analysis for different thresholds
    
    Parameters:
    true_labels (list): True class labels
    scores (list): Predicted probabilities
    """
    # Define costs and benefits (these values are examples and should be adjusted)
    cost_false_positive = 10  # Cost of incorrectly flagging real news as fake
    cost_false_negative = 30  # Cost of missing fake news
    benefit_true_positive = 20  # Benefit of correctly identifying fake news
    benefit_true_negative = 5  # Benefit of correctly passing real news
    
    # Calculate costs at different thresholds
    thresholds = np.linspace(0.01, 0.99, 50)
    total_costs = []
    false_positive_rates = []
    false_negative_rates = []
    net_benefits = []
    
    for threshold in thresholds:
        predictions = (np.array(scores) >= threshold).astype(int)
        
        # Calculate confusion matrix components
        tn = np.sum((true_labels == 0) & (predictions == 0))
        fp = np.sum((true_labels == 0) & (predictions == 1))
        fn = np.sum((true_labels == 1) & (predictions == 0))
        tp = np.sum((true_labels == 1) & (predictions == 1))
        
        # Calculate False Positive Rate and False Negative Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        false_positive_rates.append(fpr)
        false_negative_rates.append(fnr)
        
        # Calculate costs and benefits
        total_cost = (fp * cost_false_positive) + (fn * cost_false_negative)
        total_benefit = (tp * benefit_true_positive) + (tn * benefit_true_negative)
        net_benefit = total_benefit - total_cost
        
        total_costs.append(total_cost)
        net_benefits.append(net_benefit)
    
    # Find optimal threshold based on net benefit
    optimal_idx = np.argmax(net_benefits)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\nCost-Benefit Analysis:", flush=True)
    print(f"Optimal threshold for maximum net benefit: {optimal_threshold:.3f}", flush=True)
    print(f"At this threshold:", flush=True)
    print(f"  - False Positive Rate: {false_positive_rates[optimal_idx]:.3f}", flush=True)
    print(f"  - False Negative Rate: {false_negative_rates[optimal_idx]:.3f}", flush=True)
    print(f"  - Net Benefit: {net_benefits[optimal_idx]:.1f}", flush=True)

    
    # Create visualization
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot net benefit
    color = 'tab:blue'
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Net Benefit', color=color)
    ax1.plot(thresholds, net_benefits, color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axvline(x=optimal_threshold, color='r', linestyle='--', alpha=0.7, 
                label=f'Optimal Threshold: {optimal_threshold:.3f}')
    
    # Create a second y-axis for error rates
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Error Rates', color=color)
    ax2.plot(thresholds, false_positive_rates, color='tab:green', linewidth=2, linestyle='-', label='FPR')
    ax2.plot(thresholds, false_negative_rates, color='tab:red', linewidth=2, linestyle='-', label='FNR')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    
    plt.title('Cost-Benefit Analysis by Threshold')
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs("static", exist_ok=True)
    plt.savefig('static/cost_benefit_analysis.png', dpi=300)
    plt.show()  # Add this line to display the plot
    plt.close()
# Function to make predictions on new texts
def predict_text(model, vectorizer, text, device, threshold=0.5):
    """
    Make prediction on a new text input with negation handling
    
    Parameters:
    model: Trained PyTorch model
    vectorizer: Fitted TextVectorizer
    text: String containing text to classify
    device: PyTorch device
    threshold: Classification threshold
    
    Returns:
    Dictionary containing prediction results
    """
    # Check for negation in the original text
    negation_info = contains_negation(text)
    has_negation = negation_info['has_negation']
    
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Vectorize text
    sequence = vectorizer.transform([processed_text])
    
    # Convert to PyTorch tensor
    input_tensor = torch.tensor(sequence).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
    # Get probability scores
    fake_prob = probs[0, 1].item()
    real_prob = probs[0, 0].item()
    
    # Apply negation logic if needed
    # If text contains negation, flip the probabilities
    if has_negation:
        # Switch probabilities
        fake_prob, real_prob = real_prob, fake_prob
    
    # Apply threshold for classification
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
# Main function to run the entire pipeline
def main():
    parser = argparse.ArgumentParser(description="Fake News Detector Trainer")
    parser.add_argument('--train', action='store_true', help='Enable training mode')
    parser.add_argument('dataset_path', nargs='?', default='news.csv', help='Path to dataset file')
    parser.add_argument('model_choice', nargs='?', default='1', help='Model type: 1=BiLSTM, 2=CNN, 3=LSTM')
    args = parser.parse_args()
  
    # Set random seed for reproducibility
    set_seed(42)
    
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not args.train:
        print("ℹ️ No training flag provided. Exiting.")
        return

    try:
        print(f"📂 Loading dataset from: {args.dataset_path}")
        df = load_data(args.dataset_path)
        print("✅ Dataset loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Preprocess text data
    print("🧼 Preprocessing text data...")
    df['processed_text'] = df['text'].apply(lambda x: preprocess_text(x, remove_stopwords=True))
    
    # Convert string labels to integers
    label_mapping = {'real': 0, 'fake': 1}
    df['label'] = df['label'].apply(lambda label: label_mapping[label.lower()])


    # Split data into train, validation, and test sets
    X = df['processed_text'].values
    y = df['label'].values
    
    # 70% train, 15% validation, 15% test
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.1765, random_state=42, stratify=y_train_temp)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    
    # Vectorize text data
    print("Vectorizing text data...")
    vectorizer = TextVectorizer(max_features=10000, max_len=200, embedding_dim=100)
    X_train_seq = vectorizer.fit_transform(X_train)
    X_val_seq = vectorizer.transform(X_val)
    X_test_seq = vectorizer.transform(X_test)
    
    # Load embeddings (optional)
    embedding_matrix = vectorizer.load_embeddings()
    
    # Create PyTorch datasets
    train_dataset = TensorDataset(torch.tensor(X_train_seq), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val_seq), torch.tensor(y_val))
    test_dataset = TensorDataset(torch.tensor(X_test_seq), torch.tensor(y_test))
    
    # Create data loaders
    batch_size = 32
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)
    
    # Define model parameters
    vocab_size = vectorizer.vocab_size
    embedding_dim = 100
    hidden_dim = 128
    output_dim = 2  # Binary classification: real or fake
    n_layers = 2
    bidirectional = True
    dropout = 0.5
    pad_idx = 0
    
    # Choose a model architecture
    
    if args.model_choice == '2':
        model = TextCNN(vocab_size, embedding_dim, 100, [2, 3, 4, 5], output_dim, dropout, pad_idx, embedding_matrix)
    elif args.model_choice == '3':
        model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, True, dropout, pad_idx, embedding_matrix)
    else:
        model = BiLSTMAttention(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout, pad_idx, embedding_matrix)

    
    # Define optimizer and loss function
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Train the model
    print("Training model...")
    trained_model, training_stats = train_model(
        model, train_dataloader, val_dataloader, optimizer, criterion, 
        device, epochs=8, early_stopping_patience=3
    )
    # Save vectorizer word index for API use
    torch.save(vectorizer.word_index, "word_index.pt")
    print("Saved vectorizer word_index to word_index.pt")

    
    # Evaluate the model on test set
    print("Evaluating model on test set...")
    test_results = evaluate_model(trained_model, test_dataloader, criterion, device)
    
    # Interactive prediction mode
    while True:
        print("\n--- Fake News Detector ---")
        print("Enter a news article text to classify (or 'q' to quit):")
        user_input = input("> ")
        
        if user_input.lower() == 'q':
            break
            
        if len(user_input.strip()) < 10:
            print("Please enter a longer text to analyze.")
            continue
            
        result = predict_text(trained_model, vectorizer, user_input, device, threshold=test_results['optimal_threshold'])
        
        print("\nPrediction Results:")
        print(f"Classification: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Probability of being FAKE: {result['fake_probability']:.2%}")
        print(f"Probability of being REAL: {result['real_probability']:.2%}")

def setup_and_prepare(dataset_path, device):
    """
    Load data, preprocess, initialize vectorizer, and dummy model for prediction.
    Returns vectorizer and dummy model
    """
    set_seed(42)
    df = load_data(dataset_path)
    df['processed_text'] = df['text'].apply(lambda x: preprocess_text(x, remove_stopwords=True))
    label_mapping = {'real': 0, 'fake': 1}
    df['label'] = df['label'].apply(lambda l: label_mapping[l.lower()])
    X = df['processed_text'].values
    y = df['label'].values
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.1765, random_state=42, stratify=y_train_temp)
    vectorizer = TextVectorizer(max_features=10000, max_len=200, embedding_dim=100)
    X_train_seq = vectorizer.fit_transform(X_train)
    X_val_seq = vectorizer.transform(X_val)
    X_test_seq = vectorizer.transform(X_test)
    torch.save(vectorizer.word_index, "word_index.pt")
    dummy_model = BiLSTMAttention(
        vocab_size=vectorizer.vocab_size,
        embedding_dim=100,
        hidden_dim=128,
        output_dim=2,
        n_layers=2,
        dropout=0.5,
        pad_idx=0,
        embedding_matrix=vectorizer.load_embeddings()
    )
    dummy_model.to(device)
    return vectorizer, dummy_model

def train_model_by_choice(model_choice, vectorizer, device):
    """
    Train model based on choice and return stats
    """
    df = load_data("data/news.csv")
    df['processed_text'] = df['text'].apply(lambda x: preprocess_text(x, remove_stopwords=True))
    label_mapping = {'real': 0, 'fake': 1}
    df['label'] = df['label'].apply(lambda l: label_mapping[l.lower()])
    X = df['processed_text'].values
    y = df['label'].values
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.1765, random_state=42, stratify=y_train_temp)
    X_train_seq = vectorizer.transform(X_train)
    X_val_seq = vectorizer.transform(X_val)
    X_test_seq = vectorizer.transform(X_test)
    train_dataset = TensorDataset(torch.tensor(X_train_seq), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val_seq), torch.tensor(y_val))
    test_dataset = TensorDataset(torch.tensor(X_test_seq), torch.tensor(y_test))
    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=32)
    val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=32)
    test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=32)
    vocab_size = vectorizer.vocab_size
    embedding_matrix = vectorizer.load_embeddings()
    embedding_dim = 100
    hidden_dim = 128
    output_dim = 2
    n_layers = 2
    dropout = 0.5
    pad_idx = 0
    if model_choice == 'cnn':
        model = TextCNN(
            vocab_size, embedding_dim, 100, [2, 3, 4, 5], output_dim, dropout, pad_idx, embedding_matrix
        )
    elif model_choice == 'lstm':
        model = LSTMClassifier(
            vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, True, dropout, pad_idx, embedding_matrix
        )
    else:
        model = BiLSTMAttention(
            vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout, pad_idx, embedding_matrix
        )
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    trained_model, stats = train_model(
        model, train_loader, val_loader, optimizer, criterion, device, epochs=8, early_stopping_patience=3
    )
    evaluate_model(trained_model, test_loader, criterion, device)
    return stats

# Execute main function if script is run directly
if __name__ == "__main__":
    main()
    

    
