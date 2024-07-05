import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AdamW, AutoModel, AutoTokenizer, BertTokenizer, BertModel
import tokenizers
from datetime import datetime

def seed_all(seed=42):
    """
    Fix seed for reproducibility
    """
    # pytorch RNGs
    torch.manual_seed(seed)
    # numpy RNG
    np.random.seed(seed)
    
class config:
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

    BIOBERT = True
    LEARNING_RATE = 1e-5
    EPOCHS = 5

    TRAIN_FILE = '../../data/AD_CASI/csv_files/train_small_70.csv'
    VAL_FILE = '../../data/AD_CASI/csv_files/dev_small_70.csv'
    TEST_FILE = '../../data/AD_CASI/csv_files/test_small_70.csv'
    EXTERNAL_FILE = '../../data/AD_CASI/csv_files/external_data.csv'

    SAVE_DIR = '../../models/'
    DETAILS_FILE = 'details.txt'
    DICTIONARY = json.load(open('../../data/AD_CASI/casi_dictionary.json'))

    MAX_LEN = 120
    DROPOUT = 0.5
    ALPHA = 0.5
    TRAIN_BATCH_SIZE = 64
    VALID_BATCH_SIZE = 64
    OUTPUT_EMBED_SIZE = 64
    TIME_FORMAT = '%Y_%m_%d_%H_%M'
    SEED = 42
    INCLUDE_EXT = False

    if BIOBERT:
        TOKENIZER = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        MODEL_NAME = "Biobert"
    else:
        TOKENIZER = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        MODEL_NAME = "Clinicalbert"
        
def load_data(file_path):
    # Read the CSV file using pandas
    data = pd.read_csv(file_path)
    
    # Extract the necessary columns
    texts = data['text'].tolist()
    acronyms = data['acronym_'].tolist()
    expansions = data['expansion'].tolist()
    
    return texts, acronyms, expansions  

def custom_collate_fn(batch):
    # Find the maximum size of the label_mask in the batch
    max_mask_size = max([item['label_mask'].size(0) for item in batch])
    
    # Pad each item's label_mask to the maximum size
    for item in batch:
        pad_size = max_mask_size - item['label_mask'].size(0)
        item['label_mask'] = torch.cat([
            item['label_mask'],
            torch.full((pad_size,), -1e9)  # Fill the padding with large negative values
        ])
    
    # Now use the default collate function on the modified batch
    batch = torch.utils.data.dataloader.default_collate(batch)
    return batch
        
class Dataset:
    def __init__(self, text, acronym, expansions):
        self.text = text
        self.acronym = acronym
        self.expansion = expansions
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
        self.max_expansions = max(len(expansions) for expansions in config.DICTIONARY.values())  # max number of expansions
        self.acronym_expansion_dict = config.DICTIONARY

    def __len__(self):
        return len(self.text)
        
    def __getitem__(self, idx):
        text = self.text[idx]
        acronym = self.acronym[idx]
        correct_expansion = self.expansion[idx]

        # Tokenization
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=self.max_len,           # Pad & truncate all sentences.
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',      # Return PyTorch tensors
            truncation=True,
        )
        
        # input_ids = encoded['input_ids'].squeeze(0)
        # input_ids.requires_grad = True  # Enable gradient tracking
        
        # Fetch all possible expansions for the acronym from the dictionary
        possible_expansions = self.acronym_expansion_dict[acronym]
        label_indices = {exp: idx for idx, exp in enumerate(possible_expansions)}

        # Determine the label index for the correct expansion
        label_index = label_indices.get(correct_expansion, -1)  # -1 or another handling if not found
        
        # Create label mask based on possible expansions
        label_mask = torch.full((self.max_expansions,), -1e3, dtype=torch.float32)  # Initialize all to a large negative value
        valid_indices = [label_indices[exp] for exp in possible_expansions if exp in label_indices]
        label_mask[valid_indices] = 0  # Set valid indices to zero to allow them in softmax calculation

        # # Create label mask based on possible expansions
        # label_mask = torch.zeros(len(label_indices), dtype=torch.float32)
        # label_mask[list(label_indices.values())] = 1  # Only these indices are valid for this acronym
        # label_mask = -1e9 * (1 - label_mask)  # Set invalid indices to large negative value

        return {
            'input_ids': encoded['input_ids'].squeeze(0),  # Remove the batch dimension
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': torch.tensor(label_index, dtype=torch.long),
            'label_mask': label_mask
        }
        
        
class BertAD(nn.Module):
    def __init__(self, max_expansions):
        super(BertAD, self).__init__()
        if config.BIOBERT:
            self.bert = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
            self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        else:
            self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        self.dropout = nn.Dropout(config.DROPOUT)
        # Classifier transforms BERT's output to the number of possible expansions
        self.classifier = nn.Linear(self.bert.config.hidden_size, max_expansions)
              
    def forward(self, input_ids, attention_mask, label_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        logits = logits + label_mask  # Apply mask

        return logits
    
    def forward_from_embeddings(self, embeddings, attention_mask, label_mask=None):
        # Process the model forward pass using pre-computed embeddings instead of input_ids
        outputs = self.bert(inputs_embeds=embeddings, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits += label_mask  # Apply the label mask
        return logits

    
def train(model, train_dataloader, optimizer, criterion, epochs, alpha=0.5, epsilon=0.01):
    model.train()

    for epoch in range(epochs):
        total_loss = 0  # Accumulate loss over the epoch
        num_batches = 0  # Count batches to average the loss later

        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)
            label_mask = batch['label_mask'].to(config.DEVICE)

            optimizer.zero_grad()

            # Forward pass to compute the original loss
            logits = model(input_ids, attention_mask, label_mask)
            loss = criterion(logits, labels)

            # Calculate gradients of the original loss
            loss.backward(retain_graph=True)

            # Optionally compute adversarial training if gradients are available
            embeddings = model.bert.embeddings(input_ids)
            embeddings.retain_grad()  # Tell PyTorch to retain gradients for non-leaf tensor

            if embeddings.grad is not None:
                adversarial_embeddings = embeddings + epsilon * embeddings.grad.sign()
                adversarial_embeddings = torch.clamp(adversarial_embeddings, min=-1.0, max=1.0)
                adversarial_logits = model.forward_with_embeddings(adversarial_embeddings, attention_mask, label_mask)
                adversarial_loss = criterion(adversarial_logits, labels)

                total_adversarial_loss = alpha * loss + (1 - alpha) * adversarial_loss
            else:
                total_adversarial_loss = loss  # Default to regular loss if no adversarial component

            optimizer.zero_grad()
            total_adversarial_loss.backward()
            optimizer.step()

            total_loss += total_adversarial_loss.item()
            num_batches += 1

        # Output average loss for the epoch
        epoch_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}, Loss: {epoch_loss}")


        
        
def val(model, val_dataloader, criterion):
    model.eval()  # Set the model to evaluation mode

    total_eval_loss = 0

    with torch.no_grad():  # No gradients needed for validation
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)
            label_mask = batch['label_mask'].to(config.DEVICE)

            logits = model(input_ids, attention_mask, label_mask)
            loss = criterion(logits, labels)

            total_eval_loss += loss.item()
            # Calculate accuracy or other metrics as needed

    # Calculate the average loss over all validation batches
    print(f"Validation loss: {total_eval_loss / len(val_dataloader)}")
    
    
def test(model, test_dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_test_loss = 0
    predictions, true_labels = [], []

    with torch.no_grad():  # No gradients needed for testing
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)
            label_mask = batch['label_mask'].to(config.DEVICE)

            logits = model(input_ids, attention_mask, label_mask)
            loss = criterion(logits, labels)
            total_test_loss += loss.item()

            # Collect predictions and true labels for evaluation metrics
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.tolist())
            true_labels.extend(labels.tolist())

    # Calculate the average loss and evaluation metrics
    test_loss = total_test_loss / len(test_dataloader)
    print(f"Test Loss: {test_loss}")
    print(classification_report(true_labels, predictions))


if __name__ == "__main__":
    # Setup device
    print(f"Using {config.DEVICE} for training.")
    
    # Example of setting num_labels, should be set according to your specific task
    max_expansions = max(len(expansions) for expansions in config.DICTIONARY.values())
    
    print(f"Maximum number of expansions: {max_expansions}")

    # Load the model
    model = BertAD(max_expansions=max_expansions).to(config.DEVICE)

    # Load dataset
    train_texts, train_acronyms, train_expansions = load_data(config.TRAIN_FILE)
    val_texts, val_acronyms, val_expansions = load_data(config.VAL_FILE)
    test_texts, test_acronyms, test_expansions = load_data(config.TEST_FILE)

    # Initialize the Dataset
    train_dataset = Dataset(train_texts, train_acronyms, train_expansions)
    val_dataset = Dataset(val_texts, val_acronyms, val_expansions)
    test_dataset = Dataset(test_texts, test_acronyms, test_expansions)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config.VALID_BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.VALID_BATCH_SIZE, shuffle=False)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # Training loop
    print("Starting training...")
    train(model, train_dataloader, optimizer, criterion, config.EPOCHS)

    # Validation loop
    print("Starting validation...")
    val(model, val_dataloader, criterion)
    
    # Testing the model
    print("Starting testing...")
    test(model, test_dataloader, criterion)
    