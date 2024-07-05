import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
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

    TRAIN_FILE = '../../data/AD_CASI/csv_files/train_small.csv'
    VAL_FILE = '../../data/AD_CASI/csv_files/dev_small.csv'
    TEST_FILE = '../../data/AD_CASI/csv_files/test_small.csv'
    EXTERNAL_FILE = '../../data/AD_CASI/csv_files/external_data.csv'

    SAVE_DIR = '../../models/'
    DETAILS_FILE = 'details.txt'
    DICTIONARY = json.load(open('../../data/AD_CASI/casi_dictionary.json'))

    MAX_LEN = 120
    DROPOUT = 0.5
    ALPHA = 0.5
    TRAIN_BATCH_SIZE = 24
    VALID_BATCH_SIZE = 24
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
    # Extracting input_ids, attention_masks, and negative samples from the batch
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    negative_samples = [item['negative_samples'] for item in batch]

    # Padding input_ids and attention_masks to the maximum length in the batch
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    # Handling negative samples: Since they vary in number and length, pad each sublist separately
    negative_samples_padded = []
    for negatives in negative_samples:
        if negatives:  # Check if there is any negative sample
            padded_negatives = pad_sequence(negatives, batch_first=True, padding_value=0)
            negative_samples_padded.append(padded_negatives)
        else:
            negative_samples_padded.append(torch.tensor([]))  # Append an empty tensor if no negatives

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded,
        'negative_samples': negative_samples_padded
    }
        
class Dataset:
    def __init__(self, texts, acronyms, expansions):
        self.texts = texts
        self.acronyms = acronyms
        self.expansions = expansions
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
        self.max_expansions = max(len(expansions) for expansions in config.DICTIONARY.values())  # max number of expansions
        self.acronym_expansion_dict = config.DICTIONARY

    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        acronym = self.acronyms[idx]
        correct_expansion = self.expansions[idx]
        negative_expansions = [e for e in self.acronym_expansion_dict[acronym] if e != correct_expansion]

        # Tokenize the positive example
        pos_encoded = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt'
        )

        # Prepare outputs for each negative example
        neg_encoded_list = []
        for neg_exp in negative_expansions:
            neg_encoded = self.tokenizer.encode_plus(
                f"{acronym} means {neg_exp}", add_special_tokens=True, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt'
            )
            neg_encoded_list.append(neg_encoded['input_ids'].squeeze(0))

        return {
            'input_ids': pos_encoded['input_ids'].squeeze(0),
            'attention_mask': pos_encoded['attention_mask'].squeeze(0),
            'negative_samples': neg_encoded_list  # list of tensor
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
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask, negative_samples):
        # Positive sample processing
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pos_logits = self.classifier(self.dropout(outputs.pooler_output))

        # Process all negative samples at once
        if negative_samples:
            neg_input_ids = torch.cat(negative_samples, dim=0).to(input_ids.device)  # Stack and ensure device consistency
            neg_attention_mask = torch.ones_like(neg_input_ids).to(input_ids.device)  # Create an attention mask for negative samples
            
            # print("Neg input shape:", neg_input_ids.shape)  # Debugging: Check input shapes
            # print("Neg attention mask shape:", neg_attention_mask.shape)  # Debugging: Check mask shapes

            neg_outputs = self.bert(input_ids=neg_input_ids, attention_mask=neg_attention_mask)
            neg_logits = self.classifier(self.dropout(neg_outputs.pooler_output))
            neg_logits = list(neg_logits.split(1, dim=0))  # Split batched logits into a list of single tensors
        else:
            neg_logits = torch.tensor([])

        return pos_logits, neg_logits
    
    
def custom_loss_function(pos_logits, neg_logits_list, lr):
    loss = 0
    for neg_logits in neg_logits_list:
        loss += torch.log(torch.sigmoid(lr * pos_logits - lr * neg_logits))
    return -loss.mean()  # Negative of the sum of logs for optimization
    
def train(model, dataloader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            negative_samples = [ns.to(config.DEVICE).long() for ns in batch['negative_samples']]

            optimizer.zero_grad()
            pos_logits, neg_logits_list = model(input_ids, attention_mask, negative_samples)
            loss = custom_loss_function(pos_logits, neg_logits_list, config.LEARNING_RATE)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(dataloader)}")
        
def val(model, val_dataloader, device):
    model.eval()
    total_eval_accuracy = 0
    total_examples = 0

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            negative_samples = [ns.to(device) for ns in batch['negative_samples']]

            pos_logits, neg_logits_list = model(input_ids, attention_mask, negative_samples)

            pos_scores = torch.sigmoid(pos_logits)
            neg_scores = torch.sigmoid(torch.cat(neg_logits_list, dim=0))

            # Debugging: Print shapes of pos_scores and neg_scores
            print("Shape of pos_scores:", pos_scores.shape)
            print("Shape of neg_scores:", neg_scores.shape)

            # Assuming neg_scores is now a flat list of all negative scores
            # We need to reshape or split neg_scores to compare each pos_score with its corresponding neg_scores
            num_neg_per_pos = len(negative_samples[0]) if negative_samples else 0
            if num_neg_per_pos:
                neg_scores = neg_scores.view(-1, num_neg_per_pos)  # Reshape neg_scores for comparison

                # Compare pos_scores with each group of neg_scores
                correct = torch.all(pos_scores.unsqueeze(1) > neg_scores, dim=1)
                total_eval_accuracy += correct.sum().item()  # Count how many are correctly ranked

            total_examples += input_ids.size(0)

    accuracy = total_eval_accuracy / total_examples if total_examples > 0 else 0
    print(f"Validation Accuracy: {accuracy}")

    
def test(model, test_dataloader, device):
    model.eval()
    total_test_accuracy = 0
    total_examples = 0

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            negative_samples = [ns.to(device) for ns in batch['negative_samples']]

            pos_logits, neg_logits_list = model(input_ids, attention_mask, negative_samples)

            pos_scores = torch.sigmoid(pos_logits)
            neg_scores = torch.sigmoid(torch.cat(neg_logits_list, dim=0))

            # Assuming neg_scores is now a flat list of all negative scores
            # We need to reshape or split neg_scores to compare each pos_score with its corresponding neg_scores
            num_neg_per_pos = len(negative_samples[0]) if negative_samples else 0
            if num_neg_per_pos:
                neg_scores = neg_scores.view(-1, num_neg_per_pos)  # Reshape neg_scores for comparison

                # Compare pos_scores with each group of neg_scores
                correct = torch.all(pos_scores.unsqueeze(1) > neg_scores, dim=1)
                total_test_accuracy += correct.sum().item()  # Count how many are correctly ranked

            total_examples += input_ids.size(0)  # Total number of examples processed

    accuracy = total_test_accuracy / total_examples if total_examples > 0 else 0
    print(f"Test Accuracy: {accuracy}")


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
    val_dataloader = DataLoader(val_dataset, batch_size=config.VALID_BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=config.VALID_BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

    # Define loss and optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # Training loop
    print("Starting training...")
    train(model, train_dataloader, optimizer, config.EPOCHS)

    # Validation loop
    print("Starting validation...")
    val(model, val_dataloader, config.DEVICE)
    
    # Testing the model
    print("Starting testing...")
    test(model, test_dataloader, config.DEVICE)
    