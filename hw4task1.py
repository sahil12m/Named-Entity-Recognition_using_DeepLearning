import torch
import torch.nn as nn
import numpy as np
import itertools
from collections import Counter
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from conlleval import evaluate
from tqdm import tqdm
torch.manual_seed(1)
np.random.seed(1)

import datasets

dataset = datasets.load_dataset("conll2003")

word_freq = Counter(itertools.chain(*dataset['train']['tokens']))

word_freq = {
    word: frequency
    for word, frequency in word_freq.items()
    if frequency >= 2
}

w2ids = {
    word: index
    for index, word in enumerate(word_freq.keys(), start=2)
}

w2ids['[PAD]'] = 0
w2ids['[UNK]'] = 1

# Preprocess the dataset using the provided word2idx mapping
def preprocess_sample(sample):
    # Convert tokens to their respective indexes using w2ids
    input_ids = [w2ids.get(word, w2ids['[UNK]']) for word in sample['tokens']]
    
    # Update the sample with 'input_ids'
    sample['input_ids'] = input_ids
    
    # Remove 'pos tags' and 'chunk tags'
    sample.pop('pos_tags', None)
    sample.pop('chunk_tags', None)
    sample.pop('id', None)
    
    # Rename 'ner_tags' to 'labels'
    sample['labels'] = sample.pop('ner_tags')
    
    return sample

# Apply the preprocessing using .map() function
preprocessed_dataset = dataset.map(preprocess_sample)

# Assuming you have a preprocessed train, test, and validation dataset
test_dataset = preprocessed_dataset['test']

# Define the special label for 'PAD'
PAD_LABEL = 9

# Create custom collate function for DataLoader
def custom_collate(batch):
    # Separate input_ids and labels
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]
    input_id_orig = [len(terms) for terms in input_ids]
    
    # Pad input_ids and labels using pad_sequence
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=w2ids['[PAD]'])
    labels = pad_sequence(labels, batch_first=True, padding_value=PAD_LABEL)

    return {'input_ids': input_ids, 'labels': labels, 'input_id_orig': input_id_orig}

# Create DataLoader for train, test, and validation datasets
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), collate_fn=custom_collate)

idx_to_tag = {0:'O', 1:'B-PER', 2:'I-PER', 3:'B-ORG', 4:'I-ORG', 5:'B-LOC', 6:'I-LOC', 7:'B-MISC', 8:'I-MISC'}

# Define the BiLSTM model
class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_lstm_layers, lstm_hidden_dim, linear_output_dim, tagset_size):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, lstm_hidden_dim, num_layers=num_lstm_layers, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(2 * lstm_hidden_dim, linear_output_dim)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(linear_output_dim, tagset_size)
        self.dropout = nn.Dropout(p=0.33)  # Adjust the dropout rate as needed
        
    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        lstm_out, _ = self.bilstm(embeddings)
        lstm_out = self.dropout(lstm_out)  # Apply dropout to the LSTM output
        linear_out = self.linear(lstm_out)
        elu_out = self.elu(linear_out)
        logits = self.classifier(elu_out)
        return logits
    
# Define hyperparameters
vocab_size = len(w2ids)
tagset_size = 9
embedding_dim = 100
num_lstm_layers = 1
lstm_hidden_dim = 256
linear_output_dim = 128
learning_rate = 0.01
num_epochs = 100  # You can adjust the number of epochs

# Create BiLSTM model
model = BiLSTMModel(vocab_size, embedding_dim, num_lstm_layers, lstm_hidden_dim, linear_output_dim, tagset_size)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the state dictionary
model.load_state_dict(torch.load('task1_model.pt', map_location=torch.device('cpu')))
model.eval()
# Move the model to the same device as the input data (cuda or cpu)
model.to(device)

with torch.no_grad():
    preds = []
    real_labels = []
    for batch in tqdm(test_loader):
        test_input_ids, test_labels = batch['input_ids'].to(device, dtype=torch.long), batch['labels'].to(device, dtype=torch.long)
        logits = model(test_input_ids)

        predictions = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
        real_val_labels = test_labels.cpu().numpy().tolist()

        for temp in range(len(batch['input_id_orig'])):
            preds.append(predictions[temp][:batch['input_id_orig'][temp]])
            real_labels.append(real_val_labels[temp][:batch['input_id_orig'][temp]])
            
preds = list(itertools.chain(*preds))
real_labels = list(itertools.chain(*real_labels))

preds = [idx_to_tag[prediction] for prediction in preds]
real_labels = [idx_to_tag[label] for label in real_labels]

# Evaluate on validation data and print the results
print('Evaluation on test data for TASK1:\n')
metrics = evaluate(real_labels, preds)
