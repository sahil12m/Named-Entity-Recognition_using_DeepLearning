import torch
import torch.nn as nn
import numpy as np
import itertools
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from conlleval import evaluate
from tqdm import tqdm
torch.manual_seed(1)
np.random.seed(1)

import datasets

dataset = datasets.load_dataset("conll2003")

# Initialize the word_dict dictionary
word_dict = {'[PAD]': 0, '[UNK]': 1}

# Initialize the embedding_matrix list
embedding_matrix = []

# Open and read the glove file
with open('glove.6B.100d.txt', 'r', encoding='utf-8') as file:
    for line in file:
        # Split the line into words
        line_words = line.split()

        # Add the word to the dictionary and its corresponding embedding to the list
        word_dict[line_words[0]] = len(word_dict)
        embedding_matrix.append([float(x) for x in line_words[1:]])

embedding_dimension = 100

# Insert zero vector at the beginning of embedding_matrix
embedding_matrix.insert(0, np.zeros(embedding_dimension))

# Insert the average vector at the beginning of embedding_matrix
embedding_matrix.insert(1, np.average(np.asarray(embedding_matrix), axis=0))
        
# Iterate through the keys in word_dict
for key in list(word_dict.keys()):
    # Check if the key is alphabetic
    if key.isalpha():
        # Check if the capitalized form is not in the dictionary
        if key.capitalize() not in word_dict.keys():
            # Add the capitalized form to the dictionary and its corresponding vector to embedding_matrix
            word_dict[key.capitalize()] = len(word_dict)
            embedding_matrix.append(embedding_matrix[word_dict[key]])
        
        # Check if the uppercase form is not in the dictionary
        if key.upper() not in word_dict.keys():
            # Add the uppercase form to the dictionary and its corresponding vector to embedding_matrix
            word_dict[key.upper()] = len(word_dict)
            embedding_matrix.append(embedding_matrix[word_dict[key]])

# Convert embedding_matrix to a NumPy array
embedding_matrix = np.asarray(embedding_matrix)

# Preprocess the dataset using the provided word2idx mapping
def preprocess_sample_glove(sample):
    # Convert tokens to their respective indexes using w2ids
    glove_input_ids = [word_dict.get(word, word_dict['[UNK]']) for word in sample['tokens']]
    
    # Update the sample with 'input_ids'
    sample['glove_input_ids'] = glove_input_ids
    
    # Remove 'pos tags' and 'chunk tags'
    sample.pop('pos_tags', None)
    sample.pop('chunk_tags', None)
    sample.pop('id', None)
    
    # Rename 'ner_tags' to 'labels'
    sample['labels'] = sample.pop('ner_tags')
    
    return sample

# Apply the preprocessing using .map() function
preprocessed_glove_dataset = dataset.map(preprocess_sample_glove)

# Assuming you have a preprocessed train, test, and validation dataset
test_dataset = preprocessed_glove_dataset['test']

# Define the special label for 'PAD'
PAD_LABEL = 9

# Create custom collate function for DataLoader
def custom_collate(batch):
    # Separate input_ids and labels
    glove_input_ids = [torch.tensor(item['glove_input_ids']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]
    input_id_orig = [len(terms) for terms in glove_input_ids]
    
    # Pad input_ids and labels using pad_sequence
    glove_input_ids = pad_sequence(glove_input_ids, batch_first=True, padding_value=word_dict['[PAD]'])
    labels = pad_sequence(labels, batch_first=True, padding_value=PAD_LABEL)

    return {'glove_input_ids': glove_input_ids, 'labels': labels, 'input_id_orig': input_id_orig}

# Create DataLoader for train, test, and validation datasets
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), collate_fn=custom_collate)

# Define the BiLSTM model
class BiLSTMModel(nn.Module):
    def __init__(self, glove_embedding_matrix, embedding_dim, num_lstm_layers, lstm_hidden_dim, linear_output_dim, tagset_size):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(glove_embedding_matrix), freeze=False)
        self.bilstm = nn.LSTM(embedding_dim, lstm_hidden_dim, num_layers=num_lstm_layers, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(2 * lstm_hidden_dim, linear_output_dim)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(linear_output_dim, tagset_size)
        self.dropout = nn.Dropout(p=0.33)  # Adjust the dropout rate as needed
        
    def forward(self, glove_input_ids):
        embeddings = self.embedding(glove_input_ids)
        # Ensure the data type of the embeddings matches the expected data type for the LSTM layer
        embeddings = embeddings.to(torch.float32)  # Change torch.float32 to the correct data type
        lstm_out, _ = self.bilstm(embeddings)
        lstm_out = self.dropout(lstm_out)  # Apply dropout to the LSTM output
        linear_out = self.linear(lstm_out)
        elu_out = self.elu(linear_out)
        logits = self.classifier(elu_out)
        return logits
    
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# Define hyperparameters
glove_embedding_matrix = embedding_matrix
tagset_size = 9
embedding_dim = 100
num_lstm_layers = 1
lstm_hidden_dim = 256
linear_output_dim = 128
learning_rate = 0.001
num_epochs = 100  # You can adjust the number of epochs

# Create BiLSTM model
model = BiLSTMModel(glove_embedding_matrix, embedding_dim, num_lstm_layers, lstm_hidden_dim, linear_output_dim, tagset_size)
model.to(device)

idx_to_tag = {0:'O', 1:'B-PER', 2:'I-PER', 3:'B-ORG', 4:'I-ORG', 5:'B-LOC', 6:'I-LOC', 7:'B-MISC', 8:'I-MISC'}

# Load the state dictionary
model.load_state_dict(torch.load('task2_model.pt', map_location=torch.device('cpu')))
model.eval()

with torch.no_grad():
    preds = []
    real_labels = []
    for batch in tqdm(test_loader):
        test_glove_input_ids, test_labels = batch['glove_input_ids'].to(device, dtype=torch.long), batch['labels'].to(device, dtype=torch.long)
        logits = model(test_glove_input_ids)

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
print('Evaluation on test data for TASK2:\n')
metrics = evaluate(real_labels, preds)
