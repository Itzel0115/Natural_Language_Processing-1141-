# -*- coding: utf-8 -*-
"""NLP_HW2_NCCU_111307051

# LSTM-arithmetic

## Dataset
- [Arithmetic dataset](https://drive.google.com/file/d/1cMuL3hF9jefka9RyF4gEBIGGeFGZYHE-/view?usp=sharing)
"""

# ! pip install seaborn
# ! pip install opencc
# ! pip install -U scikit-learn

import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.nn.utils.rnn
import torch.utils.data
import matplotlib.pyplot as plt
import seaborn as sns
import opencc
import os
from sklearn.model_selection import train_test_split

data_path = './data'

df_train = pd.read_csv(os.path.join(data_path, 'arithmetic_train.csv'))
df_eval = pd.read_csv(os.path.join(data_path, 'arithmetic_eval.csv'))
df_train.head()

# transform the input data to string
df_train['tgt'] = df_train['tgt'].apply(lambda x: str(x))
df_train['src'] = df_train['src'].add(df_train['tgt'])
df_train['len'] = df_train['src'].apply(lambda x: len(x))

df_eval['tgt'] = df_eval['tgt'].apply(lambda x: str(x))

"""# Build Dictionary
 - The model cannot perform calculations directly with plain text.
 - Convert all text (numbers/symbols) into numerical representations.
 - Special tokens
    - '&lt;pad&gt;'
        - Each sentence within a batch may have different lengths.
        - The length is padded with '&lt;pad&gt;' to match the longest sentence in the batch.
    - '&lt;eos&gt;'
        - Specifies the end of the generated sequence.
        - Without '&lt;eos&gt;', the model will not know when to stop generating.
"""

char_to_id = {'<pad>': 0, '<eos>': 1}

# write your code here
# Build a dictionary and give every token in the train dataset an id
# The dictionary should contain <eos> and <pad>
# char_to_id is to conver charactors to ids, while id_to_char is the opposite

next_id = 2

# ensure we're iterating over strings
for expr in df_train['src'].astype(str):
    for ch in expr:
        if ch not in char_to_id:
            char_to_id[ch] = next_id
            next_id += 1

id_to_char = {i: c for c, i in char_to_id.items()}
vocab_size = len(char_to_id)
print('Vocab size{}'.format(vocab_size))

"""# Data Preprocessing
 - The data is processed into the format required for the model's input and output. (End with \<eos\> token)

"""

# Write your code here

# Convert sequences to id-lists for model input/output. (End with <eos> token)

# make eval the same format as train (src = src + tgt, and length)
df_eval['src'] = df_eval['src'].add(df_eval['tgt'])
df_eval['len'] = df_eval['src'].apply(lambda x: len(x))

PAD_ID = char_to_id['<pad>']  # 應為 0
criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID)
EOS_ID = char_to_id['<eos>']

def seq_to_char_ids(s: str):
    s = str(s)
    return [char_to_id[ch] for ch in s] + [EOS_ID]

def seq_to_label_ids(s: str):
    s = str(s)
    # position of '=' ; everything before (including '=') should be ignored (0)
    eq_pos = s.index('=') if '=' in s else len(s) - 1
    tgt_part = s[eq_pos + 1:]  # characters after '=' (the true answer)
    return [0] * (eq_pos + 1) + [char_to_id[ch] for ch in tgt_part] + [EOS_ID]

# apply to both train and eval
for df in (df_train, df_eval):
    df['char_id_list']  = df['src'].apply(seq_to_char_ids)
    df['label_id_list'] = df['src'].apply(seq_to_label_ids)

df_train.head()

"""# Hyper Parameters

|Hyperparameter|Meaning|Value|
|-|-|-|
|`batch_size`|Number of data samples in a single batch|64|
|`epochs`|Total number of epochs to train|10|
|`embed_dim`|Dimension of the word embeddings|256|
|`hidden_dim`|Dimension of the hidden state in each timestep of the LSTM|256|
|`lr`|Learning Rate|0.001|
|`grad_clip`|To prevent gradient explosion in RNNs, restrict the gradient range|1|
"""

# hyperparams
epochs = 6
embed_dim = 256
hidden_dim = 512
grad_clip = 1.0
lr = 1e-3
weight_decay = 1e-4

"""# Data Batching
- Use `torch.utils.data.Dataset` to create a data generation tool called  `dataset`.
- The, use `torch.utils.data.DataLoader` to randomly sample from the `dataset` and group the samples into batches.

- Example: 1+2-3=0
    - Model input: 1 + 2 - 3 = 0
    - Model output: / / / / / 0 &lt;eos&gt;  (the '/' can be replaced with &lt;pad&gt;)
    - The key for the model's output is that the model does not need to predict the next character of the previous part. What matters is that once the model sees '=', it should start generating the answer, which is '0'. After generating the answer, it should also generate&lt;eos&gt;
"""

class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        # return the amount of data
        return len(self.sequences)

    def __getitem__(self, index):
        # Extract the input data x and the ground truth y from the data
        row = self.sequences.iloc[index]
        x_full = row['char_id_list']      # e.g., [ ..., '=', a1, a2, ..., <eos>]
        y_full = row['label_id_list']
        x = x_full[:-1]
        y = y_full[1:]
        return x, y

# collate function, used to build dataloader
def collate_fn(batch):
    batch_x = [torch.tensor(data[0]) for data in batch]
    batch_y = [torch.tensor(data[1]) for data in batch]
    batch_x_lens = torch.LongTensor([len(x) for x in batch_x])
    batch_y_lens = torch.LongTensor([len(y) for y in batch_y])

    # Pad the input sequence
    pad_batch_x = torch.nn.utils.rnn.pad_sequence(batch_x,
                                                  batch_first=True,
                                                  padding_value=char_to_id['<pad>'])

    pad_batch_y = torch.nn.utils.rnn.pad_sequence(batch_y,
                                                  batch_first=True,
                                                  padding_value=char_to_id['<pad>'])

    return pad_batch_x, pad_batch_y, batch_x_lens, batch_y_lens

ds_train = Dataset(df_train[['char_id_list', 'label_id_list']])

# Build dataloader of train set and eval set, collate_fn is the collate function

dl_train = torch.utils.data.DataLoader(
    ds_train, batch_size=256, shuffle=True,
    collate_fn=collate_fn, num_workers=2, pin_memory=True
)

ds_eval  = Dataset(df_eval[['char_id_list', 'label_id_list']])

dl_eval = torch.utils.data.DataLoader(
    ds_eval, batch_size=512, shuffle=False,
    collate_fn=collate_fn, num_workers=2, pin_memory=True
)

"""# Model Design

## Execution Flow
1. Convert all characters in the sentence into embeddings.
2. Pass the embeddings through an LSTM sequentially.
3. The output of the LSTM is passed into another LSTM, and additional layers can be added.
4. The output from all time steps of the final LSTM is passed through a Fully Connected layer.
5. The character corresponding to the maximum value across all output dimensions is selected as the next character.

## Loss Function
Since this is a classification task, Cross Entropy is used as the loss function.

## Gradient Update
Adam algorithm is used for gradient updates.
"""

class CharRNN(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(CharRNN, self).__init__()

        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=embed_dim,
                                            padding_idx=char_to_id['<pad>'])

        self.rnn_layer1 = torch.nn.LSTM(input_size=embed_dim,
                                        hidden_size=hidden_dim,
                                        batch_first=True)

        self.rnn_layer2 = torch.nn.LSTM(input_size=hidden_dim,
                                        hidden_size=hidden_dim,
                                        batch_first=True)

        self.linear = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_dim,
                                                          out_features=hidden_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(in_features=hidden_dim,
                                                          out_features=vocab_size))

    def forward(self, batch_x, batch_x_lens):
        return self.encoder(batch_x, batch_x_lens)

    # The forward pass of the model
    def encoder(self, batch_x, batch_x_lens):
        batch_x = self.embedding(batch_x)

        batch_x = torch.nn.utils.rnn.pack_padded_sequence(batch_x,
                                                          batch_x_lens,
                                                          batch_first=True,
                                                          enforce_sorted=False)

        batch_x, _ = self.rnn_layer1(batch_x)
        batch_x, _ = self.rnn_layer2(batch_x)

        batch_x, _ = torch.nn.utils.rnn.pad_packed_sequence(batch_x,
                                                            batch_first=True)

        batch_x = self.linear(batch_x)

        return batch_x

    def generator(self, start_char, max_len=200):

        char_list = [char_to_id[c] for c in start_char]

        next_char = None

        while len(char_list) < max_len:
            # Write your code here
            # Pack the char_list to tensor
            x = torch.tensor(char_list, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]
            x_len = torch.LongTensor([x.size(1)])

            # Input the tensor to the embedding layer, LSTM layers, linear respectively
            logits = self.encoder(x, x_len)  # [1, T, vocab]

            # Obtain the next token prediction y (last timestep logits)
            y = logits[0, x_len.item() - 1, :]  # [vocab]

            next_char =  int(torch.argmax(y).item()) # Use argmax function to get the next token prediction

            if next_char == char_to_id['<eos>']:
                break

            char_list.append(next_char)

        return [id_to_char[ch_id] for ch_id in char_list]

torch.manual_seed(2)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(2)
torch.backends.cudnn.benchmark = True  # 加速

# Write your code here. Specify a device (cuda or cpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CharRNN(vocab_size,
                embed_dim,
                hidden_dim)

PAD_ID = char_to_id['<pad>']
criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID) #Cross-entropy loss function. The loss function should ignore <pad>


optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) #  Use AdamW for Optimizer

"""# Training
1. The outer `for` loop controls the `epoch`
    1. The inner `for` loop uses `data_loader` to retrieve batches.
        1. Pass the batch to the `model` for training.
        2. Compare the predicted results `batch_pred_y` with the true labels `batch_y` using Cross Entropy to calculate the loss `loss`
        3. Use `loss.backward` to automatically compute the gradients.
        4. Use `torch.nn.utils.clip_grad_value_` to limit the gradient values between `-grad_clip` &lt; and &lt; `grad_clip`.
        5. Use `optimizer.step()` to update the model (backpropagation).
2.  After every `1000` batches, output the current loss to monitor whether it is converging.
"""

from tqdm import tqdm
from copy import deepcopy
model = model.to(device)
model.train()
i = 0
for epoch in range(1, epochs+1):
    # The process bar
    bar = tqdm(dl_train, desc=f"Train epoch {epoch}")
    for batch_x, batch_y, batch_x_lens, batch_y_lens in bar:
        # Write your code here
        # Clear the gradient
        optimizer.zero_grad()

        batch_pred_y = model(batch_x.to(device), batch_x_lens)

        # Write your code here
        # Input the prediction and ground truths to loss function
        # Back propagation

        loss = criterion(
        batch_pred_y.reshape(-1, batch_pred_y.size(-1)),
            batch_y.to(device).reshape(-1)
                )
        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip) # gradient clipping

        # Write your code here
        # Optimize parameters in the model
        optimizer.step()

        i+=1
        if i%50==0:
            bar.set_postfix(loss = loss.item())

    # Evaluate your model
    matched = 0
    total = 0
    bar_eval = tqdm(df_eval.iterrows(), total=len(df_eval), desc=f"Validation epoch {epoch}")
    for _, row in bar_eval:
        batch_x = row['src']
        batch_y = row['tgt']

        prediction = model.generator(
            (batch_x.split('=', 1)[0] + '=') if '=' in str(batch_x) else str(batch_x)
        )  # An example of using generator: model.generator('1+1=')

        # Write your code here. Input the batch_x to the model and generate the predictions
        # Check whether the prediction match the ground truths
        # Compute exact match (EM) on the eval dataset
        # EM = correct/total

        # robust to list output; strip <eos>/spaces; compare only RHS of '='
        if isinstance(prediction, list):
            prediction = ''.join(prediction)

        def rhs(s: str) -> str:
            s = str(s).replace('<eos>', '').strip()
            return s.split('=', 1)[-1]  # if no '=', returns the whole string

        pred_ans = rhs(prediction)
        gt_ans   = rhs(batch_y)

        matched += int(pred_ans == gt_ans)
        total   += 1


    print(matched/total)