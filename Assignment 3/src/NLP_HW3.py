# -*- coding: utf-8 -*-
"""
Assignment 3 — SemEval 2014 Task 1 / BERT Multi-Task (relatedness + entailment).
"""

import os
import random
import numpy as np
import torch

from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from datasets import load_dataset
from evaluate import load
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
#  You can install and import any other libraries if needed

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Some Chinese punctuations will be tokenized as [UNK], so we replace them with English ones
token_replacement = [
    ["：" , ":"],
    ["，" , ","],
    [""" , "\""],
    [""" , "\""],
    [""" , "\""],
    ["？" , "?"],
    ["……" , "..."],
    ["！" , "!"]
]

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased", cache_dir="./cache/",do_lower_case=True)

class SemevalDataset(Dataset):
    def __init__(self, split="train") -> None:
        super().__init__()
        assert split in ["train", "validation", "test"]
        self.data = load_dataset(
            "sem_eval_2014_task_1", split=split, trust_remote_code=True, cache_dir="./cache/"
        ).to_list()

    def __getitem__(self, index):
        d = self.data[index]
        # Replace Chinese punctuations with English ones
        for k in ["premise", "hypothesis"]:
            for tok in token_replacement:
                d[k] = d[k].replace(tok[0], tok[1])
        return d

    def __len__(self):
        return len(self.data)

data_sample = SemevalDataset(split="train").data[:3]
print(f"Dataset example: \n{data_sample[0]} \n{data_sample[1]} \n{data_sample[2]}")

# Define the hyperparameters
# You can modify these values if needed
lr = 3e-5
epochs = 3
train_batch_size = 8
validation_batch_size = 8

def collate_fn(batch):
    # TODO1-1: Implement the collate_fn function

    # texts
    premises   = [ex["premise"] for ex in batch]
    hypotheses = [ex["hypothesis"] for ex in batch]

    # tokenize sentence pairs for BERT
    enc = tokenizer(
        premises,
        hypotheses,
        padding=True,
        truncation=True,
        max_length=192,
        return_tensors="pt",
    )

    # regression: float
    y_reg = torch.tensor(
        [float(ex["relatedness_score"]) for ex in batch],
        dtype=torch.float
    )

    # classification label
    label2id = {"CONTRADICTION": 0, "NEUTRAL": 1, "ENTAILMENT": 2}
    cls_ids = []
    for ex in batch:
        v = ex["entailment_judgment"]
        if isinstance(v, str):
            cls_ids.append(label2id[v])
        else:
            cls_ids.append(int(v))

    y_cls = torch.tensor(cls_ids, dtype=torch.long)

    return enc, y_reg, y_cls

# TODO1-2: Define your DataLoader
train_ds = SemevalDataset(split="train")
val_ds   = SemevalDataset(split="validation")
test_ds  = SemevalDataset(split="test")

dl_train = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True,  collate_fn=collate_fn)
dl_validation = DataLoader(val_ds, batch_size=validation_batch_size, shuffle=False, collate_fn=collate_fn)
dl_test = DataLoader(test_ds, batch_size=validation_batch_size, shuffle=False, collate_fn=collate_fn)

# TODO2: Construct your model
class MultiLabelModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # base encoder (BERT)
        self.bert = BertModel.from_pretrained(
            "google-bert/bert-base-uncased", cache_dir="./cache/"
        )
        hidden = self.bert.config.hidden_size

        # regularization
        self.dropout = torch.nn.Dropout(0.1)

        # two task heads
        self.reg_head = torch.nn.Linear(hidden, 1)   # relatedness_score (regression)
        self.cls_head = torch.nn.Linear(hidden, 3)   # entailment_judgement (3 classes)

    def forward(self, **kwargs):
        out = self.bert(**kwargs)
        pooled = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0]
        x = self.dropout(pooled)

        reg = self.reg_head(x).squeeze(-1)  # shape: (batch,)
        cls = self.cls_head(x)              # shape: (batch, 3)
        return reg, cls

# TODO3: Define your optimizer and loss function

model = MultiLabelModel().to(device)

# TODO3-1: Define your Optimizer
optimizer = AdamW(model.parameters(), lr=lr)

# 改進：加入 linear scheduler
total_steps = epochs * len(dl_train)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# TODO3-2: Define your loss functions (you should have two)
criterion_reg = torch.nn.MSELoss()           # for relatedness_score (regression)
criterion_cls = torch.nn.CrossEntropyLoss()  # for entailment_judgement (classification)

SAVE_DIR = "./saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

best_score = 0.0
for ep in range(epochs):
    # ===== Training =====
    pbar = tqdm(dl_train)
    pbar.set_description(f"Training epoch [{ep+1}/{epochs}]")
    model.train()

    for enc, y_reg, y_cls in pbar:
        # clear gradient
        optimizer.zero_grad()

        # move to device
        for k in enc:
            enc[k] = enc[k].to(device)
        y_reg = y_reg.to(device)
        y_cls = y_cls.to(device)

        # forward pass
        pred_reg, logits = model(**enc)

        # compute loss (sum of sub-task losses)
        loss_reg = criterion_reg(pred_reg, y_reg)
        loss_cls = criterion_cls(logits, y_cls)
        loss = loss_reg + loss_cls

        # back-propagation
        loss.backward()

        # model optimization
        optimizer.step()
        scheduler.step()

        pbar.set_postfix(
            loss=float(loss.item()),
            reg=float(loss_reg.item()),
            cls=float(loss_cls.item())
        )

    # ===== Validation =====
    pbar = tqdm(dl_validation)
    pbar.set_description(f"Validation epoch [{ep+1}/{epochs}]")
    model.eval()

    # 每個 epoch 都重新建立 metric
    psr_metric = load("pearsonr")
    acc_metric = load("accuracy")

    with torch.no_grad():
        for enc, y_reg, y_cls in pbar:
            for k in enc:
                enc[k] = enc[k].to(device)
            y_reg = y_reg.to(device)
            y_cls = y_cls.to(device)

            pred_reg, logits = model(**enc)
            pred_cls = logits.argmax(dim=-1)

            # Pearson for regression
            psr_metric.add_batch(
                predictions=pred_reg.detach().cpu().numpy(),
                references=y_reg.detach().cpu().numpy()
            )

            # Accuracy for classification
            acc_metric.add_batch(
                predictions=pred_cls.detach().cpu().numpy(),
                references=y_cls.detach().cpu().numpy()
            )

    pearson_corr = psr_metric.compute()["pearsonr"]
    accuracy = acc_metric.compute()["accuracy"]
    val_sum = pearson_corr + accuracy

    print(
        f"[Epoch {ep+1}] Val Pearson={pearson_corr:.4f} | "
        f"Val Acc={accuracy:.4f} | Sum={val_sum:.4f}"
    )

    if val_sum > best_score:
        best_score = val_sum
        torch.save(
            model.state_dict(),
            os.path.join(SAVE_DIR, "best_model.ckpt")
        )
        print(f"New best ({best_score:.4f})checkpoint saved.")

# Load the best model
model = MultiLabelModel().to(device)
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.ckpt"), weights_only=True))

pbar = tqdm(dl_test, desc="Test")
model.eval()

psr_metric = load("pearsonr")
acc_metric = load("accuracy")

with torch.no_grad():
    for enc, y_reg, y_cls in pbar:
        # move to device
        for k in enc:
            enc[k] = enc[k].to(device)
        y_reg = y_reg.to(device)
        y_cls = y_cls.to(device)

        # forward
        pred_reg, logits = model(**enc)
        pred_cls = logits.argmax(dim=-1)

        # add to metrics
        psr_metric.add_batch(
            predictions=pred_reg.detach().cpu().numpy(),
            references=y_reg.detach().cpu().numpy()
        )
        acc_metric.add_batch(
            predictions=pred_cls.detach().cpu().numpy(),
            references=y_cls.detach().cpu().numpy()
        )

# compute & report
test_pearson = psr_metric.compute()["pearsonr"]
test_acc = acc_metric.compute()["accuracy"]
print(f"Test Pearson: {test_pearson:.4f} | Test Accuracy: {test_acc:.4f}")
