# Assignment 3

## Overview
This folder contains my submission for **NTHU Natural Language Processing (2025) — Assignment 3 (SemEval / BERT Multi-Task)**.

- `src/`: source code for the BERT-based multi-task model (relatedness regression + entailment classification) and training / evaluation pipeline  
- `reports/`: public version of the report.

> Note: cached datasets, model checkpoints, and generated outputs are intentionally excluded to keep the repository clean and lightweight.

---

## Reference (Assignment Spec & Public Data Links)
The official assignment specification and public data links are provided by the TA repository:
https://github.com/IKMLab/NTHU_Natural_Language_Processing/tree/main/2025/Assignments/Assignment3

---

## Dataset Notes
This assignment uses the **SemEval 2014 Task 1** dataset (loaded via Hugging Face `datasets`).  
The script will download and cache the dataset automatically under `cache/` on first run.

## Run
From the `Assignment 3` directory:
```bash
pip install -r requirements.txt
python src/NLP_HW3.py
```
