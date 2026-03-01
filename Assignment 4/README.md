# Assignment 4

## Overview
This folder contains my submission for **NTHU Natural Language Processing (2025) — Assignment 4 (RAG with LangChain & Ollama)**.

- `src/`: source code for the RAG (Retrieval-Augmented Generation) system, implemented in `nlp_hw4_nccu_111307051.py`.  
- `outputs/`: public version of the evaluation results (query / ground-truth / prediction).  
- `requirements.txt`: Python dependencies for reproducing the experiments.

> Note: large language models, cached datasets, model checkpoints, and other generated artifacts are intentionally excluded to keep the repository clean and lightweight.

---

## Reference (Assignment Spec & Public Data Links)
The official assignment specification and public data links are provided by the TA repository:
https://github.com/IKMLab/NTHU_Natural_Language_Processing/tree/main/2025/Assignments/Assignment4

---

## Dataset & External Resources
- **Cat facts corpus**: the script expects a `cat-facts.txt` file (one fact per line).  
  - In Colab, you can download it with the command mentioned in the script.  
  - When running locally from the `Assignment 4` directory, keep `cat-facts.txt` in the same directory (or adjust the path in the script accordingly).
- **Question–answer pairs**: the script reads `questions_answers.txt` from the working directory and evaluates the RAG system on these pairs.
These data files are not included in this repository.

---

## Environment & LLM Setup
- Python packages are listed in `requirements.txt`.  
- The script is designed to work with:
  - **Ollama** running locally (e.g., model `llama3.2:1b`).  
  - Optionally, a **Hugging Face** account and token if you want to use HF-hosted models.

If you use Hugging Face, set your token via an environment variable before running:

```bash
export HF_TOKEN="your_hf_token_here"
```

---

## Run
From the `Assignment 4` directory:

```bash
pip install -r requirements.txt
python src/nlp_hw4_nccu_111307051.py
```

Please make sure Ollama is installed and the required model (e.g., `llama3.2:1b`) has been pulled before running the script.

