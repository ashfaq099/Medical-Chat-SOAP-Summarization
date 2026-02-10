# Medical Chat SOAP Summarization – Task 03(B)

## Project Overview

This project addresses **Task 03(B): Fine-Tuning a Large Language Model for Medical Chat Summarization** from the Technical Skills Assessment. The goal is to fine-tune a pre-trained sequence-to-sequence Large Language Model (LLM) to generate concise, coherent, and clinically faithful **SOAP (Subjective, Objective, Assessment, Plan)** summaries from raw medical dialogue transcripts.



---

## Setup Instructions

### Environment

* Python 3.10+
* GPU-enabled environment recommended (Google Colab was used)


---

## Model Information

* **Base Model:** `google/flan-t5-base`
* **Architecture Type:** Sequence-to-Sequence Transformer
* **Tokenizer:** Corresponding Flan-T5 tokenizer
* **Fine-Tuning Method:** Parameter-Efficient Fine-Tuning (LoRA)

### Rationale for Model Choice

A seq2seq architecture is well-suited for transforming long-form medical dialogues into structured summaries. `Flan-T5-base` was selected because it is instruction-tuned, relatively lightweight, and effective for text-to-text tasks.

To reduce computational cost while preserving adaptation capability, **LoRA (Low-Rank Adaptation)** was used instead of full fine-tuning. This allows updating a small number of trainable parameters while keeping the base model frozen.

---

## Fine-Tuning Process

### Thought Process and Approach

1. **Understand the Objective Clearly**
   The task is not generic summarization but structured SOAP summarization. This requires preserving medical facts, compressing information, and maintaining clinical structure.

2. **Establish a Baseline**
   Before fine-tuning, the pre-trained model was evaluated on test samples to understand its default behavior and weaknesses (verbosity, inconsistent SOAP structure).

3. **Data Preparation**

   * Dataset loaded from provided CSV and Excel files
   * Data cleaned to remove empty or malformed entries
   * Text length analysis performed to understand compression ratios
   * Data converted into HuggingFace `Dataset` format

4. **Instruction-Based Prompting**
   Instead of raw dialogue-to-summary mapping, an explicit instruction prompt was added to guide the model toward SOAP-style generation.

5. **Tokenization Strategy**

   * Input dialogues truncated to a fixed maximum length
   * Target SOAP summaries tokenized separately
   * Padding handled dynamically via a data collator

6. **Fine-Tuning with LoRA**

   * LoRA applied to attention query and value matrices
   * Small batch sizes with gradient accumulation used for GPU efficiency
   * Conservative learning rate chosen for stability

### Complexity and Challenges

* Medical conversations contain domain-specific terminology and implicit context
* Maintaining strict SOAP boundaries is more difficult than free-form summarization
* Long dialogues increase the risk of information omission
* Evaluation requires both lexical and semantic assessment

---

## Evaluation Results

### Quantitative Evaluation

The fine-tuned model was evaluated using standard text-generation metrics:

* **ROUGE (ROUGE-1, ROUGE-2, ROUGE-L):** Measures lexical overlap between generated summaries and reference SOAP notes
* **BERTScore:** Measures semantic similarity using contextual embeddings, which is especially important for medical text

These metrics were selected because lexical similarity alone is insufficient for evaluating clinical correctness.

### Baseline vs Fine-Tuned Performance

* Baseline model outputs were verbose and inconsistently structured
* Fine-tuned model produced shorter, more structured SOAP summaries
* Quantitative scores improved after fine-tuning

### Qualitative Analysis

* **Good Cases:** Clear symptom extraction, correct SOAP segmentation, concise treatment plans
* **Challenging Cases:** Partial omission of objective findings or excessive generalization

Most failures occurred with long or ambiguous dialogues rather than hallucinated medical facts.

---

## Reproducibility

* All experiments are contained in a single Jupyter Notebook
* Evaluation outputs saved as JSON and CSV files

This setup ensures the project can be reproduced in any compatible environment.

---
## Libraries Used

- **torch** – Core deep learning framework used for model execution, training, and GPU acceleration.
- **transformers** – HuggingFace library used for loading the FLAN-T5 model, tokenizer, `Seq2SeqTrainer`, training arguments, and text generation.
- **datasets** – Used to convert CSV/Excel files into HuggingFace `Dataset` and `DatasetDict` formats, enabling efficient preprocessing and batching.
- **peft** – Used to apply LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning of the seq2seq model without updating all weights.
- **accelerate** – Handles device placement and optimization under the hood during training.
- **bitsandbytes** – Installed to enable efficient low-precision operations (used implicitly for memory-efficient training setups).
- **evaluate** – HuggingFace evaluation framework used to compute ROUGE metrics in a standardized way.
- **rouge-score** – Used for ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum evaluation of generated SOAP summaries.
- **bert-score** – Used to compute BERTScore (Precision, Recall, F1) to evaluate semantic similarity beyond lexical overlap.
- **sentencepiece** – Required for tokenizer support for the FLAN-T5 model family.
- **pandas** – Used for data loading (CSV/Excel), cleaning, analysis, and saving detailed prediction results.
- **numpy** – Used for numerical operations, statistical analysis, and metric aggregation.
- **matplotlib** – Used for visualizing evaluation results and performance distributions.
- **seaborn** – Used to enhance visualization aesthetics and readability.
- **openpyxl** – Used to read Excel-based validation and test datasets.
