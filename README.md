# Medical Chat SOAP Summarization 



## Project Overview

This project addresses **Fine-Tuning a Large Language Model for Medical Chat Summarization** . The objective is to fine-tune a pre-trained sequence-to-sequence Large Language Model (LLM) to generate concise, coherent, and clinically faithful **SOAP (Subjective, Objective, Assessment, Plan)** summaries from raw medical dialogue transcripts.



## Dataset Information

### Source
- **Dataset:** Medical Chat Summarization Dataset
- **Format:** CSV and Excel files containing dialogue-summary pairs
- **Columns:** `dialogue` (input) and `soap_summary` (target)

### Dataset Split

| Split | Samples | Purpose |
|-------|---------|---------|
| **Training** | 9,250 | Model fine-tuning |
| **Validation** | 500 | Hyperparameter tuning |
| **Test** | 250 | Final evaluation |
| **Total** | 10,000 | Complete dataset |

### Data Structure
```
project/
├── medical_dialogue_train.csv     # Training data
├── medical_dialogue_val.xlsx      # Validation data
└── medical_dialogue_test.xlsx     # Test data
```


---

## Setup Instructions

### Prerequisites
- **Python:** 3.10+
- **Hardware:** GPU with CUDA support (Google Colab T4 used)


### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/ashfaq099/Technical-Skills-Assessment_UIU.git
cd Technical-Skills-Assessment_UIU

# Mount Google Drive (if using Colab)
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA 12.1 support
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core libraries
pip install -U transformers datasets accelerate peft bitsandbytes

# Install evaluation libraries
pip install -U evaluate rouge-score bert-score sentencepiece

# Install data processing libraries
pip install -U pandas numpy matplotlib seaborn openpyxl
```

### 3. Verify Installation

```python
import torch
import transformers
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Transformers version: {transformers.__version__}")
```

### 4. Download Dataset

Place the following files in your working directory:
- `medical_dialogue_train.csv`
- `medical_dialogue_val.xlsx`  
- `medical_dialogue_test.xlsx`

### 5. Run the Notebook

```bash
# If using Jupyter
jupyter notebook Ashfaqur_Rahman_Task_3B.ipynb

# If using Google Colab
# Upload the notebook and run all cells
```

---

## Model Information

### Base Model
- **Model Name:** `google/flan-t5-base`
- **Architecture:** Sequence-to-Sequence Transformer (Encoder-Decoder)
  
- **Source:** [Hugging Face Model Hub](https://huggingface.co/google/flan-t5-base)

- **Base Parameters:** 247.58 million parameters
- **With LoRA Adapters:** 249.35 million total parameters

**Trainable Parameters:** 1.77M out of 249.35M total (0.71%)

### Why FLAN-T5-base?

1. **Instruction-Tuned:** Pre-trained on instruction-following tasks, ideal for structured output generation
2. **Seq2Seq Architecture:** Natural fit for dialogue → summary transformation
3. **Balance:** Good performance with manageable computational requirements
4. **Medical Capability:** Strong zero-shot performance on medical text

### Tokenizer Configuration
- **Max Input Length:** 512 tokens
- **Max Target Length:** 256 tokens
- **Padding:** Dynamic (handled by DataCollator)
- **Truncation:** Applied to both inputs and targets

---

## Fine-Tuning Process

### Overview

The fine-tuning approach uses **LoRA (Low-Rank Adaptation)** for parameter-efficient training, updating only ~0.65% of model parameters while maintaining performance.

### Step-by-Step Methodology

#### 1. Data Preparation
- **Loading:** Read CSV/Excel files using pandas
- **Cleaning:** Remove empty or malformed entries
- **Instruction Prompting:** Add explicit SOAP instruction template
  ```
  "Summarize the following medical dialogue into SOAP format:\n{dialogue}"
  ```
- **Train/Val/Test Split:** 9250/500/250 samples
- **Tokenization:** Apply FLAN-T5 tokenizer with max lengths

#### 2. Model Setup
- **Base Model:** Load `google/flan-t5-base` with float16 precision
- **LoRA Application:** Attach LoRA adapters to attention layers
- **Device Mapping:** Automatic GPU placement

#### 3. LoRA Configuration

```python
LoraConfig(
    r=16,                          # Rank of update matrices
    lora_alpha=32,                 # Scaling factor (2x rank)
    target_modules=["q", "v"],     # Apply to Query and Value projections
    lora_dropout=0.05,             # Regularization
    bias="none",                   # No bias adaptation
    task_type="SEQ_2_SEQ_LM"      # Sequence-to-sequence task
)
```

**Trainable Parameters:** Only 1.77M out of 249.35M total (0.71%)

#### 4. Training Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Epochs** | 3 | Sufficient for convergence |
| **Batch Size (per device)** | 4 | GPU memory constraint |
| **Gradient Accumulation Steps** | 4 | Effective batch size = 16 |
| **Learning Rate** | 1e-4 | Conservative for stability |
| **Weight Decay** | 0.01 | L2 regularization |
| **Warmup Ratio** | 0.1 | 10% warmup steps |
| **Max Gradient Norm** | 1.0 | Gradient clipping |
| **FP16 Precision** | False | FP32 for stability (testing) |
| **Evaluation Strategy** | steps | Every 250 steps |
| **Save Strategy** | steps | Save every 250 steps |
| **Best Model Selection** | eval_loss | Keep lowest validation loss |
| **Random Seed** | 42 | Reproducibility |

#### 5. Training Details
- **Optimizer:** AdamW
- **Scheduler:** Linear warmup + decay
- **Total Training Steps:** ~1,735 steps (3 epochs × 578 steps/epoch)
- **Training Time:** 0.75 hours (~45 minutes on Google Colab T4 GPU)
- **Final Training Loss:** 5.9547

#### 6. Generation Configuration
```python
GenerationConfig(
    max_length=256,           # Maximum summary length
    num_beams=4,              # Beam search with 4 beams
    early_stopping=True,      # Stop when all beams finish
    temperature=0.7,          # Not used (do_sample=False)
    do_sample=False           # Deterministic beam search
)
```

### Complexity and Challenges

#### Challenges Encountered

1. **Class Imbalance in SOAP Components**
   - Some SOAP sections (Plan) are more difficult to extract
   - Assessment sections most consistently captured (71% coverage)
   - Plan sections least consistent (17% coverage)

2. **Long Dialogue Handling**
   - Some dialogues exceed 512 token limit
   - Truncation may lose critical information
   - Mitigation: Careful truncation strategy preserving start/end

3. **Medical Terminology**
   - Domain-specific vocabulary requires careful handling
   - Base model's instruction tuning helps but not perfect
   - Some terms require contextual understanding

4. **SOAP Structure Consistency**
   - Maintaining strict S/O/A/P formatting across diverse dialogues
   - Balance between structure adherence and information completeness

---

## Evaluation Results

### Quantitative Metrics

#### ROUGE Scores (Lexical Overlap)

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **ROUGE-1** | 0.5033 | 50.3% unigram overlap with references |
| **ROUGE-2** | 0.3003 | 30.0% bigram overlap (phrasal similarity) |
| **ROUGE-L** | 0.3688 | 36.9% longest common subsequence |
| **ROUGE-Lsum** | 0.3689 | 36.9% summary-level LCS |

**Interpretation:**
- ROUGE-1  indicates good keyword capture
- ROUGE-2  shows decent phrasal consistency
- ROUGE-L captures structural similarity

#### BERTScore (Semantic Similarity)

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Precision** | 0.9076 | 90.8% of generated content is semantically relevant |
| **Recall** | 0.8645 | 86.5% of reference content is captured |
| **F1** | 0.8855 | 88.6% overall semantic alignment |

**Interpretation:**
- F1 (0.88) indicates strong semantic preservation
- High precision (90.8%) means low hallucination rate
- Good recall (86.5%) shows comprehensive information capture

### SOAP Component Coverage (Test Set)

Analysis of 100 test samples:

| SOAP Section | Coverage | Notes |
|--------------|----------|-------|
| **Subjective (S)** | 40% | Patient complaints and history |
| **Objective (O)** | 55% | Physical exam findings and vitals |
| **Assessment (A)** | 71% | Diagnosis and clinical impression |
| **Plan (P)** | 17% | Treatment recommendations |

**Key Insight:** Assessment is most consistently captured, while Plan generation needs improvement.


---

## Qualitative Analysis

### Example 1: Good Summary (High ROUGE-L Score)

**Performance:** ROUGE-L > 0.60

**Analysis:**
- All SOAP components clearly identified
- Medical terminology preserved accurately
- Logical flow maintained (S → O → A → P)
- Concise while retaining critical details
- No hallucinated information

**Success Factors:**
- Clear, well-structured input dialogue
- Explicit mention of symptoms, findings, diagnosis, and plan
- Moderate length (within token limits)

### Example 2: Challenging Case (Low ROUGE-L Score)

**Performance:** ROUGE-L < 0.20

**Analysis:**
-  Subjective section captured but incomplete
-  Objective findings partially missed
-  Assessment identified correctly
-  Plan section very brief or missing
- Some medical details lost in compression

**Failure Factors:**
- Very long input dialogue (>600 tokens before truncation)
- Implicit information requiring inference
- Multiple interleaved topics in conversation
- Plan discussed informally at end (truncated)

### Common Patterns

**What Works Well:**
1. Dialogues with clear chief complaint stated early
2. Explicit vital signs and physical exam findings
3. Direct diagnostic statements by physician
4. Structured treatment plans with specific medications

**What Needs Improvement:**
1. Long, meandering conversations with topic shifts
2. Implicit information (patient history, social context)
3. Complex differential diagnoses
4. Detailed multi-step treatment algorithms

**Note:** Detailed baseline vs fine-tuned examples, full evaluation tables, and per-sample results are available in the notebook: `Ashfaqur_Rahman_Task_3B.ipynb`.

---

## Output Files

After running the notebook, the following files are generated:

### Generated Files

```
outputs/
├── evaluation_results.json          # Quantitative metrics (ROUGE, BERTScore)
├── detailed_predictions.csv         # All test predictions with references
├── evaluation_results.png           # Visualization of score distributions
└── model/
    └── medical_soap_finetuned/      # Fine-tuned LoRA adapters
        ├── adapter_config.json      # adapter_model.safetensors 
        ├── adapter_config.json     # LoRA settings
        ├── tokenizer              # Tokenization files
        ├── training_args.json      # Hyperparameters
        └── training_results.json   # Metrics
```

### File Descriptions

| File | Description | 
|------|-------------|
| `evaluation_results.json` | JSON with all metric scores and metadata | 
| `detailed_predictions.csv` | CSV with input, generated, reference, and ROUGE scores | 
| `evaluation_results.png` | 4-panel visualization (score distributions, component coverage) | 
| `adapter_model.safetensors` | LoRA adapter weights (only 1.6M parameters) | 

### Accessing Results

```python
import json
import pandas as pd

# Load evaluation metrics
with open('evaluation_results.json', 'r') as f:
    results = json.load(f)
print(f"ROUGE-1: {results['rouge_scores']['rouge1']}")

# Load detailed predictions
predictions_df = pd.read_csv('detailed_predictions.csv')
print(predictions_df[['generated', 'reference', 'rouge_l_score']].head())
```

---

## Reproducibility

### Ensuring Reproducibility

#### 1. Random Seeds
```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)  # Applied throughout the notebook
```


#### 3. Hardware Specifications
- **Platform:** Google Colab
- **GPU:** Tesla T4 (16GB VRAM)
- **RAM:** 12GB system memory
- **CUDA:** 12.1

#### 4. Expected Runtime
- **Data Loading:** ~30 seconds
- **Model Initialization:** ~2 minutes
- **Training (3 epochs):** ~45 minutes
- **Evaluation (100 samples):** ~5 minutes
- **Total:** ~55-60 minutes



---

## How to Run

### Option 1: Google Colab (Recommended)

1. **Upload notebook** to Google Colab
2. **Mount Google Drive:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. **Upload datasets** to Colab or Drive
4. **Run all cells** sequentially (Runtime → Run all)
5. **Results** will be saved to Google Drive

### Option 2: Local Environment

```bash
# 1. Install dependencies
pip install -U torch transformers datasets peft accelerate evaluate rouge-score bert-score sentencepiece pandas numpy matplotlib seaborn openpyxl

# 2. Launch Jupyter
jupyter notebook Ashfaqur_Rahman_Task_3B.ipynb

# 3. Execute all cells
# Results will be saved to ./outputs/
```

### Option 3: Command Line (if converted to script)

```bash
python train_soap_summarizer.py \
    --train_file medical_dialogue_train.csv \
    --val_file medical_dialogue_val.xlsx \
    --test_file medical_dialogue_test.xlsx \
    --output_dir ./model/medical_soap_finetuned \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 1e-4
```

---

## Libraries Used

This project utilizes the following libraries:

### Core Deep Learning
- **torch** – PyTorch framework for model execution, training, and GPU acceleration
- **transformers** – HuggingFace library for FLAN-T5 model, tokenizer, `Seq2SeqTrainer`, and generation

### Data & Efficiency
- **datasets** – HuggingFace datasets for efficient data loading and preprocessing
- **peft** – Parameter-Efficient Fine-Tuning (LoRA implementation)
- **accelerate** – Device management and distributed training utilities
- **bitsandbytes** – Memory-efficient operations and quantization support

### Evaluation
- **evaluate** – HuggingFace evaluation framework for standardized metrics
- **rouge-score** – ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum evaluation
- **bert-score** – Semantic similarity evaluation using contextual embeddings

### Data Processing & Visualization
- **pandas** – Data loading, cleaning, and results analysis
- **numpy** – Numerical operations and statistical computations
- **matplotlib** – Plotting and visualization
- **seaborn** – Enhanced visualization aesthetics
- **openpyxl** – Reading Excel validation/test datasets

### Utilities
- **sentencepiece** – Tokenizer backend for FLAN-T5
- **warnings** – Suppress non-critical warnings for cleaner output
- **datetime** – Timestamp generation for results

---


## Acknowledgments

- **Dataset:** Medical Chat Summarization dataset (publicly available)
- **Base Model:** Google FLAN-T5-base ([Hugging Face](https://huggingface.co/google/flan-t5-base))
- **Framework:** HuggingFace Transformers and PEFT libraries
- **Compute:** Google Colab (Tesla T4 GPU)


---


