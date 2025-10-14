# [EMNLP'25] DEL-ToM: Inference-Time Scaling for Theory-of-Mind Reasoning via Dynamic Epistemic Logic

[![EMNLP 2025](https://img.shields.io/badge/EMNLP-2025-blue)](https://2025.emnlp.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2508.17348-red)](https://arxiv.org/abs/2505.17348)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)
[![Models](https://img.shields.io/badge/Hugging%20Face-Models-yellow)](https://huggingface.co/joooelw)
[![Datasets](https://img.shields.io/badge/Hugging%20Face-Datasets-yellow)](https://huggingface.co/joooelw)

This repository contains the implementation of **DEL-ToM**, designed to run on **NVIDIA H100** GPUs.  
It provides step-by-step instructions for **(1)** environment setup and **(2)** fine-tuning the **Process Belief Model (PBM)** using **Axolotl** on a curated ToM dataset.  
In addition, it includes code for **(3)** generating the PBM training data and **(4)** performing inference-time scaling using PBM together with Best-of-N (BoN).

## 1. Environment Setup

It is recommended to create a conda environment using **Python 3.11** and **PyTorch 2.7.1**, as these versions ensure full compatibility with **Axolotl**.


```bash
# Create and activate environment
conda create -n del_tom python=3.11 -y
conda activate del_tom

# Install PyTorch
pip install torch==2.7.1

# Clone the DEL-ToM repository
git clone https://github.com/joel-wu/DEL-ToM
cd DEL-ToM
```

## 2. (Optional) Train PBM with Axolotl

We fine-tuned [**meta-llama/Llama-3.1-8B-Instruct**](https://huggingface.co/joooelw/ToM-PBM-8B) and [**meta-llama/Llama-3.2-3B-Instruct**](https://huggingface.co/joooelw/ToM-PBM-3B) on our [**PBM dataset**]((https://huggingface.co/datasets/joooelw/ToM-PBM-Train)) (20,000 conversations). Here is an example about how to train a PBM.


### 2.1 Install Axolotl

```bash
pip install --no-build-isolation "axolotl[flash-attn,deepspeed]"
```

If you plan to use gated models, log in first:

```bash
huggingface-cli login
```

### 2.2 Configuration

We provide a ready-to-use **llama.yml** configuration file for full fine-tuning on **Llama-3.2-1B-Instruct** using `ChatML` formatting and training only on assistant turns.

**Key points:**
- Loads conversations from the dataset (`conversations` list with `role` and `content`).
- Uses chat_template: `chatml`.
- roles_to_train: `["assistant"]`, masks non-assistant tokens.
- Full fine-tuning by default (for LoRA, see Axolotl documentation).
- You can set your own model and output path by editing `base_model` and `output_dir` in the YAML.

### 2.3 Run Training

```bash
axolotl train llama.yml
```

That’s it - Axolotl will automatically:

1. Download the base model
2. Preprocess the dataset 
3. Start full fine-tuning 
4. Save the trained model to your specified output directory.

We have released our fine-tuned models on Hugging Face:

- **[joooelw/ToM-PBM-8B](https://huggingface.co/joooelw/ToM-PBM-8B)** (base model: *meta-llama/Llama-3.1-8B-Instruct*)
- **[joooelw/ToM-PBM-3B](https://huggingface.co/joooelw/ToM-PBM-3B)** (base model: *meta-llama/Llama-3.2-3B-Instruct*)  


Both models are trained on our PRM dataset. Later, we will download these checkpoints for evaluation and inference-time scaling experiments.

## 3. (Optional) PBM Data Generation via Simulator

If you are curious about how to **generate ToM data**, please see this section.  
Otherwise, you can simply use our released PBM models for inference.


### 3.1 Simulator Construction

Our simulator is adapted from the [Hi-ToM Dataset Repository](https://github.com/ying-hui-he/Hi-ToM_dataset).  
We mainly modify the following files to enable **belief tracking**:

- `./Hi-ToM_dataset/generate_tasks.py` 
- `./Hi-ToM_dataset/tasks.py` 

These modifications allow the simulator to record the full belief trace of each agent during task execution.

### 3.2 Generate New ToM Train/Test Questions with Belief Trace

To generate new ToM training or testing data with belief trace:

```bash
cd Hi-ToM_dataset
python generate_tasks.py -w world_xlarge.txt -n 20 -ptn=0.1
python generate_tasks.py -w world_xlarge.txt -n 20 -ptn=0.1 --tell True
```
The generated belief traces and questions will appear under the `./Hi-ToM_dataset/data_ToMh` directory.
Each JSON file contains both the full belief-state sequence and the corresponding ToM reasoning questions.

### 3.3 Generate PBM Train/Test Data via OpenAI API

If you want to recreate the PBM dataset from the simulator outputs, you can run our one-shot pipeline that:

1. calls an LLM to produce step-by-step traces,

2. aligns them with ground-truth belief states, and

3. exports PRM-style training lines.

```bash
# 1) Put your OpenAI key in the env (required)
export OPENAI_API_KEY="YOUR_KEY_HERE"

# 2) Install deps for the generator
pip install openai aiofiles tqdm tiktoken

# 3) From repo root, run the pipeline script
cd ..
python generate_PBM_dataset.py \
  --base-dir Hi-ToM_dataset/data_ToMh \
  --trace-out gpt_trace.jsonl \
  --conversations-out gpt_trace_conversations_format.jsonl \
  --prm-out ToM_PBM_test.jsonl \
  --model gpt-4o-mini \
  --concurrency 200 \
  --only-length length_1
```

- `--base-dir` points to the simulator output directory from 3.2.

- `--trace-out` stores raw LLM traces.

- `--conversations-out` stores aligned stepwise +/- conversations.

- `--prm-out` writes the final PBM-style jsonl: one line per sample as {"conversations": [...]}.

- Use `--only-length all` to include `length_1/2/3` in one run.

This will produce a PBM jsonl similar in structure to our released datasets (e.g., [**joooelw/ToM-PRM-Train**]((https://huggingface.co/datasets/joooelw/ToM-PBM-Train))). 

## 4. Inference-Time Scaling for ToM Reasoning

> Sections 2 & 3 are optional.  
> Our released PBM models datasets are available on Hugging Face, so you can directly perform inference.

### 4.1 Reasoning Trace Generation & Collection

We generate multiple candidate reasoning traces using a base LLM (e.g., Qwen) and later evaluate them with our trained PBM.

> ⚠️ **Note:** `vLLM` may conflict with Axolotl dependencies.  
> It’s recommended to create a **new virtual environment** for inference.

```bash
# Install vLLM
pip install vllm

# Inference
python bon_generation.py \
  --data_path ./Hi-ToM_dataset/Hi-ToM_data \
  --model Qwen/Qwen3-4B \
  --n_samples 256
```

Command line arguments:
- `--data_path`: Path to the Hi-ToM dataset directory.  
- `--model`: Hugging Face model name for the generator (e.g., `Qwen/Qwen3-4B`).  
- `--n_samples`: Number of candidate completions per prompt.  
- `--output_dir`: Directory to save JSONL results.  
- `--batch_size`: Batch size for generation.  
- `--temperature`: Sampling temperature (falls back to model default).  
- `--max_tokens`: Maximum number of generated tokens (falls back to model default).  

Files are saved to `--output_dir` as: `ToM_BoN_candidates_{model_name}_n{n_samples}.jsonl`


Each line is a JSON object:

```json
{
  "path": "path/to/sample.txt",
  "question": "question text",
  "answer": "ground truth answer",
  "answers": [
    "generated_answer_1",
    "generated_answer_2"
  ]
}
```

### 4.2 Scoring Reasoning Traces with PBM

Once reasoning traces are generated, we evaluate them using the PBM to identify which reasoning trajectory best aligns with ground-truth belief dynamics.

Run the PBM evaluation script:

```bash
python PBM_evaluation.py \
  --input_file outputs/ToM_BoN_candidates_Qwen_Qwen3_4B_n256.jsonl \
  --pbm_model joooelw/ToM-PBM-8B \
  --output_file outputs/Qwen3-4B_scores.jsonl \
  --n_traces 256 \
  --batch_size 128
```
Command line arguments:
- `--input_file`: Path to the generated reasoning traces.  
- `--pbm_model`: Model name for the trained PBM.  
- `--output_file`: Output file path to save PBM scoring results.  
- `--n_traces`: Number of reasoning traces to evaluate.  
- `--batch_size`: Batch size for PBM evaluation. 

> ⚠️ **Note:**  
> This script is designed for our released PBM models (`joooelw/ToM-PBM-3B`, `joooelw/ToM-PBM-8B`).  
> If you train your own PBM, make sure to **align the chat template and tokenizer format**  
> before computing scores, so that message roles and special tokens match your model’s configuration.

### 4.3 Evaluation and Visualization

Finally, we evaluate the PBM reranking performance and visualize the Pass@N improvement curves across different sample sizes.

Install dependencies:

```bash
pip install matplotlib
```

Run evaluation and plotting:

```bash
python evaluate_draw.py \
  --candidates_file outputs/ToM_BoN_candidates_Qwen_Qwen3_4B_n256.jsonl \
  --scores_file outputs/Qwen3-4B_scores.jsonl \
  --model_name "Qwen3-4B" \
  --n_values 1 4 16 64 256 1024 \
  --trials 20
```
Command line arguments:
- `--candidates_file`: Path to candidates JSONL file from bon_generation.py (required)
- `--scores_file`: Path to PBM scores JSONL file from PBM_evaluation.py (required)
- `--output_dir`: Output directory for plots (default: ./plots)
- `--model_name`: Model name for plot titles (default: Model)
- `--mode`: Evaluation mode - vanilla, weighted, or both (default: both)
- `--trials`: Number of random trials for evaluation (default: 10)
- `--n_values`: N values to test (default: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
- `--output_format`: Output format - pdf, png, or svg (default: pdf)


The script generates accuracy plots showing:
- **Majority**: Simple majority voting baseline
- **Avg**: Average score across all steps
- **Last**: Score of the final step
- **Min**: Minimum score across all steps
- **Prod**: Product of all step scores

Each plot shows:
- Accuracy vs N
- Confidence bands from multiple random trials

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{wu2025del-tom,
  title={DEL-ToM: Inference-Time Scaling for Theory-of-Mind Reasoning via Dynamic Epistemic Logic},
  author={Wu, Yuheng and Xie, Jianwen and Zhang, Denghui and Xu, Zhaozhuo},
  booktitle={Proceedings of the Conference on Empirical Methods in Natural Language Processing},
  month={November},
  year={2025}
}
```

For questions or issues, please contact [Yuheng Wu](mailto:yuhengwu@stanford.edu) or open an issue in this repository.