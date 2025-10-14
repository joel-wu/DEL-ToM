import json
import argparse
from pathlib import Path
from tqdm import tqdm
from vllm import LLM, SamplingParams
import asyncio
import aiofiles
from more_itertools import chunked
from transformers import AutoTokenizer

# ========== Prompt Construction ==========
def construct_prompt(story: str, question: str, answer: str, note: str) -> str:
    return f"""Here is a story that unfolds in chronological order.

You will be asked a question about the story, which may involve either:
(1) Locating an object, or
(2) Inferring an agent's mental state (e.g., what A thinks B thinks C thinks).

To solve it, think step-by-step. At each step, repeat the current line from the story, then explain its effect on beliefs. Use [Null] if someone does not yet have knowledge. If a belief chain cannot be formed (e.g., some agent exited too early), freeze belief at the last available step.

<Note>
{note}

In public or private communication:
- The speaker believes the listener will believe the claim.
- If the listener exited the room earlier than the speaker, they will believe it.

If the question is zero-order (e.g., "Where is X really?"), then in each step, only track the actual location of the object (e.g., "X is in [Y]"). You do not need to track nested beliefs.

Here is an example:
<Story>
1 Amelia, Chloe, Liam, Owen and Benjamin entered the TV_room.
2 The celery is in the red_envelope.
3 Amelia made no movements and stayed in the TV_room for 1 minute.
4 Chloe lost his watch.
5 Amelia exited the TV_room.
6 Chloe moved the celery to the green_bucket.
7 Chloe exited the TV_room.
8 Liam moved the celery to the red_bathtub.
9 Liam exited the TV_room.
10 Owen made no movements and stayed in the TV_room for 1 minute.
11 Owen exited the TV_room.
12 Benjamin made no movements and stayed in the TV_room for 1 minute.
13 Benjamin exited the TV_room.
14 Amelia, Chloe, Liam, Owen and Benjamin entered the waiting_room.
15 Liam publicly claimed that celery is in the white_bathtub now.
16 Benjamin privately told Liam that the celery is in the blue_drawer now.

<Question>
Where does Owen think Liam thinks Chloe thinks the celery is?

<Trace>
## Step 1 ##
Amelia, Chloe, Liam, Owen and Benjamin entered the TV_room.  
Everyone is present, but the celery's location is still unknown.  
Owen thinks Liam thinks Chloe thinks the celery is in [Null]

## Step 2 ##
The celery is in the red_envelope.  
Everyone observes this.  
Owen thinks Liam thinks Chloe thinks the celery is in [red_envelope]

## Step 3 ##
Amelia made no movements and stayed in the TV_room for 1 minute.  
No effect.  
Owen thinks Liam thinks Chloe thinks the celery is in [red_envelope]

## Step 4 ##
Chloe lost his watch.  
Irrelevant.  
Owen thinks Liam thinks Chloe thinks the celery is in [red_envelope]

## Step 5 ##
Amelia exited the TV_room.  
Irrelevant.  
Owen thinks Liam thinks Chloe thinks the celery is in [red_envelope]

## Step 6 ##
Chloe moved the celery to the green_bucket.  
Only Chloe, Liam, Owen, Benjamin are present. They all see this move.  
Owen thinks Liam thinks Chloe thinks the celery is in [green_bucket]

## Step 7 ##
Chloe exited the TV_room.  
Chloe’s belief frozen; still [green_bucket]  
Owen thinks Liam thinks Chloe thinks the celery is in [green_bucket]

## Step 8 ##
Liam moved the celery to the red_bathtub.  
Only Liam, Owen, Benjamin present. They observe the move. Chloe not present, so her belief unchanged.  
Owen thinks Liam thinks Chloe thinks the celery is in [green_bucket]

## Step 9 ##
Liam exited the TV_room.  
No change.  
Owen thinks Liam thinks Chloe thinks the celery is in [green_bucket]

## Step 10 ##
Owen made no movements and stayed in the TV_room for 1 minute.  
Irrelevant. 
Owen thinks Liam thinks Chloe thinks the celery is in [green_bucket]

## Step 11 ##
Owen exited the TV_room.  
Owen’s belief frozen.  
Owen thinks Liam thinks Chloe thinks the celery is in [green_bucket]

## Step 12 ##
Benjamin made no movements and stayed in the TV_room for 1 minute.  
Irrelevant.  
Owen thinks Liam thinks Chloe thinks the celery is in [green_bucket]

## Step 13 ##
Benjamin exited the TV_room.  
No change.  
Owen thinks Liam thinks Chloe thinks the celery is in [green_bucket]

## Step 14 ##
Everyone entered the waiting_room.  
No effect on beliefs.  
Owen thinks Liam thinks Chloe thinks the celery is in [green_bucket]

## Step 15 ##
Liam publicly claimed that celery is in the white_bathtub now.  
Owen hears this statement. However, public speech only affects first- and second-order beliefs (e.g., what Liam believes, what Owen thinks Liam believes, and what Liam thinks Owen believes). It does not change Owen’s belief about what Liam thinks Chloe thinks.  
Owen thinks Liam thinks Chloe thinks the celery is in [green_bucket]

## Step 16 ##
Benjamin privately told Liam that the celery is in the blue_drawer now.  
Owen does not hear this, but more importantly, private communication only affects beliefs between the speaker and the listener. It can change what Liam believes (based on exit order), or what Liam thinks Benjamin believes (based on exit order), or what Benjamin thinks Liam believes (always change) — but it cannot affect higher-order beliefs. So this does not change Owen’s belief about what Liam thinks Chloe thinks.  
Owen thinks Liam thinks Chloe thinks the celery is in [green_bucket]

Final Answer: [green_bucket]

Now it's your turn.

<Story>
{story}

<Question>
{question}

Give a step-by-step trace as in the example. Then, give the final answer in one line like:  
Final Answer: [your choice]

<Trace>
"""

# ========== Parse Sample ==========
def parse_sample_file(file_path: Path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    story_lines = []
    question, answer, note = "", "", ""
    for line in lines:
        line = line.strip()

        if line.startswith("Question:"):
            question = line[len("Question:"):].strip()
        elif line.startswith("Answer:"):
            answer = line[len("Answer:"):].strip()
        elif line.startswith("Note:"):
            note = line[len("Note:"):].strip()
        elif line.startswith("The following story happens in chronological order.") or \
             line.startswith("Story:") or \
             line.startswith("Choices:"):
            # Skip intro and choices
            continue
        else:
            story_lines.append(line)

    clean_story = "\n".join(story_lines).strip()
    return clean_story, question, answer, note

# ========== Process All ==========
async def process_all_vllm(base_path: str, output_path: str, model_name: str, config: dict):
    """Process all samples using vLLM for batch generation."""
    base = Path(base_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use default cache directory or let transformers handle it
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Warning: Could not load tokenizer for {model_name}: {e}")
        tokenizer = None

    # load previous progress
    processed_paths = set()
    if out_path.exists():
        with open(out_path, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    processed_paths.add(record["path"])
                except:
                    continue

    prompts = []
    metadata = []

    for tell_type in ["Tell", "No_Tell"]:
        subdir = base / tell_type / "CoT" / "length_1"
        for sample_folder in sorted(subdir.iterdir()):
            if not sample_folder.is_dir():
                continue
            for i in range(6):
                fpath = sample_folder / f"order_{i}.txt"
                if not fpath.exists() or str(fpath) in processed_paths:
                    continue
                story, question, answer, note = parse_sample_file(fpath)
                if "Qwen3" in model_name:
                    messages = [{"role": "user", "content": construct_prompt(story, question, answer, note) + "/no_think"}]
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                elif "OLMo" in model_name or "allenai/OLMo" in model_name:
                    messages = [{"role": "user", "content": construct_prompt(story, question, answer, note)}]
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False) + "<|endoftext|>"
                else:
                    prompt=construct_prompt(story, question, answer, note)
                prompts.append(prompt)
                metadata.append({
                    "path": str(fpath),
                    "question": question,
                    "answer": answer
                })

    print(f"Total to process: {len(prompts)}")

    # ========== Generate via vLLM ==========
    llm = LLM(model=model_name, enforce_eager=True)
    sampling_param_kwargs = config["sampling_params"]
    sampling_params = SamplingParams(**sampling_param_kwargs)

    batch_size = config.get("batch_size", 16)

    # ========== Save ==========
    async with aiofiles.open(output_path, "a", encoding="utf-8") as f_out:
        for prompt_batch, meta_batch in zip(chunked(prompts, batch_size), chunked(metadata, batch_size)):
            results = llm.generate(prompt_batch, sampling_params)
            for meta, res in zip(meta_batch, results):
                completions = [o.text.strip() for o in res.outputs]
                record = {
                    **meta,
                    "answers": completions
                }
                await f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            await f_out.flush()


def get_default_config(model_name: str, n_samples: int = 1) -> dict:
    """Get default configuration for a model."""
    configs = {
        "qwen": {
            "sampling_params": {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": 8192,
                "n": n_samples,
                "top_k": 20
            },
            "batch_size": 16
        },
        "olmo": {
            "sampling_params": {
                "max_tokens": 8192 * 2,
                "n": n_samples,
            },
            "batch_size": 8
        },
        "deepseek": {
            "sampling_params": {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": 8192 * 2,
                "n": n_samples,
            },
            "batch_size": 16
        }
    }
    
    # Determine model family
    model_lower = model_name.lower()
    if "qwen" in model_lower:
        return configs["qwen"]
    elif "olmo" in model_lower:
        return configs["olmo"]
    elif "deepseek" in model_lower:
        return configs["deepseek"]
    else:
        # Default config
        return {
            "sampling_params": {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": 8192,
                "n": n_samples,
            },
            "batch_size": 16
        }


def main():
    parser = argparse.ArgumentParser(description="Generate BoN (Best-of-N) candidates for Theory of Mind evaluation")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to Hi-ToM dataset directory")
    parser.add_argument("--model", type=str, required=True,
                       help="Model name (e.g., 'Qwen/Qwen2.5-7B-Instruct' or 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')")
    parser.add_argument("--n_samples", type=int, default=1,
                       help="Number of samples to generate per prompt (default: 1)")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Output directory for results (default: ./outputs)")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size for generation (default: auto-determined by model)")
    parser.add_argument("--temperature", type=float, default=None,
                       help="Temperature for sampling (default: model-specific)")
    parser.add_argument("--max_tokens", type=int, default=None,
                       help="Maximum tokens to generate (default: model-specific)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model configuration
    config = get_default_config(args.model, args.n_samples)
    
    # Override config with command line arguments if provided
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.temperature is not None:
        config["sampling_params"]["temperature"] = args.temperature
    if args.max_tokens is not None:
        config["sampling_params"]["max_tokens"] = args.max_tokens
    
    # Generate output filename
    model_tag = args.model.replace("/", "_").replace("-", "_")
    output_path = output_dir / f"ToM_BoN_candidates_{model_tag}_n{args.n_samples}.jsonl"
    
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Data path: {args.data_path}")
    print(f"  N samples: {args.n_samples}")
    print(f"  Output: {output_path}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Sampling params: {config['sampling_params']}")
    print()
    
    # Run generation
    asyncio.run(process_all_vllm(
        base_path=args.data_path,
        output_path=str(output_path),
        model_name=args.model,
        config=config
    ))


if __name__ == "__main__":
    main()
