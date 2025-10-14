# vLLM-based PRM scoring for Theory of Mind evaluation

import json
import re
import math
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from more_itertools import chunked

# ------------- Extract steps and final answer -------------
def extract_trace_steps(trace: str):
    return re.findall(r"## Step (\d+) ##(.*?)(?=## Step |\Z|Final Answer:)", trace, re.DOTALL)

def extract_final_answer(trace: str):
    m = re.search(r"Final Answer:\s*\[(.*?)\]", trace)
    return m.group(1).strip() if m else "Null"

# ------------- Build prompt using full prefix -------------
def build_chat_prompt(prompt_prefix: str, steps: list, upto: int, tokenizer):
    full_user_prompt = prompt_prefix + "\n\n" + "\n\n".join(
        f"## Step {k+1} ##\n{steps[k][1].strip()}" for k in range(upto + 1)
    )
    messages = [
        {"role": "user", "content": full_user_prompt},
        {"role": "assistant", "content": "+"},
    ]
    
    print(tokenizer.apply_chat_template(messages, tokenize=False))
    return tokenizer.apply_chat_template(messages, tokenize=False)



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

<trace>
"""

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

def score_prompts_batched(llm, prompts, plus_id, minus_id, batch_size=64):
    sampling = SamplingParams(max_tokens=1, temperature=0.0, logprobs=10)
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i: i + batch_size]
        results = llm.generate(batch, sampling)
        yield results, i  


# ------------- Main function: add PRM scores to candidates -------------
def annotate_prm_scores_global(candidates_path: Path, prm_ckpt: str, output_path: Path,
                                parse_sample_file, construct_prompt,
                                n_traces: int = 64, batch_size: int = 64):
    tokenizer = AutoTokenizer.from_pretrained(prm_ckpt, trust_remote_code=True)
    llm = LLM(model=prm_ckpt, enforce_eager=True)
    plus_id, minus_id = tokenizer.encode("+")[-1], tokenizer.encode("-")[-1]

    problems = []

    print("Preparing prompts...")
    all_prompts = []
    reverse_index = []

    debug_path = output_path
    debug_path.parent.mkdir(parents=True, exist_ok=True)

    processed_set = set()
    if debug_path.exists():
        with debug_path.open() as f_in:
            for line in f_in:
                try:
                    rec = json.loads(line)
                    key = (rec["problem_idx"], rec["trace_idx"], rec["step_idx"])
                    processed_set.add(key)
                except Exception:
                    continue

    with candidates_path.open() as fin:
        for problem_idx, line in enumerate(tqdm(fin, desc="Preparing Prompts")):
            item = json.loads(line)
            problems.append(item)
            story, question, answer, note = parse_sample_file(item["path"])
            prompt_prefix = construct_prompt(story, question, answer, note)
            traces = item["answers"][:n_traces]

            for t_idx, trace in enumerate(traces):
                steps = extract_trace_steps(trace)
                for s_idx in range(len(steps)):
                    key = (problem_idx, t_idx, s_idx)
                    if key in processed_set:  # 🔧 Step 2: skip if already processed
                        continue
                    full_prompt = build_chat_prompt(prompt_prefix, steps, s_idx, tokenizer)
                    full_prompt = full_prompt.rsplit("+", 1)[0].rstrip() + "\n\n"

                    all_prompts.append(full_prompt)
                    reverse_index.append(key)


    with debug_path.open("a") as fdebug:
        with tqdm(total=len(all_prompts), desc="Scoring prompts") as pbar:
            for result_batch, start in score_prompts_batched(llm, all_prompts, plus_id, minus_id, batch_size):
                batch_prompts = all_prompts[start: start + len(result_batch)]
                batch_indices = reverse_index[start: start + len(result_batch)]

                for prompt, (pidx, tidx, sidx), res in zip(batch_prompts, batch_indices, result_batch):
                    logprobs = res.outputs[0].logprobs[-1]
                    lp_plus = logprobs.get(plus_id)
                    lp_minus = logprobs.get(minus_id)

                    if lp_plus is None or lp_minus is None:
                        score = 0.5
                    else:
                        score = 1 / (1 + math.exp(lp_minus.logprob - lp_plus.logprob))

                    fdebug.write(json.dumps({
                        "problem_idx": pidx,
                        "trace_idx": tidx,
                        "step_idx": sidx,
                        "score": score,
                        "generated": res.outputs[0].text.strip(),
                    }, ensure_ascii=False) + "\n")
                fdebug.flush()
                pbar.update(len(result_batch))

def main():
    parser = argparse.ArgumentParser(description="PBM (Process-Based Model) evaluation for Theory of Mind candidates")
    parser.add_argument("--input_file", type=str, required=True,
                       help="Path to input JSONL file containing candidates")
    parser.add_argument("--pbm_model", type=str, required=True,
                       help="Path to PBM model checkpoint (local directory or Hugging Face model)")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Path to output JSONL file for PRM scores")
    parser.add_argument("--n_traces", type=int, default=64,
                       help="Number of traces to evaluate per problem (default: 64)")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for PRM scoring (default: 64)")
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"PBM Evaluation Configuration:")
    print(f"  Input file: {input_path}")
    print(f"  PBM model: {args.pbm_model}")
    print(f"  Output file: {output_path}")
    print(f"  N traces: {args.n_traces}")
    print(f"  Batch size: {args.batch_size}")
    print()
    
    # Run PRM scoring
    annotate_prm_scores_global(
        candidates_path=input_path,
        prm_ckpt=args.pbm_model,
        output_path=output_path,
        parse_sample_file=parse_sample_file,
        construct_prompt=construct_prompt,
        n_traces=args.n_traces,
        batch_size=args.batch_size,
    )
    
    print(f"PRM scoring completed! Results saved to: {output_path}")


if __name__ == "__main__":
    main()
