"""
generate_PBM_dataset.py

Stages:
  1) Walk Hi-ToM outputs (order_*.txt) and async-call OpenAI to produce step-by-step traces -> trace.jsonl
  2) Align GPT traces with ground-truth trace.jsonl to build stepwise +/- conversations -> conversations.jsonl
  3) Export PRM training lines {"conversations": [...]} and shuffle -> ToM_PRM.jsonl

Usage (one command):
python generate_PBM_dataset.py \
  --base-dir Hi-ToM_dataset/data_ToMh \
  --trace-out trace.jsonl \
  --conversations-out conversations.jsonl \
  --prm-out ToM_PRM.jsonl \
  --model gpt-4o-mini \
  --concurrency 200 \
  --only-length length_1

Environment:
  export OPENAI_API_KEY="..."
"""

import os
import re
import json
import argparse
import asyncio
import random
from pathlib import Path
from typing import Tuple, Dict, Any, List

import aiofiles
from tqdm import tqdm

# OpenAI async client
try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None

# Optional token counting
try:
    import tiktoken
except Exception:
    tiktoken = None


DEFAULT_MODEL = "gpt-4o-mini"

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

def parse_sample_file(file_path: Path) -> Tuple[str, str, str, str]:
    """Read order_*.txt and extract (story, question, answer, note)."""
    story_lines, question, answer, note = [], "", "", ""
    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("Question:"):
                question = line[len("Question:"):].strip()
            elif line.startswith("Answer:"):
                answer = line[len("Answer:"):].strip()
            elif line.startswith("Note:"):
                note = line[len("Note:"):].strip()
            elif line.startswith("The following story happens in chronological order.") or \
                 line.startswith("Story:") or \
                 line.startswith("Choices:"):
                continue
            else:
                story_lines.append(line)
    return "\n".join(story_lines).strip(), question, answer, note

async def get_gpt_trace(prompt: str, model: str, client: "AsyncOpenAI") -> str:
    """Single chat completion call."""
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return resp.choices[0].message.content.strip()

async def generate_traces(
    base_dir: Path,
    output_file: Path,
    model: str = DEFAULT_MODEL,
    concurrency: int = 200,
    only_length: str | None = "length_1",
) -> None:
    """Walk {Tell,No_Tell}/CoT/{length}/sample_*/order_*.txt and write GPT traces to jsonl."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set env var OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", os.getenv("OPENAI_API_BASE", "")).strip() or None
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # incremental resume: skip paths already present
    processed = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    processed.add(rec.get("path", ""))
                except Exception:
                    pass

    sem = asyncio.Semaphore(concurrency)
    tasks = []

    async def handle_file(fpath: Path):
        async with sem:
            story, question, answer, note = parse_sample_file(fpath)
            prompt = construct_prompt(story, question, answer, note)
            try:
                trace = await get_gpt_trace(prompt, model=model, client=client)
                result = {
                    "path": str(fpath),
                    "story": story,
                    "question": question,
                    "answer": answer,
                    "gpt_trace": trace
                }
            except Exception as e:
                result = {"path": str(fpath), "error": str(e)}
            async with aiofiles.open(output_file, "a", encoding="utf-8") as out:
                await out.write(json.dumps(result, ensure_ascii=False) + "\n")

    for tell_type in ["Tell", "No_Tell"]:
        subdir = base_dir / tell_type / "CoT"
        if not subdir.exists():
            continue
        for length in ["length_1", "length_2", "length_3"]:
            if only_length and length != only_length:
                continue
            length_dir = subdir / length
            if not length_dir.exists():
                continue
            for sample_folder in sorted(length_dir.iterdir()):
                if not sample_folder.is_dir():
                    continue
                for i in range(6):
                    fpath = sample_folder / f"order_{i}.txt"
                    if fpath.exists() and str(fpath) not in processed:
                        tasks.append(handle_file(fpath))

    print(f"[Trace] Pending files: {len(tasks)}")
    pbar = tqdm(total=len(tasks), desc="Generating GPT traces")
    for coro in asyncio.as_completed(tasks):
        await coro
        pbar.update(1)
    pbar.close()

# ---------- Alignment & conversation building ----------

def load_gt_traces(gt_entry_path: str) -> List[Dict[str, Any]]:
    """Read ground-truth trace.jsonl next to the order_*.txt file and sort by time t."""
    trace_path = Path(gt_entry_path).with_name("trace.jsonl")
    if not trace_path.exists():
        raise FileNotFoundError(f"Ground-truth trace file not found: {trace_path}")
    traces = []
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                traces.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    traces.sort(key=lambda x: x.get("t", -1))
    return traces

def parse_question(question: str) -> Tuple[str, int]:
    """Extract nested 'thinks' chain and depth ('Owen->Liam->Chloe', 3) or ('real', 0)."""
    tokens = question.strip().split()
    agents, used = [], set()
    for i, tok in enumerate(tokens):
        if tok.lower() in {"think", "thinks"}:
            for j in range(i - 1, -1, -1):
                if re.fullmatch(r"[A-Z][a-z]+", tokens[j]) and tokens[j] not in used:
                    agents.append(tokens[j]); used.add(tokens[j]); break
    if not agents:
        return ("real", 0)
    return ("->".join(agents), len(agents))

def depth_key(depth: int) -> str:
    return {0: "real", 1: "first", 2: "second", 3: "third", 4: "fourth"}.get(depth, "real")

def gt_answer_at_step(trace_data: Dict[str, Any], depth: int, chain: str) -> str:
    """Pick the correct value from GT trace at a given depth."""
    key = depth_key(depth)
    data = trace_data.get(key, {})
    if depth == 0:
        return data if isinstance(data, str) else "Null"
    if not isinstance(data, dict):
        return "Null"
    return data.get(chain, "Null")

def build_conversation(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Turn one GPT trace into stepwise (user: step text -> assistant: +/-) conversation."""
    traces = load_gt_traces(entry["path"])
    chain, depth = parse_question(entry.get("question", ""))

    true_answers = [gt_answer_at_step(t, depth, chain) for t in traces]

    gpt_trace = entry.get("gpt_trace", "")
    steps = re.findall(r"## Step (\d+) ##(.*?)(?=## Step |\Z|Final Answer:)", gpt_trace, re.DOTALL)
    final_answer_match = re.search(r"Final Answer:\s*\[(.*?)\]", gpt_trace)

    conversation, all_prefix_correct, correct_count = [], True, 0

    # Prepend the full prompt before the first step (keeps your original behavior)
    full_prompt_prefix = construct_prompt(
        entry.get("story", "").strip(),
        entry.get("question", "").strip(),
        entry.get("answer", "").strip(),
        note=""
    )

    for i, (step_num, step_body) in enumerate(steps):
        s_num = int(step_num)
        body = step_body.strip()
        user_text = (f"{full_prompt_prefix}\n\n## Step {s_num} ##\n{body}") if i == 0 else f"## Step {s_num} ##\n{body}"
        user_msg = {"role": "user", "content": user_text}

        brackets = re.findall(r"\[(.*?)\]", body)
        step_pred = brackets[-1].strip() if brackets else "Null"

        t_idx = s_num - 1
        true_ans = true_answers[t_idx] if 0 <= t_idx < len(true_answers) else "Unavailable"

        is_correct = all_prefix_correct and (step_pred == true_ans)
        if not is_correct:
            all_prefix_correct = False

        assistant_msg = {
            "role": "assistant",
            "content": "+" if is_correct else "-",
            "metadata": {
                "t": traces[t_idx]["t"] if 0 <= t_idx < len(traces) else None,
                "expected": true_ans,
                "actual": step_pred,
                "correct": is_correct
            }
        }
        if is_correct:
            correct_count += 1

        conversation.extend([user_msg, assistant_msg])

    # Final answer consistency check on last step
    if steps:
        last_step_num, last_body = steps[-1]
        last_num = int(last_step_num)
        last_brackets = re.findall(r"\[(.*?)\]", last_body)
        step_final_pred = last_brackets[-1].strip() if last_brackets else "ParseError"
        final_answer = final_answer_match.group(1).strip() if final_answer_match else "ParseError"

        user_msg = {
            "role": "user",
            "content": f"## Step {last_num} ##\n{last_body.strip()}\n\nFinal Answer: [{final_answer}]"
        }

        t_idx = last_num - 1
        true_ans = true_answers[t_idx] if 0 <= t_idx < len(true_answers) else "Unavailable"

        is_correct = (step_final_pred == final_answer) and all_prefix_correct and (final_answer == true_ans)
        if not is_correct:
            all_prefix_correct = False

        assistant_msg = {
            "role": "assistant",
            "content": "+" if is_correct else "-",
            "metadata": {
                "t": traces[t_idx]["t"] if 0 <= t_idx < len(traces) else None,
                "expected": true_ans,
                "actual": final_answer,
                "correct": is_correct
            }
        }
        if is_correct:
            correct_count += 1

        conversation.extend([user_msg, assistant_msg])

        if len(traces) != len(steps):
            print(f"⚠️  Length mismatch for {entry['path']}: GT {len(traces)} vs GPT {len(steps)}")

    return {
        "path": entry["path"],
        "conversation": conversation,
        "metadata": {
            "true_answers": true_answers,
            "total_steps": len(steps),
            "correct_steps": correct_count - 1
        }
    }

async def build_conversations(trace_in: Path, out_path: Path) -> None:
    """Read GPT traces and write aligned conversations jsonl."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total, fully_correct = 0, 0
    from collections import Counter
    dist = Counter()

    async with aiofiles.open(trace_in, "r", encoding="utf-8") as fin, \
               aiofiles.open(out_path, "w", encoding="utf-8") as fout:
        async for line in fin:
            entry = json.loads(line)
            if "gpt_trace" not in entry:  # skip failed requests
                continue
            result = build_conversation(entry)
            await fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            total += 1
            if result["metadata"]["correct_steps"] == result["metadata"]["total_steps"]:
                fully_correct += 1
            dist[result["metadata"]["correct_steps"]] += 1

    acc = (fully_correct / total) if total else 0.0
    print(f"[Conv] Total: {total}, Fully-correct: {fully_correct}, Acc(all-correct): {acc:.4f}")
    for k in sorted(dist.keys()):
        print(f"[Conv] {k} steps correct: {dist[k]}")

# ---------- Export PRM ----------

async def export_prm(conversations_in: Path, prm_out: Path, token_model: str = "gpt-4") -> None:
    """Write {"conversations": [...]} lines and shuffle."""
    prm_out.parent.mkdir(parents=True, exist_ok=True)
    encoding = None
    if tiktoken is not None:
        try:
            encoding = tiktoken.encoding_for_model(token_model)
        except Exception:
            pass

    buf, max_tokens = [], 0
    async with aiofiles.open(conversations_in, "r", encoding="utf-8") as fin:
        async for line in fin:
            obj = json.loads(line)
            dialogue = [{"role": m["role"], "content": m["content"]} for m in obj.get("conversation", [])]
            buf.append(json.dumps({"conversations": dialogue}, ensure_ascii=False))
            if encoding is not None:
                txt = " ".join(m["content"] for m in dialogue)
                max_tokens = max(max_tokens, len(encoding.encode(txt)))

    random.shuffle(buf)
    async with aiofiles.open(prm_out, "w", encoding="utf-8") as fout:
        await fout.write("\n".join(buf) + ("\n" if buf else ""))

    if encoding is not None:
        print(f"[PRM] Max token length ~= {max_tokens}")
    print(f"[PRM] Wrote {len(buf)} lines -> {prm_out}")

# ---------- Main: single-shot pipeline ----------

def main():
    parser = argparse.ArgumentParser(description="One-shot PBM/PRM dataset generator")
    parser.add_argument("--base-dir", type=Path, required=True, help="Hi-ToM root (e.g., Hi-ToM_dataset/data_ToMh)")
    parser.add_argument("--trace-out", type=Path, required=True, help="Output jsonl for GPT traces")
    parser.add_argument("--conversations-out", type=Path, required=True, help="Output jsonl for aligned conversations")
    parser.add_argument("--prm-out", type=Path, required=True, help="Output jsonl for PRM training")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI model name")
    parser.add_argument("--concurrency", type=int, default=200, help="Async request concurrency")
    parser.add_argument("--only-length", type=str, default="length_1",
                        choices=["length_1", "length_2", "length_3", "all"],
                        help="Choose a length bucket or 'all'")
    parser.add_argument("--token-model", type=str, default="gpt-4", help="For optional token counting")
    args = parser.parse_args()

    only_len = None if args.only_length == "all" else args.only_length

    # 1) Generate GPT traces
    asyncio.run(generate_traces(
        base_dir=args.base_dir,
        output_file=args.trace_out,
        model=args.model,
        concurrency=args.concurrency,
        only_length=only_len
    ))

    # 2) Build conversations from traces
    asyncio.run(build_conversations(
        trace_in=args.trace_out,
        out_path=args.conversations_out
    ))

    # 3) Export PRM {"conversations": [...]} lines and shuffle
    asyncio.run(export_prm(
        conversations_in=args.conversations_out,
        prm_out=args.prm_out,
        token_model=args.token_model
    ))

if __name__ == "__main__":
    main()
