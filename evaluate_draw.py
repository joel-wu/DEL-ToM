"""
Best‑of‑N Accuracy:  vanilla  vs  weighted
 - Five lines are always drawn: Majority, avg, last, min, prod
 - vanilla   : pick single trace with highest score (for each rule)
 - weighted  : sum scores per answer, pick answer with max total (for each rule)
"""

import json, re, random, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt



# ---------- small helpers ---------- #
def extract_final_answer(trace: str) -> str:
    m = re.search(r"Final Answer:\s*\[(.*?)\]", trace)
    return m.group(1).strip() if m else "Null"


def prod(seq):
    out = 1.0
    for v in seq:
        out *= v
    return out


def load_score_dict(score_file: Path):
    """return dict[(pidx,tidx)] -> list[(step,score)]"""
    tbl = defaultdict(list)
    with open(score_file) as f:
        for ln in f:
            d = json.loads(ln)
            tbl[(d["problem_idx"], d["trace_idx"])].append((d["step_idx"], d["score"]))
    return tbl


def precompute_trace_scores(candidates, score_dict):
    """trace_scores[(pidx,tidx)] = {'avg':..,'last':..,'min':..,'prod':..}"""
    tbl = {}
    for pidx, item in enumerate(candidates):
        for tidx in range(len(item["answers"])):
            vals = sorted(score_dict.get((pidx, tidx)), key=lambda p: p[0])
            if not vals:
                continue

            mid = len(vals) // 2
            seg = vals

            scrs = [s for _, s in seg]
            if not scrs: 
                continue
            tbl[(pidx, tidx)] = {
                "avg":  sum(scrs) / len(scrs),
                "last": scrs[-1],
                "min":  min(scrs),
                "prod": prod(scrs),
            }
    return tbl
# ----------------------------------- #


def compute_acc_random_trials(
    cand_path: Path,
    score_path: Path,
    mode: str = "vanilla",
    n_values=None,
    trials: int = 10,
):
    assert mode in {"vanilla", "weighted"}
    if n_values is None:
        n_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    cand_data   = [json.loads(ln) for ln in open(cand_path)]
    score_dict  = load_score_dict(score_path)
    trace_sc    = precompute_trace_scores(cand_data, score_dict)

    rules = ["majority", "avg", "last", "min", "prod"]
    mean, band = {r: [] for r in rules}, {r: [] for r in rules}

    for N in n_values:
        trial_hits = {r: [] for r in rules}

        for _ in range(trials):
            hits = dict.fromkeys(rules, 0)

            for pidx, item in enumerate(cand_data):
                gt_ans = item["answer"]
                tot_tr = len(item["answers"])
                if tot_tr < N:
                    continue
                sampled = random.sample(range(tot_tr), N)

                # ---- Majority ----
                finals = [extract_final_answer(item["answers"][t]) for t in sampled]
                if max(set(finals), key=finals.count) == gt_ans:
                    hits["majority"] += 1

                # collect scores
                scored = [(t, trace_sc.get((pidx, t))) for t in sampled
                          if (pidx, t) in trace_sc]

                if not scored:
                    continue

                # vanilla or weighted decision for each rule
                for rule in ["avg", "last", "min", "prod"]:
                    if mode == "vanilla":
                        # pick single trace
                        best_t = max(scored, key=lambda x: x[1][rule])[0]
                        pred   = extract_final_answer(item["answers"][best_t])
                    else:  # weighted
                        agg = defaultdict(float)
                        for t, sc in scored:
                            ans = extract_final_answer(item["answers"][t])
                            agg[ans] += sc[rule]
                        pred = max(agg.items(), key=lambda kv: kv[1])[0]

                    if pred == gt_ans:
                        hits[rule] += 1

            total = len(cand_data)
            for r in rules:
                trial_hits[r].append(hits[r] / total)

        # aggregate mean / range
        for r in rules:
            vals = trial_hits[r]
            mean[r].append(float(np.mean(vals)))
            band[r].append((float(np.min(vals)), float(np.max(vals))))

    return {"n": n_values, "mean": mean, "band": band, "mode": mode}


def plot_res(res, out_file="acc.pdf", title="Best‑of‑N Accuracy"):
    n_vals, means, bands = res["n"], res["mean"], res["band"]
    style = {"majority": ("o", "Majority"),
             "avg":      ("s", "avg"),
             "last":     ("^", "last"),
             "min":      ("x", "min"),
             "prod":     ("*", "prod")}

    all_mean = sum(means.values(), [])
    y_min = max(0, min(all_mean) - .05)
    y_max = min(1, max(all_mean) + .05)

    plt.figure(figsize=(7,4))
    plt.rcParams.update({
        "font.size": 16,  # default font size
        "axes.titlesize": 18,  # title size
        "axes.labelsize": 16,  # axis label size
        "xtick.labelsize": 14,  # x tick size
        "ytick.labelsize": 14,  # y tick size
        "legend.fontsize": 12  # legend size
    })

    for rule in style:
        m, lab = style[rule]
        vals   = means[rule]
        low    = [b[0] for b in bands[rule]]
        high   = [b[1] for b in bands[rule]]
        plt.plot(n_vals, vals, marker=m, label=f"{lab}")
        plt.fill_between(n_vals, low, high, alpha=.2)

    plt.xscale("log", base=2)
    plt.xticks(n_vals)
    plt.xlabel("N  (# traces)")
    plt.ylabel("Accuracy")
    plt.ylim(y_min, y_max)
    plt.grid(ls="--", alpha=.6)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate and plot Best-of-N accuracy results")
    parser.add_argument("--candidates_file", type=str, required=True,
                       help="Path to candidates JSONL file (from bon_generation.py)")
    parser.add_argument("--scores_file", type=str, required=True,
                       help="Path to PRM scores JSONL file (from PBM_evaluation.py)")
    parser.add_argument("--output_dir", type=str, default="./plots",
                       help="Output directory for plots (default: ./plots)")
    parser.add_argument("--model_name", type=str, default="Model",
                       help="Model name for plot titles (default: Model)")
    parser.add_argument("--mode", type=str, choices=["vanilla", "weighted", "both"], default="both",
                       help="Evaluation mode: vanilla, weighted, or both (default: both)")
    parser.add_argument("--trials", type=int, default=10,
                       help="Number of random trials for evaluation (default: 10)")
    parser.add_argument("--n_values", type=int, nargs="+", default=None,
                       help="N values to test (default: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])")
    parser.add_argument("--output_format", type=str, choices=["pdf", "png", "svg"], default="pdf",
                       help="Output format for plots (default: pdf)")
    
    args = parser.parse_args()
    
    # Set default N values if not provided
    if args.n_values is None:
        args.n_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert paths to Path objects
    candidates_path = Path(args.candidates_file)
    scores_path = Path(args.scores_file)
    
    print(f"Evaluation Configuration:")
    print(f"  Candidates file: {candidates_path}")
    print(f"  Scores file: {scores_path}")
    print(f"  Model name: {args.model_name}")
    print(f"  Mode: {args.mode}")
    print(f"  Trials: {args.trials}")
    print(f"  N values: {args.n_values}")
    print(f"  Output directory: {output_dir}")
    print(f"  Output format: {args.output_format}")
    print()
    
    # Determine modes to run
    modes_to_run = []
    if args.mode == "both":
        modes_to_run = ["vanilla", "weighted"]
    else:
        modes_to_run = [args.mode]
    
    # Run evaluation for each mode
    for mode in modes_to_run:
        print(f"Running {mode} evaluation...")
        
        res = compute_acc_random_trials(
            cand_path=candidates_path,
            score_path=scores_path,
            mode=mode,
            n_values=args.n_values,
            trials=args.trials,
        )
        
        # Generate output filename
        output_file = output_dir / f"{mode}_{args.model_name}_boN.{args.output_format}"
        
        # Create plot
        plot_res(
            res,
            out_file=str(output_file),
            title=f"{mode.capitalize()} Best-of-N, {args.model_name}"
        )
        
        print(f"Saved: {output_file}")
        print()


if __name__ == "__main__":
    main()