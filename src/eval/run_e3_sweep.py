"""
Experiment 3 (E3): Consortium Scaling (N in {2, 3, 4}) and Predictors of Failure (E8).
In-Memory GPU Swapper, Crash-Resumable Checkpointing, and Diagnostic Collector.

Features:
1. In-place GPU state_dict loading (0 disk usage, 0.1s swap latency).
2. Atomic saving to outputs/results/e3_sweep_results.json (or Google Drive) with auto-resume.
3. Pre-caches B_0 reference log-probabilities on the 200 UltraFeedback held-out pairs.
4. Exact prompt formatting and verified answer evaluators imported directly from:
   - src/eval/eval_pref.py (compute_sequence_logp)
   - src/eval/eval_math.py (extract_predicted_answer, is_equivalent)
   - src/eval/eval_code.py (extract_code, run_with_timeout)
   - src/eval/eval_ifeval.py (check_constraint)
5. Simplex weight parameterization:
   - w_pref = alpha_pref
   - w_domain_k = (1.0 - alpha_pref) / (N - 1)
   - Uniform points: alpha=1/N (0.3333 for N=3, 0.2500 for N=4)
6. Computes all 5 failure diagnostics in CPU RAM:
   - Predictor 1: Effective Preference Magnitude Share (eta_pref)
   - Predictor 2: Preference Sign-Overrule Rate (Clash_overruled on top-20% coords)
   - Predictor 3: Directional Alignment Survival (CosSim_surv)
   - Predictor 4: Preference Projection Recovery (pi_pref)
   - Predictor 5: Importance-Weighted Sign Overrule Rate (Clash_weighted)
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import gc
import json
import math
import shutil
import argparse
import numpy as np
import torch
torch.set_num_threads(4)
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from configs.base_config import MODEL_ID, TORCH_DTYPE, ATTN_IMPLEMENTATION
from src.merge.merge_methods import ties_merge_tensors, clean_state_dict_keys, load_state_dict_cpu
from src.eval.eval_pref import compute_sequence_logp
from src.eval.eval_math import extract_predicted_answer, is_equivalent
from src.eval.eval_code import extract_code, run_with_timeout
from src.eval.eval_ifeval import check_constraint

# ==========================================
# Benchmark Calibration Anchors (Table 1)
# ==========================================
B0_MATH_ACC = 53.3333
DELTA_MATH = 5.3334          # SVAMP: E_math (58.67%) - B0 (53.33%)

B0_CODE_ACC = 31.1284
DELTA_CODE = 7.7821          # MBPP: E_code (38.91%) - B0 (31.13%)

B0_INSTR_ACC = 29.6667
DELTA_INSTR = 8.3333        # IFEval Strict: E_instr (38.00%) - B0 (29.67%)

DELTA_PREF_MARGIN = 0.3148926  # UltraFeedback 200: E_pref margin (+0.3149)

RANDOM_SEED = 42

# ==========================================
# In-Memory Merge Engine for Arbitrary N
# ==========================================
def merge_n_experts(
    method: str,
    weights: list[float],
    base_sd: dict,
    task_vectors: list[dict],
    density: float = 0.20,
    drop_rate: float = 0.50
) -> dict:
    """
    Merges N task vectors onto base_sd with simplex weights sum(weights) == 1.0.
    task_vectors: [tau_pref, tau_domain1, tau_domain2, ...]
    weights: [w_pref, w_domain1, w_domain2, ...]
    """
    merged_sd = {}
    N = len(task_vectors)

    for k in base_sd.keys():
        b_val = base_sd[k]

        # Non-floating point or embedding / head weights stay anchored to base
        if not b_val.is_floating_point() or "embed" in k or "lm_head" in k:
            merged_sd[k] = b_val.clone()
            continue

        tau_list = [tv[k] for tv in task_vectors]

        if method == "linear":
            tau_merged = torch.zeros_like(b_val)
            for w, tau in zip(weights, tau_list):
                tau_merged += w * tau
            merged_sd[k] = b_val + tau_merged

        elif method == "ties":
            # Scale each task vector by its simplex weight before sign election
            weighted_taus = [w * tau.float() for w, tau in zip(weights, tau_list)]
            tau_ties, _ = ties_merge_tensors(
                weighted_taus,
                density=density,
                alpha=1.0,
                divide_by_count=False  # Crucial: sum agreeing pre-weighted coordinates
            )
            merged_sd[k] = b_val + tau_ties.to(b_val.dtype)

        elif method == "dare_ties":
            # Deterministic DARE per expert tensor to prevent stochastic mask jitter
            keep_prob = 1.0 - drop_rate
            dare_taus = []
            for j, (w, tau) in enumerate(zip(weights, tau_list)):
                gen = torch.Generator(device="cpu").manual_seed(RANDOM_SEED + j)
                mask = torch.bernoulli(torch.full_like(tau.float(), keep_prob), generator=gen).to(torch.bool)
                tau_rescaled = (tau.float() * mask) / keep_prob
                dare_taus.append(w * tau_rescaled)

            tau_dare_ties, _ = ties_merge_tensors(
                dare_taus,
                density=density,
                alpha=1.0,
                divide_by_count=False
            )
            merged_sd[k] = b_val + tau_dare_ties.to(b_val.dtype)

        else:
            raise ValueError(f"Unknown merge method: {method}")

    return merged_sd


# ==========================================
# Five Mechanistic Predictors of Failure (E8)
# ==========================================
def compute_failure_diagnostics(
    base_sd: dict,
    task_vectors: list[dict],
    merged_sd: dict,
    weights: list[float],
    pref_idx: int = 0,
    top_quantile: float = 0.20
) -> dict:
    """
    Computes all 5 theoretical failure predictors in CPU RAM:
    1. eta_pref: Effective Preference Magnitude Share
    2. clash_overruled: Preference Sign-Overrule Rate on top-20% coordinates
    3. csim_surv: Directional Alignment Survival cos(tau_merged, tau_pref)
    4. pi_pref: Preference Projection Recovery <tau_merged, tau_pref> / ||tau_pref||^2
    5. clash_weighted: Importance-Weighted Sign Overrule Rate (Fisher proxy)
    """
    tau_pref_dict = task_vectors[pref_idx]

    # Pre-accumulate scalar dot products and squared norms
    dot_merged_pref = 0.0
    norm_sq_merged = 0.0
    norm_sq_pref = 0.0
    norm_sq_per_expert = [0.0] * len(task_vectors)

    total_top_coords = 0
    total_clash_coords = 0
    layer_clash_data = []

    for k in base_sd.keys():
        b_val = base_sd[k]
        if not b_val.is_floating_point() or "embed" in k or "lm_head" in k:
            continue

        p_val = tau_pref_dict[k].float()
        m_val = (merged_sd[k] - b_val).float()

        # Dot product and norms
        dot_k = torch.sum(m_val * p_val).item()
        m_sq = torch.sum(m_val * m_val).item()
        p_sq = torch.sum(p_val * p_val).item()

        dot_merged_pref += dot_k
        norm_sq_merged += m_sq
        norm_sq_pref += p_sq

        for j, tv in enumerate(task_vectors):
            norm_sq_per_expert[j] += torch.sum(tv[k].float() ** 2).item()

        # Top-20% sign overrule calculation per tensor
        p_flat = p_val.flatten()
        m_flat = m_val.flatten()
        k_top = int(top_quantile * p_flat.numel())

        if k_top > 0 and p_sq > 1e-12:
            topk_vals, _ = torch.topk(torch.abs(p_flat), k_top)
            thresh = topk_vals[-1]
            top_mask = (torch.abs(p_flat) >= thresh) & (p_flat != 0)

            sign_p = torch.sign(p_flat[top_mask])
            sign_m = torch.sign(m_flat[top_mask])
            clashed = (sign_p != sign_m)

            n_top = int(top_mask.sum().item())
            n_clash = int(clashed.sum().item())

            total_top_coords += n_top
            total_clash_coords += n_clash

            layer_clash_data.append({
                "layer_p_norm_sq": p_sq,
                "layer_top_coords": n_top,
                "layer_clash_coords": n_clash
            })

    # 1. Effective Preference Magnitude Share (eta_pref)
    expert_norms = [math.sqrt(max(0.0, s)) for s in norm_sq_per_expert]
    weighted_norms = [w * n for w, n in zip(weights, expert_norms)]
    sum_weighted_norms = sum(weighted_norms)
    eta_pref = (weighted_norms[pref_idx] / max(1e-12, sum_weighted_norms)) if sum_weighted_norms > 0 else 0.0

    # 2. Preference Sign-Overrule Rate (Clash_overruled)
    clash_overruled = (total_clash_coords / max(1, total_top_coords)) if total_top_coords > 0 else 0.0

    # 3. Directional Alignment Survival (CosSim_surv)
    norm_m = math.sqrt(max(0.0, norm_sq_merged))
    norm_p = math.sqrt(max(0.0, norm_sq_pref))
    csim_surv = (dot_merged_pref / max(1e-12, (norm_m * norm_p))) if (norm_m > 0 and norm_p > 0) else 0.0

    # 4. Preference Projection Recovery (pi_pref)
    pi_pref = (dot_merged_pref / max(1e-12, norm_sq_pref)) if norm_sq_pref > 0 else 0.0

    # 5. Importance-Weighted Sign Overrule Rate (Clash_weighted)
    clash_weighted = 0.0
    if norm_sq_pref > 0:
        for ld in layer_clash_data:
            weight = ld["layer_p_norm_sq"] / norm_sq_pref
            layer_rate = (ld["layer_clash_coords"] / max(1, ld["layer_top_coords"]))
            clash_weighted += weight * layer_rate

    return {
        "eta_pref": float(eta_pref),
        "clash_overruled": float(clash_overruled),
        "csim_surv": float(csim_surv),
        "pi_pref": float(pi_pref),
        "clash_weighted": float(clash_weighted)
    }


# ==========================================
# In-Memory Evaluation Routines
# ==========================================
def eval_preference_in_memory(gpu_model, tokenizer, pairs, ref_cache, beta=0.1) -> tuple[float, float, float]:
    """Computes raw win rate, mean DPO implicit reward margin, and normalized PRR."""
    raw_wins = 0
    margins = []
    for i, it in enumerate(pairs):
        tot_c, norm_c, _ = compute_sequence_logp(gpu_model, tokenizer, it["prompt"], it["chosen"])
        tot_r, norm_r, _ = compute_sequence_logp(gpu_model, tokenizer, it["prompt"], it["rejected"])
        if norm_c > norm_r:
            raw_wins += 1
        ref_c = ref_cache[i]["chosen"]
        ref_r = ref_cache[i]["rejected"]
        margin = beta * ((tot_c - tot_r) - (ref_c - ref_r))
        margins.append(margin)

    raw_rate = (raw_wins / len(pairs)) * 100.0
    mean_margin = float(sum(margins) / len(margins))
    prr = (mean_margin / DELTA_PREF_MARGIN) * 100.0
    return raw_rate, mean_margin, prr


def eval_svamp_in_memory(gpu_model, tokenizer, svamp_items, batch_size=16) -> tuple[float, float]:
    """Evaluates mathematical reasoning accuracy on SVAMP 300 problems."""
    total = len(svamp_items)
    correct = 0
    for b_idx in range(0, total, batch_size):
        batch = svamp_items[b_idx:b_idx + batch_size]
        prompts = []
        for it in batch:
            msgs = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": it["question"]}
            ]
            prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(gpu_model.device)
        in_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            outputs = gpu_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        gens = tokenizer.batch_decode(outputs[:, in_len:], skip_special_tokens=True)
        for it, gen in zip(batch, gens):
            pred = extract_predicted_answer(gen)
            if is_equivalent(pred, it["gold"]):
                correct += 1

    svamp_acc = (correct / total) * 100.0
    cr_math = ((svamp_acc - B0_MATH_ACC) / DELTA_MATH) * 100.0
    return svamp_acc, cr_math


def eval_mbpp_in_memory(gpu_model, tokenizer, mbpp_items, batch_size=16) -> tuple[float, float]:
    """Evaluates Python code generation Pass@1 on sanitized MBPP test split (257 problems)."""
    total = len(mbpp_items)
    passed = 0
    for b_idx in range(0, total, batch_size):
        batch = mbpp_items[b_idx:b_idx + batch_size]
        prompts = []
        for it in batch:
            prompt_text = it["prompt"]
            test_list = it["test_list"]
            msgs = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": f"Write a Python function to solve the following problem:\n{prompt_text}\nYour code should satisfy these tests:\n" + "\n".join(test_list[:1])}
            ]
            prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(gpu_model.device)
        in_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            outputs = gpu_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        gens = tokenizer.batch_decode(outputs[:, in_len:], skip_special_tokens=True)
        for it, gen in zip(batch, gens):
            code = extract_code(gen)
            test_assertions = it["test_list"]
            if run_with_timeout(code, test_assertions, timeout=3.0):
                passed += 1

    mbpp_acc = (passed / total) * 100.0
    cr_code = ((mbpp_acc - B0_CODE_ACC) / DELTA_CODE) * 100.0
    return mbpp_acc, cr_code


def eval_ifeval_in_memory(gpu_model, tokenizer, ifeval_items, batch_size=16) -> tuple[float, float]:
    """Evaluates strict instruction following on IFEval 300 prompts."""
    total = len(ifeval_items)
    passed_prompts = 0
    for b_idx in range(0, total, batch_size):
        batch = ifeval_items[b_idx:b_idx + batch_size]
        prompts = []
        for it in batch:
            msgs = [{"role": "user", "content": it["prompt"]}]
            prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(gpu_model.device)
        in_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            outputs = gpu_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        gens = tokenizer.batch_decode(outputs[:, in_len:], skip_special_tokens=True)

        for it, gen_text in zip(batch, gens):
            instruction_ids = it.get("instruction_id_list", [])
            kwargs_list = it.get("kwargs", [])
            prompt_all_passed = True
            for inst_id, kw in zip(instruction_ids, kwargs_list):
                kw_dict = kw if isinstance(kw, dict) else {}
                if not check_constraint(inst_id, kw_dict, gen_text, prompt=it["prompt"]):
                    prompt_all_passed = False
                    break
            if prompt_all_passed and len(instruction_ids) > 0:
                passed_prompts += 1

    ifeval_acc = (passed_prompts / total) * 100.0
    cr_instr = ((ifeval_acc - B0_INSTR_ACC) / DELTA_INSTR) * 100.0
    return ifeval_acc, cr_instr


# ==========================================
# Atomic Save Utility
# ==========================================
def save_atomic_json(filepath: str, data: dict):
    """Writes results atomically via a temporary file to prevent corruption."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    temp_path = filepath + ".tmp"
    with open(temp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(temp_path, filepath)


# ==========================================
# Main Experiment 3 Sweep Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Experiment 3 (E3) Consortium Scaling Sweep")
    parser.add_argument("--results_file", type=str, default="outputs/results/e3_sweep_results.json",
                        help="Path to output JSON file (supports Google Drive paths)")
    parser.add_argument("--batch_size", type=int, default=16, help="Evaluation batch size")
    args = parser.parse_args()

    results_file = args.results_file
    print(f"\n==================================================================")
    print(f"🚀 Launching Experiment 3 (E3) Full Consortium Scaling Sweep")
    print(f"   Target Results File: {results_file}")
    print(f"==================================================================\n")

    # Load existing progress if available
    results_dict = {}
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                results_dict = json.load(f)
            print(f"✅ Found existing results file with {len(results_dict)} checkpoints. Resuming seamlessly...\n")
        except Exception as e:
            print(f"⚠️ Warning: Could not read existing results: {e}. Starting fresh.\n")

    # 1. Load Datasets
    print("📦 Pre-loading evaluation datasets...")
    # UltraFeedback (200 pairs)
    pref_path = "outputs/datasets/e_pref_200.jsonl"
    if not os.path.exists(pref_path):
        pref_path = "outputs/datasets/e_pref_test.jsonl"
    pref_pairs = []
    with open(pref_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= 200:
                break
            if line.strip():
                pref_pairs.append(json.loads(line))
    print(f"   • UltraFeedback pairs: {len(pref_pairs)}")

    # SVAMP (300 problems)
    try:
        svamp_ds = load_dataset("ChilleD/SVAMP", split="test")
    except Exception:
        svamp_ds = load_dataset("Chillee/SVAMP", split="test")
    svamp_items = []
    for item in svamp_ds:
        body = item.get("Body", "").strip()
        q = item.get("Question", "").strip()
        full_q = f"{body} {q}".strip() if body else q
        gold = str(item.get("Answer", "")).strip()
        if gold.endswith(".0"):
            gold = gold[:-2]
        svamp_items.append({"question": full_q, "gold": gold})
    print(f"   • SVAMP problems: {len(svamp_items)}")

    # MBPP (257 sanitized problems)
    try:
        mbpp_ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    except Exception:
        mbpp_ds = load_dataset("mbpp", "sanitized", split="test")
    mbpp_items = list(mbpp_ds)
    print(f"   • MBPP problems: {len(mbpp_items)}")

    # IFEval (300 prompts)
    try:
        ifeval_ds = load_dataset("google/IFEval", split="train").select(range(300))
    except Exception:
        ifeval_ds = load_dataset("wiseman/ifeval", split="train").select(range(300))
    ifeval_items = list(ifeval_ds)
    print(f"   • IFEval prompts: {len(ifeval_items)}")

    # 2. Tokenizer & GPU Model Container
    print("\n⚡ Initializing Tokenizer and single GPU Model Container...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION
    ).to(device)
    gpu_model.eval()

    # 3. Reference Log-Probabilities Cache for B_0
    ref_cache_file = "outputs/results/b0_pref_200_cache.json"
    if os.path.exists(ref_cache_file):
        print(f"\n📂 Loading cached B0 reference log-probabilities from {ref_cache_file}...")
        with open(ref_cache_file, "r") as f:
            ref_cache = json.load(f)
    else:
        print("\n⚡ Pre-computing B0 reference log-probabilities for UltraFeedback...")
        ref_cache = []
        for it in tqdm(pref_pairs, desc="Pre-caching B0 logps"):
            tot_c, _, _ = compute_sequence_logp(gpu_model, tokenizer, it["prompt"], it["chosen"])
            tot_r, _, _ = compute_sequence_logp(gpu_model, tokenizer, it["prompt"], it["rejected"])
            ref_cache.append({"chosen": tot_c, "rejected": tot_r})
        os.makedirs("outputs/results", exist_ok=True)
        with open(ref_cache_file, "w") as f:
            json.dump(ref_cache, f, indent=2)

    # 4. Load State Dicts on CPU & Compute Task Vectors
    print("\n🧠 Loading Base Model and Specialist State Dicts into CPU RAM (bfloat16)...")
    base_sd = load_state_dict_cpu(MODEL_ID)

    def resolve_specialist_path(name: str) -> str:
        ckpt = f"outputs/checkpoints/{name}"
        if os.path.exists(ckpt):
            return ckpt
        adapt = f"outputs/adapters/{name}"
        if os.path.exists(adapt):
            return adapt
        return ckpt

    ckpt_paths = {
        "pref": resolve_specialist_path("e_pref"),
        "math": resolve_specialist_path("e_math"),
        "code": resolve_specialist_path("e_code"),
        "instr": resolve_specialist_path("e_instr")
    }
    print("\n🧠 Loading Specialist Models and Computing Task Vectors (one-by-one to save RAM)...")
    task_vectors = {}
    for name, path in ckpt_paths.items():
        sd = load_state_dict_cpu(path)
        print(f"   • Computing task vector tau_{name} = theta_{name} - theta_0...")
        task_vectors[name] = {k: (sd[k] - base_sd[k]).to(torch.bfloat16) for k in base_sd.keys()}
        del sd
        gc.collect()

    # 5. Define Sweep Configurations
    # N=3: [E_pref, E_math, E_code]
    # N=4: [E_pref, E_math, E_code, E_instr]
    methods = ["linear", "ties", "dare_ties"]

    grid_n3 = [0.1, 0.2, 0.3, 0.3333, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    grid_n4 = [0.1, 0.2, 0.2500, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    tasks_by_n = {
        3: ["pref", "math", "code"],
        4: ["pref", "math", "code", "instr"]
    }
    grids_by_n = {
        3: grid_n3,
        4: grid_n4
    }

    total_runs = (len(grid_n3) + len(grid_n4)) * len(methods)
    print(f"\n==================================================================")
    print(f"📊 Starting Execution Loop: {total_runs} Total Checkpoints")
    print(f"   • N=3: {len(grid_n3)} weights x {len(methods)} methods = {len(grid_n3) * len(methods)} runs")
    print(f"   • N=4: {len(grid_n4)} weights x {len(methods)} methods = {len(grid_n4) * len(methods)} runs")
    print(f"==================================================================\n")

    run_idx = 0
    for N in [3, 4]:
        active_expert_names = tasks_by_n[N]
        active_tvs = [task_vectors[name] for name in active_expert_names]
        alphas = grids_by_n[N]

        for method in methods:
            for alpha_pref in alphas:
                run_idx += 1
                key = f"N{N}_{method}_{alpha_pref:.4f}"

                if key in results_dict:
                    print(f"[{run_idx}/{total_runs}] Skipping {key} (Already Evaluated).")
                    continue

                w_domain = (1.0 - alpha_pref) / (N - 1)
                weights = [alpha_pref] + [w_domain] * (N - 1)

                print(f"[{run_idx}/{total_runs}] Merging {key}: w_pref={alpha_pref:.4f}, w_domain={w_domain:.4f}...")

                # Merge on CPU
                merged_sd = merge_n_experts(
                    method=method,
                    weights=weights,
                    base_sd=base_sd,
                    task_vectors=active_tvs,
                    density=0.20,
                    drop_rate=0.50
                )

                # Compute Failure Diagnostics (E8)
                diagnostics = compute_failure_diagnostics(
                    base_sd=base_sd,
                    task_vectors=active_tvs,
                    merged_sd=merged_sd,
                    weights=weights,
                    pref_idx=0,
                    top_quantile=0.20
                )

                # In-Place Weight Swap to GPU
                gpu_model.load_state_dict(merged_sd, strict=True)
                del merged_sd
                gc.collect()

                # Run Evaluations
                raw_win, dpo_margin, prr = eval_preference_in_memory(gpu_model, tokenizer, pref_pairs, ref_cache)
                svamp_acc, cr_math = eval_svamp_in_memory(gpu_model, tokenizer, svamp_items, batch_size=args.batch_size)
                mbpp_acc, cr_code = eval_mbpp_in_memory(gpu_model, tokenizer, mbpp_items, batch_size=args.batch_size)

                result_entry = {
                    "N": N,
                    "method": method,
                    "alpha_pref": alpha_pref,
                    "weights": weights,
                    "raw_win_rate": raw_win,
                    "dpo_margin": dpo_margin,
                    "prr": prr,
                    "svamp_acc": svamp_acc,
                    "cr_math": cr_math,
                    "mbpp_acc": mbpp_acc,
                    "cr_code": cr_code,
                    "diagnostics": diagnostics
                }

                if N == 4:
                    ifeval_acc, cr_instr = eval_ifeval_in_memory(gpu_model, tokenizer, ifeval_items, batch_size=args.batch_size)
                    mean_cr = (cr_math + cr_code + cr_instr) / 3.0
                    result_entry["ifeval_acc"] = ifeval_acc
                    result_entry["cr_instr"] = cr_instr
                    result_entry["mean_cr"] = mean_cr
                    print(f"   => PRR: {prr:.1f}% | CR_m: {cr_math:.1f}% | CR_c: {cr_code:.1f}% | CR_i: {cr_instr:.1f}% | Mean CR: {mean_cr:.1f}% | Clash: {diagnostics['clash_overruled']:.3f}")
                else:
                    mean_cr = (cr_math + cr_code) / 2.0
                    result_entry["mean_cr"] = mean_cr
                    print(f"   => PRR: {prr:.1f}% | CR_m: {cr_math:.1f}% | CR_c: {cr_code:.1f}% | Mean CR: {mean_cr:.1f}% | Clash: {diagnostics['clash_overruled']:.3f}")

                # Atomic save to disk
                results_dict[key] = result_entry
                save_atomic_json(results_file, results_dict)

                # GPU memory hygiene
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"\n==================================================================")
    print(f"🎉 Experiment 3 (E3) Sweep Completed Successfully!")
    print(f"   Results safely preserved at: {results_file}")
    print(f"==================================================================\n")


if __name__ == "__main__":
    main()
