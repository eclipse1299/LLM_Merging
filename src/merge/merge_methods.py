"""
Comprehensive Model Merging Engine for Experiment 2 (E2).
Implements:
1. Linear Task Arithmetic (Naive Unscaled Baseline)
2. Linear Task Arithmetic (Norm-Balanced Variant)
3. SLERP (Spherical Linear Interpolation)
4. TIES-Merging (Trim, Elect Signs, Disjoint Merge with coordinate logging)
5. DARE-TIES (Random Drop + Rescale + TIES)
"""
import os
import sys
import copy
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from configs.base_config import MODEL_ID, TORCH_DTYPE

def clean_state_dict_keys(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strips PEFT and model wrapper prefixes so keys align coordinate-by-coordinate."""
    cleaned = {}
    for k, v in sd.items():
        clean_k = k
        for prefix in ["base_model.model.", "base_model."]:
            if clean_k.startswith(prefix):
                clean_k = clean_k[len(prefix):]
        cleaned[clean_k] = v
    return cleaned

def load_state_dict_cpu(model_path: str) -> dict[str, torch.Tensor]:
    """Loads model state dict into CPU memory in bfloat16, normalizing key prefixes."""
    print(f"Loading state dict on CPU: {model_path}...")
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        print(f"Detected PEFT adapter at {model_path}, baking on CPU...")
        try:
            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            peft_m = PeftModel.from_pretrained(base, model_path)
            model = peft_m.merge_and_unload()
            sd = {k: v.to(torch.bfloat16).clone() for k, v in model.state_dict().items()}
            del peft_m, base, model
        except Exception as e:
            print(f"PEFT loading encountered {e}, falling back to pure-PyTorch LoRA baking...")
            from safetensors.torch import load_file
            base = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            sd = {k: v.to(torch.bfloat16).clone() for k, v in base.state_dict().items()}
            del base

            with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
                cfg = json.load(f)
            scale = cfg.get("lora_alpha", 32) / cfg.get("r", 16)
            safetensors_path = os.path.join(model_path, "adapter_model.safetensors")
            adapter = load_file(safetensors_path)
            for k in adapter:
                if "lora_A" in k:
                    b_key = k.replace("lora_A", "lora_B")
                    base_key = k.replace("base_model.model.", "").replace(".lora_A", "")
                    A = adapter[k].float()
                    B = adapter[b_key].float()
                    delta = scale * (B @ A)
                    clean_k = base_key
                    for prefix in ["base_model.model.", "base_model."]:
                        if clean_k.startswith(prefix):
                            clean_k = clean_k[len(prefix):]
                    if clean_k in sd:
                        sd[clean_k] = (sd[clean_k].float() + delta).to(torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        sd = {k: v.to(torch.bfloat16).clone() for k, v in model.state_dict().items()}
        del model
    import gc
    gc.collect()
    return clean_state_dict_keys(sd)

def slerp_tensor(v0: torch.Tensor, v1: torch.Tensor, t: float, eps: float = 1e-7) -> torch.Tensor:
    """Spherical Linear Interpolation between two tensors."""
    orig_shape = v0.shape
    v0_flat = v0.flatten().float()
    v1_flat = v1.flatten().float()

    n0 = torch.norm(v0_flat)
    n1 = torch.norm(v1_flat)

    if n0 < eps or n1 < eps:
        return ((1.0 - t) * v0_flat + t * v1_flat).view(orig_shape).to(v0.dtype)

    u0 = v0_flat / n0
    u1 = v1_flat / n1

    dot = torch.sum(u0 * u1)
    dot = torch.clamp(dot, -1.0, 1.0)

    # If collinear, standard linear interpolation
    if torch.abs(dot) > (1.0 - 1e-4):
        res = (1.0 - t) * v0_flat + t * v1_flat
        return res.view(orig_shape).to(v0.dtype)

    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    c0 = torch.sin((1.0 - t) * omega) / sin_omega
    c1 = torch.sin(t * omega) / sin_omega

    u_interp = c0 * u0 + c1 * u1
    n_interp = (1.0 - t) * n0 + t * n1

    res = n_interp * (u_interp / torch.norm(u_interp))
    return res.view(orig_shape).to(v0.dtype)

def ties_merge_tensors(
    tau_list: list[torch.Tensor],
    density: float = 0.20,
    alpha: float = 0.50,
    divide_by_count: bool = True
) -> tuple[torch.Tensor, dict]:
    """
    TIES-Merging on arbitrary K task vectors [tau_1, tau_2, ..., tau_K].
    Step 1: Trim top-density parameters
    Step 2: Elect sign consensus across active updates
    Step 3: Disjoint merge with consensus sign masking
    divide_by_count: If True, averages agreeing vectors (standard TIES).
                     If False, sums agreeing vectors (used when inputs are already pre-weighted with sum(w)=1).
    """
    stats = {
        "total_coords": tau_list[0].numel(),
        "multiple_active": 0,
        "sign_agreed": 0,
        "sign_clashed": 0
    }

    trimmed = []
    masks = []
    for tau in tau_list:
        flat = tau.flatten().float()
        k = int(density * flat.numel())
        if k > 0:
            topk_vals, _ = torch.topk(torch.abs(flat), k)
            thresh = topk_vals[-1]
            mask = torch.abs(flat) >= thresh
        else:
            mask = torch.zeros_like(flat, dtype=torch.bool)
        trimmed.append(flat * mask)
        masks.append(mask)

    stacked_trimmed = torch.stack(trimmed, dim=0)  # [K, N]
    stacked_masks = torch.stack(masks, dim=0)      # [K, N]

    # Sign consensus: sum of trimmed updates across all K models
    sum_tau = stacked_trimmed.sum(dim=0)
    consensus_sign = torch.sign(sum_tau)

    # Detailed clash logging
    active_count = stacked_masks.sum(dim=0)
    stats["multiple_active"] = int((active_count >= 2).sum().item())

    signs = torch.sign(stacked_trimmed)
    agree_mask = (signs == consensus_sign.unsqueeze(0)) & (consensus_sign.unsqueeze(0) != 0)

    # Tally sign agreement and clashes among active coordinates
    active_mask = active_count >= 2
    clash_coords = active_mask & ((signs != consensus_sign.unsqueeze(0)) & stacked_masks).any(dim=0)
    stats["sign_clashed"] = int(clash_coords.sum().item())
    stats["sign_agreed"] = int((active_mask & ~clash_coords).sum().item())

    # Disjoint merge: filter by consensus sign
    masked_updates = stacked_trimmed * agree_mask
    count_agree = agree_mask.float().sum(dim=0)

    valid_mask = count_agree > 0
    merged_flat = torch.zeros_like(sum_tau)
    if divide_by_count:
        merged_flat[valid_mask] = masked_updates.sum(dim=0)[valid_mask] / count_agree[valid_mask]
    else:
        merged_flat[valid_mask] = masked_updates.sum(dim=0)[valid_mask]

    res = (alpha * merged_flat).view(tau_list[0].shape).to(tau_list[0].dtype)
    return res, stats

def dare_tensors(tau: torch.Tensor, drop_rate: float = 0.50) -> torch.Tensor:
    """DARE: Randomly drops parameters and rescales survivors."""
    if drop_rate <= 0.0:
        return tau
    keep_prob = 1.0 - drop_rate
    mask = torch.bernoulli(torch.full_like(tau.float(), keep_prob)).to(torch.bool)
    res = (tau.float() * mask) / keep_prob
    return res.to(tau.dtype)

def merge_checkpoints(
    method: str,
    output_dir: str,
    base_sd: dict,
    math_sd: dict,
    pref_sd: dict,
    alpha_math: float = 0.50,
    alpha_pref: float = 0.50,
    t_slerp: float = 0.50,
    density_ties: float = 0.20,
    drop_dare: float = 0.50
):
    print(f"\n{'='*70}\nEXECUTING MERGE: {method.upper()} -> {output_dir}\n{'='*70}")
    merged_sd = {}
    aggregate_stats = {
        "method": method,
        "total_params": 0,
        "multiple_active": 0,
        "sign_agreed": 0,
        "sign_clashed": 0
    }

    for k in base_sd.keys():
        b_val = base_sd[k]
        m_val = math_sd[k]
        p_val = pref_sd[k]

        if not b_val.is_floating_point() or "embed" in k or "lm_head" in k:
            merged_sd[k] = b_val.clone()
            continue

        tau_m = m_val - b_val
        tau_p = p_val - b_val
        aggregate_stats["total_params"] += b_val.numel()

        if method == "linear":
            # Pure naive unscaled linear merge
            tau_merged = alpha_math * tau_m + alpha_pref * tau_p
            merged_sd[k] = b_val + tau_merged

        elif method == "linear_balanced":
            # Norm-balanced linear merge
            norm_m = torch.norm(tau_m.float()) + 1e-8
            norm_p = torch.norm(tau_p.float()) + 1e-8
            target_norm = (norm_m + norm_p) / 2.0
            tau_m_bal = (tau_m.float() / norm_m) * target_norm
            tau_p_bal = (tau_p.float() / norm_p) * target_norm
            tau_merged = (alpha_math * tau_m_bal + alpha_pref * tau_p_bal).to(b_val.dtype)
            merged_sd[k] = b_val + tau_merged

        elif method == "slerp":
            # Standard SLERP directly between full model weights (matching MergeKit)
            merged_sd[k] = slerp_tensor(m_val, p_val, t=t_slerp)

        elif method == "ties":
            tau_list = [alpha_math * tau_m, alpha_pref * tau_p]
            tau_merged, stats = ties_merge_tensors(tau_list, density=density_ties, alpha=1.0, divide_by_count=False)
            merged_sd[k] = b_val + tau_merged
            for stat_k in stats:
                if stat_k in aggregate_stats:
                    aggregate_stats[stat_k] += stats[stat_k]

        elif method == "dare_ties":
            tau_m_dare = dare_tensors(alpha_math * tau_m, drop_rate=drop_dare)
            tau_p_dare = dare_tensors(alpha_pref * tau_p, drop_rate=drop_dare)
            tau_merged, stats = ties_merge_tensors([tau_m_dare, tau_p_dare], density=density_ties, alpha=1.0, divide_by_count=False)
            merged_sd[k] = b_val + tau_merged
            for stat_k in stats:
                if stat_k in aggregate_stats:
                    aggregate_stats[stat_k] += stats[stat_k]

        else:
            raise ValueError(f"Unknown merge method: {method}")

    if method in ["ties", "dare_ties"] and aggregate_stats.get("sign_clashed", 0) > 0:
        c = aggregate_stats["sign_clashed"]
        ma = aggregate_stats.get("multiple_active", 0)
        sa = aggregate_stats.get("sign_agreed", 0)
        print("\n--- TIES Sign Consensus Analysis ---")
        print(f"Total Multi-Active Coordinates: {ma:,}")
        print(f"Total Agreed Coordinates      : {sa:,}")
        print(f"Total Conflicting Coordinates : {c:,}")
        print("------------------------------------")

    # Save merged model to disk
    print(f"Saving merged weights to: {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load base model skeleton on CPU to save complete HuggingFace repo
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    model.load_state_dict(merged_sd)
    model.save_pretrained(output_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(output_dir)

    del model
    del merged_sd

    # Save merge metadata
    meta_path = os.path.join(output_dir, "merge_meta.json")
    with open(meta_path, "w") as f:
        json.dump(aggregate_stats, f, indent=2)
    print(f"Successfully saved {method} model to {output_dir}")
    return aggregate_stats

def merge_4_experts(
    method: str,
    output_dir: str,
    base_sd: dict,
    pref_sd: dict,
    math_sd: dict,
    code_sd: dict,
    instr_sd: dict,
    weights: dict[str, float] = None,
    density_ties: float = 0.20,
    drop_dare: float = 0.50
) -> dict:
    """
    Merges all 4 domain specialists (E_pref, E_math, E_code, E_instr) simultaneously.
    Supports methods: 'linear', 'linear_balanced', 'ties', 'dare_ties'.
    weights: dictionary specifying each model's contribution (e.g., {'pref': 0.4, 'math': 0.2, 'code': 0.2, 'instr': 0.2}).
    """
    if weights is None:
        weights = {"pref": 0.25, "math": 0.25, "code": 0.25, "instr": 0.25}

    print(f"\n{'='*70}\nEXECUTING 4-EXPERT MERGE: {method.upper()} -> {output_dir}\nWeights: {weights}\n{'='*70}")
    merged_sd = {}
    aggregate_stats = {
        "method": method,
        "weights": weights,
        "total_params": 0,
        "multiple_active": 0,
        "sign_agreed": 0,
        "sign_clashed": 0
    }

    # Clean prefixes on all models
    base_sd = clean_state_dict_keys(base_sd)
    pref_sd = clean_state_dict_keys(pref_sd)
    math_sd = clean_state_dict_keys(math_sd)
    code_sd = clean_state_dict_keys(code_sd)
    instr_sd = clean_state_dict_keys(instr_sd)

    w_p = weights.get("pref", 0.25)
    w_m = weights.get("math", 0.25)
    w_c = weights.get("code", 0.25)
    w_i = weights.get("instr", 0.25)

    for k in base_sd.keys():
        b_val = base_sd[k]
        if k not in pref_sd or k not in math_sd or k not in code_sd or k not in instr_sd:
            merged_sd[k] = b_val.clone()
            continue

        p_val = pref_sd[k]
        m_val = math_sd[k]
        c_val = code_sd[k]
        i_val = instr_sd[k]

        if not b_val.is_floating_point() or "embed" in k or "lm_head" in k:
            merged_sd[k] = b_val.clone()
            continue

        tau_p = p_val - b_val
        tau_m = m_val - b_val
        tau_c = c_val - b_val
        tau_i = i_val - b_val
        aggregate_stats["total_params"] += b_val.numel()

        if method == "linear":
            tau_merged = w_p * tau_p + w_m * tau_m + w_c * tau_c + w_i * tau_i
            merged_sd[k] = b_val + tau_merged

        elif method == "linear_balanced":
            norms = [torch.norm(t.float()) + 1e-8 for t in [tau_p, tau_m, tau_c, tau_i]]
            avg_norm = sum(norms) / len(norms)
            tau_p_b = (tau_p.float() / norms[0]) * avg_norm
            tau_m_b = (tau_m.float() / norms[1]) * avg_norm
            tau_c_b = (tau_c.float() / norms[2]) * avg_norm
            tau_i_b = (tau_i.float() / norms[3]) * avg_norm
            tau_merged = (w_p * tau_p_b + w_m * tau_m_b + w_c * tau_c_b + w_i * tau_i_b).to(b_val.dtype)
            merged_sd[k] = b_val + tau_merged

        elif method == "ties":
            tau_list = [w_p * tau_p, w_m * tau_m, w_c * tau_c, w_i * tau_i]
            tau_merged, stats = ties_merge_tensors(tau_list, density=density_ties, alpha=1.0, divide_by_count=False)
            merged_sd[k] = b_val + tau_merged
            for stat_k in stats:
                if stat_k in aggregate_stats:
                    aggregate_stats[stat_k] += stats[stat_k]

        elif method == "dare_ties":
            tau_list = [
                dare_tensors(w_p * tau_p, drop_rate=drop_dare),
                dare_tensors(w_m * tau_m, drop_rate=drop_dare),
                dare_tensors(w_c * tau_c, drop_rate=drop_dare),
                dare_tensors(w_i * tau_i, drop_rate=drop_dare),
            ]
            tau_merged, stats = ties_merge_tensors(tau_list, density=density_ties, alpha=1.0, divide_by_count=False)
            merged_sd[k] = b_val + tau_merged
            for stat_k in stats:
                if stat_k in aggregate_stats:
                    aggregate_stats[stat_k] += stats[stat_k]

        else:
            raise ValueError(f"Unknown merge method: {method}")

    if method in ["ties", "dare_ties"] and aggregate_stats.get("sign_clashed", 0) > 0:
        c = aggregate_stats["sign_clashed"]
        ma = aggregate_stats.get("multiple_active", 0)
        sa = aggregate_stats.get("sign_agreed", 0)
        print("\n--- 4-Expert TIES Sign Consensus Analysis ---")
        print(f"Total Multi-Active Coordinates: {ma:,}")
        print(f"Total Agreed Coordinates      : {sa:,}")
        print(f"Total Conflicting Coordinates : {c:,}")
        print("---------------------------------------------")

    print(f"Saving merged 4-expert weights to: {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    model.load_state_dict(merged_sd)
    model.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(output_dir)

    del model
    del merged_sd

    meta_path = os.path.join(output_dir, "merge_meta.json")
    with open(meta_path, "w") as f:
        json.dump(aggregate_stats, f, indent=2)
    print(f"Successfully saved 4-expert {method} model to {output_dir}")
    return aggregate_stats

def run_phase_2a_merges():
    """Generates the 4 canonical merged checkpoints for Phase 2A (Midpoint Sanity Check)."""
    base_sd = load_state_dict_cpu(MODEL_ID)
    math_sd = load_state_dict_cpu("outputs/checkpoints/e_math")
    pref_sd = load_state_dict_cpu("outputs/checkpoints/e_pref")

    merges = [
        ("linear", "outputs/checkpoints/merge_linear_naive"),
        ("linear_balanced", "outputs/checkpoints/merge_linear_balanced"),
        ("slerp", "outputs/checkpoints/merge_slerp"),
        ("ties", "outputs/checkpoints/merge_ties"),
        ("dare_ties", "outputs/checkpoints/merge_dare_ties"),
    ]

    all_stats = {}
    for method, out_path in merges:
        stats = merge_checkpoints(
            method=method,
            output_dir=out_path,
            base_sd=base_sd,
            math_sd=math_sd,
            pref_sd=pref_sd,
            alpha_math=0.50,
            alpha_pref=0.50,
            t_slerp=0.50,
            density_ties=0.20,
            drop_dare=0.50
        )
        all_stats[method] = stats

    stats_out = "outputs/results/phase2a_merge_stats.json"
    os.makedirs(os.path.dirname(stats_out), exist_ok=True)
    with open(stats_out, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nAll Phase 2A merges created successfully! Metadata saved to {stats_out}")

if __name__ == "__main__":
    run_phase_2a_merges()
