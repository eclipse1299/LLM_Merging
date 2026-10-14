"""
MBPP (Pass@1) Evaluation Suite with 3-Second Timeout Worker Process.
Evaluates Python code generation on MBPP using concurrent.futures.ProcessPoolExecutor
to protect against infinite loops or hanging execution.
"""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import sys
import re
import json
import argparse
import subprocess
import torch
torch.set_num_threads(2)
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from configs.base_config import TORCH_DTYPE, ATTN_IMPLEMENTATION


def run_with_timeout(program: str, test_assertions: list[str], timeout: float = 3.0) -> bool:
    """Runs test execution in a clean subprocess with an explicit OS-level timeout and kill.
    Guarantees no child processes hang indefinitely on Windows.
    """
    full_code = program + "\n" + "\n".join(test_assertions)
    try:
        p = subprocess.Popen(
            [sys.executable, "-c", full_code],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL
        )
        try:
            ret = p.wait(timeout=timeout)
            return ret == 0
        except subprocess.TimeoutExpired:
            p.kill()
            try:
                p.wait(timeout=1.0)
            except Exception:
                pass
            return False
    except Exception:
        return False


def extract_code(text: str) -> str:
    """Extracts python code blocks from generated text."""
    pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n\n".join(m.strip() for m in matches if m.strip())
    pattern_generic = r"```\s*(.*?)\s*```"
    generic_matches = re.findall(pattern_generic, text, re.DOTALL)
    if generic_matches:
        return "\n\n".join(m.strip() for m in generic_matches if m.strip())
    return text


def evaluate_mbpp(
    model_path: str,
    max_samples: int = None,
    output_json: str = None
):
    print(f"Loading MBPP dataset...")
    # Load sanitized MBPP or standard split
    try:
        dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    except Exception:
        dataset = load_dataset("mbpp", "sanitized", split="test")

    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    print(f"Evaluating {len(dataset)} MBPP problems on model: {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=TORCH_DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION
    ).to("cuda")
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    passed_count = 0
    results = []
    batch_size = 16

    items = list(dataset)
    for i in tqdm(range(0, len(items), batch_size), desc="Evaluating MBPP (Batched)"):
        batch = items[i:i + batch_size]
        prompts = []
        for item in batch:
            prompt_text = item["prompt"]
            test_list = item["test_list"]
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": f"Write a Python function to solve the following problem:\n{prompt_text}\nYour code should satisfy these tests:\n" + "\n".join(test_list[:1])}
            ]
            formatted_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(formatted_input)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("<|im_end|>")[0]]
            )

        gen_tokens = outputs[:, input_len:]
        decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        for item, gen_text in zip(batch, decoded):
            code = extract_code(gen_text)
            passed = run_with_timeout(code, item["test_list"], timeout=3.0)
            if passed:
                passed_count += 1
            results.append({
                "task_id": item.get("task_id", 0),
                "prompt": item["prompt"],
                "generated_code": code,
                "passed": passed
            })

    import gc
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pass_at_1 = (passed_count / len(dataset)) * 100.0
    print(f"\n==========================================")
    print(f"MBPP Pass@1: {pass_at_1:.2f}% ({passed_count}/{len(dataset)})")
    print(f"==========================================")

    final_payload = {
        "model": model_path,
        "pass_at_1": pass_at_1,
        "total": len(dataset),
        "passed": passed_count,
        "details": results
    }

    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(final_payload, f, indent=2)
        print(f"Saved evaluation results to: {output_json}")

    return final_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on MBPP with 3s worker timeout.")
    parser.add_argument("--model_path", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate (e.g. 200 for dev)")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save results JSON")
    args = parser.parse_args()

    evaluate_mbpp(args.model_path, args.max_samples, args.output_json)
