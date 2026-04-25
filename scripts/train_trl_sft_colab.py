#!/usr/bin/env python3
"""
scripts/train_trl_sft_colab.py
================================
Colab-friendly TRL SFT training entry point.

Reads demonstrations from results/demos.jsonl and fine-tunes
TinyLlama-1.1B-Chat via HuggingFace TRL SFTTrainer.
Falls back to CPU policy adaptation if TRL/GPU unavailable.

Usage (Colab):
    python scripts/train_trl_sft_colab.py \\
        --demos results/demos.jsonl \\
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --output results/trained_model

    # CPU-only fallback (no GPU needed):
    python scripts/train_trl_sft_colab.py --cpu-only
"""
import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_demos(demos_path: str):
    records = []
    with open(demos_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} demo steps from {demos_path}")
    return records


def build_prompt(record: dict) -> str:
    """Convert a demo record into a training prompt."""
    obs = {
        "robots":          record.get("robots", {}),
        "completed_orders": record.get("completed", []),
    }
    action = record.get("action", {})
    prog   = record.get("program_progress")
    neg    = record.get("negotiation_events", [])

    prog_str = ""
    if prog:
        prog_str = (f"\nProgram compliance: {prog.get('compliance_score','?')} "
                    f"({prog.get('met','?')}/{prog.get('total','?')} constraints met)")
    neg_str = f"\nNegotiation events: {len(neg)}" if neg else ""

    return (
        f"<|system|>You are a warehouse robot agent.</s>\n"
        f"<|user|>State: {json.dumps(obs)}"
        f"{prog_str}{neg_str}</s>\n"
        f"<|assistant|>{json.dumps(action)}</s>"
    )


def run_trl_sft(demos, model_name, output_dir, max_steps=500):
    """GPU path: TRL SFTTrainer."""
    try:
        from datasets import Dataset
        from trl import SFTTrainer, SFTConfig
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig
    except ImportError:
        print("TRL/transformers not available. Use --cpu-only for fallback.")
        return False

    texts = [build_prompt(r) for r in demos]
    dataset = Dataset.from_dict({"text": texts})

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    lora_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                          target_modules=["q_proj","v_proj"])

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=output_dir,
            max_steps=max_steps,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            logging_steps=10,
            save_steps=200,
            fp16=True,
            report_to="none",
        ),
        peft_config=lora_cfg,
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")
    return True


def run_cpu_fallback(demos, output_dir):
    """CPU path: frequency-weighted action adaptation (no GPU needed)."""
    from collections import defaultdict, Counter

    action_counts: dict = defaultdict(Counter)
    for r in demos:
        state_key = str(r.get("completed", []))
        act = r.get("action", {}).get("action_type", "move")
        action_counts[state_key][act] += 1

    results = {
        "method":        "cpu_frequency_adaptation",
        "total_demos":   len(demos),
        "state_buckets": len(action_counts),
        "top_actions":   {k: dict(v.most_common(3)) for k, v in list(action_counts.items())[:5]},
    }
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "sft_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"CPU fallback results saved to {out}")
    return results


def main():
    ap = argparse.ArgumentParser(description="TRL SFT training from JSONL demos")
    ap.add_argument("--demos",    default="results/demos.jsonl")
    ap.add_argument("--model",    default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--output",   default="results/trained_model")
    ap.add_argument("--max-steps",type=int, default=500)
    ap.add_argument("--cpu-only", action="store_true",
                    help="Skip TRL, use CPU policy adaptation fallback")
    args = ap.parse_args()

    if not os.path.exists(args.demos):
        print(f"Demos file not found: {args.demos}")
        print("Run scripts/collect_demos_http.py first.")
        sys.exit(1)

    demos = load_demos(args.demos)

    if args.cpu_only:
        run_cpu_fallback(demos, args.output)
    else:
        ok = run_trl_sft(demos, args.model, args.output, args.max_steps)
        if not ok:
            print("TRL unavailable, falling back to CPU mode...")
            run_cpu_fallback(demos, args.output)


if __name__ == "__main__":
    main()
