#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""
End-to-end test: Compare HuggingFace model vs Bridge-converted MCore model.

Simply loads both models and compares forward/backward outputs.
"""

import torch
import argparse
from transformers import AutoConfig, AutoModelForCausalLM


def compare_tensors(name: str, t1: torch.Tensor, t2: torch.Tensor, rtol: float = 1e-4, atol: float = 1e-5):
    """Compare two tensors and print results."""
    if t1.shape != t2.shape:
        print(f"  {name}: SHAPE MISMATCH {t1.shape} vs {t2.shape}")
        return False

    max_diff = (t1 - t2).abs().max().item()
    mean_diff = (t1 - t2).abs().mean().item()
    rel_diff = ((t1 - t2).abs() / (t1.abs() + 1e-8)).mean().item()

    passed = torch.allclose(t1, t2, rtol=rtol, atol=atol)
    status = "✓" if passed else "✗"

    print(f"  {name}: {status} max={max_diff:.2e}, mean={mean_diff:.2e}, rel={rel_diff:.2e}")
    return passed


def test_forward_backward(
    model_path: str = "openai/gpt-oss-120b",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = 2,
    seq_len: int = 128,
):
    """
    Compare HF model and Bridge model forward/backward.

    Args:
        model_path: HuggingFace model path
        device: cuda or cpu
        dtype: torch.float32, torch.float16, or torch.bfloat16
        batch_size: batch size for test
        seq_len: sequence length for test
    """
    print("=" * 70)
    print("GPT-OSS End-to-End Test: HuggingFace vs Bridge Model")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Device: {device}, Dtype: {dtype}")
    print(f"Batch: {batch_size}, SeqLen: {seq_len}")
    print()

    # =========================================================================
    # 1. Load HuggingFace Model
    # =========================================================================
    print("[1/4] Loading HuggingFace model...")
    hf_config = AutoConfig.from_pretrained(model_path)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
    )
    hf_model.eval()
    print(f"  HF model loaded: {sum(p.numel() for p in hf_model.parameters()):,} params")

    # =========================================================================
    # 2. Load Bridge Model (MCore)
    # =========================================================================
    print("\n[2/4] Loading Bridge model (MCore)...")
    from mbridge.core import ParallelStates
    from slime_plugins.mbridge import GptOssBridge

    parallel_states = ParallelStates(tp_size=1, pp_size=1, ep_size=1, cp_size=1)

    bridge = GptOssBridge(
        hf_config=hf_config,
        dtype=dtype,
        parallel_states=parallel_states,
    )

    mcore_model = bridge.get_model(weight_path=model_path, bf16=(dtype == torch.bfloat16))
    if isinstance(mcore_model, list):
        mcore_model = mcore_model[0]
    mcore_model.to(device)
    mcore_model.eval()
    print(f"  MCore model loaded: {sum(p.numel() for p in mcore_model.parameters()):,} params")

    # =========================================================================
    # 3. Prepare Input
    # =========================================================================
    print("\n[3/4] Preparing test input...")
    torch.manual_seed(42)
    input_ids = torch.randint(0, hf_config.vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # For gradient comparison
    labels = torch.randint(0, hf_config.vocab_size, (batch_size, seq_len), device=device)

    print(f"  Input shape: {input_ids.shape}")

    # =========================================================================
    # 4. Forward Pass Comparison
    # =========================================================================
    print("\n[4/4] Comparing Forward Pass...")

    # HF forward
    with torch.no_grad():
        hf_output = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hf_logits = hf_output.logits

    # MCore forward
    with torch.no_grad():
        mcore_output = mcore_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        # MCore output format may differ
        if hasattr(mcore_output, 'logits'):
            mcore_logits = mcore_output.logits
        elif isinstance(mcore_output, torch.Tensor):
            mcore_logits = mcore_output
        else:
            mcore_logits = mcore_output[0]

    print("\n  Forward Pass Results:")
    forward_passed = compare_tensors("Logits", hf_logits, mcore_logits)

    # =========================================================================
    # 5. Backward Pass Comparison
    # =========================================================================
    print("\n  Backward Pass Results:")

    # HF backward
    hf_model.train()
    hf_output = hf_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        labels=labels,
    )
    hf_loss = hf_output.loss
    hf_loss.backward()

    # Collect HF gradients
    hf_grads = {}
    for name, param in hf_model.named_parameters():
        if param.grad is not None:
            hf_grads[name] = param.grad.clone()

    # MCore backward
    mcore_model.train()
    mcore_output = mcore_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    if hasattr(mcore_output, 'logits'):
        mcore_logits = mcore_output.logits
    elif isinstance(mcore_output, torch.Tensor):
        mcore_logits = mcore_output
    else:
        mcore_logits = mcore_output[0]

    # Compute same loss
    mcore_loss = torch.nn.functional.cross_entropy(
        mcore_logits.view(-1, hf_config.vocab_size),
        labels.view(-1),
    )
    mcore_loss.backward()

    # Compare loss
    loss_passed = compare_tensors("Loss", hf_loss.detach(), mcore_loss.detach())

    # Compare key gradients (sample a few)
    grad_checks = []
    sample_layers = ["embed_tokens", "layers.0.self_attn.q_proj", "layers.0.mlp", "lm_head"]

    for key in sample_layers:
        for hf_name, hf_grad in hf_grads.items():
            if key in hf_name and "weight" in hf_name:
                # Find corresponding MCore param
                for mcore_name, mcore_param in mcore_model.named_parameters():
                    if mcore_param.grad is not None:
                        # Rough name matching (may need adjustment)
                        if key.split(".")[-1] in mcore_name:
                            passed = compare_tensors(
                                f"Grad[{key}]",
                                hf_grad,
                                mcore_param.grad,
                                rtol=1e-3, atol=1e-4
                            )
                            grad_checks.append(passed)
                            break
                break

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    all_passed = forward_passed and loss_passed and all(grad_checks)

    print(f"  Forward Pass:  {'✓ PASSED' if forward_passed else '✗ FAILED'}")
    print(f"  Loss Match:    {'✓ PASSED' if loss_passed else '✗ FAILED'}")
    print(f"  Gradients:     {'✓ PASSED' if all(grad_checks) else '✗ FAILED'} ({sum(grad_checks)}/{len(grad_checks)})")
    print()
    print(f"  Overall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("=" * 70)

    return all_passed


def test_simple_forward(
    model_path: str = "openai/gpt-oss-120b",
    device: str = "cuda",
):
    """Simplified test - just forward pass comparison."""
    print("=" * 70)
    print("Simple Forward Test: HF vs Bridge")
    print("=" * 70)

    # Load HF
    print("Loading HF model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device
    )

    # Load Bridge
    print("Loading Bridge model...")
    from mbridge.core import ParallelStates
    from slime_plugins.mbridge import GptOssBridge

    hf_config = AutoConfig.from_pretrained(model_path)
    ps = ParallelStates(tp_size=1, pp_size=1, ep_size=1, cp_size=1)
    bridge = GptOssBridge(hf_config=hf_config, dtype=torch.bfloat16, parallel_states=ps)
    mcore_model = bridge.get_model(weight_path=model_path, bf16=True)
    if isinstance(mcore_model, list):
        mcore_model = mcore_model[0]
    mcore_model.to(device)

    # Test input
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)

    # Forward
    print("Running forward pass...")
    with torch.no_grad():
        hf_out = hf_model(input_ids).logits
        mcore_out = mcore_model(input_ids)
        if hasattr(mcore_out, 'logits'):
            mcore_out = mcore_out.logits
        elif not isinstance(mcore_out, torch.Tensor):
            mcore_out = mcore_out[0]

    # Compare
    max_diff = (hf_out - mcore_out).abs().max().item()
    print(f"\nMax difference: {max_diff:.2e}")
    print(f"Result: {'✓ PASSED' if max_diff < 1e-3 else '✗ FAILED'}")

    return max_diff < 1e-3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GptOssBridge conversion")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Model path")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--simple", action="store_true", help="Run simple forward test only")

    args = parser.parse_args()

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    if args.simple:
        success = test_simple_forward(args.model, args.device)
    else:
        success = test_forward_backward(
            model_path=args.model,
            device=args.device,
            dtype=dtype_map[args.dtype],
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        )

    exit(0 if success else 1)
