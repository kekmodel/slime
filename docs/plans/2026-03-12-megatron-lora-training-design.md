# Megatron Backend LoRA Training Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add LoRA (Low-Rank Adaptation) training support to slime's Megatron backend, including MoE expert LoRA, with merge-based weight transfer to preserve rollout throughput.

**Key Constraint:** Rollout time (480s) dominates iteration cycle (~85%). SGLang LoRA serving overhead (5-15%) is unacceptable. Therefore: merge LoRA weights into base before transfer, SGLang runs in normal mode.

**Key Benefits:**
- Train time reduction (~75s → ~55s): smaller optimizer states
- CPU memory savings: ref model = base model (adapter off), no separate copy
- GPU memory savings: optimizer states for LoRA params only (~0.1-1% of base)
- Checkpoint size: adapter-only saves (~99% reduction)
- Rollout throughput: zero impact (merged weights = standard model)

---

## 1. Architecture Overview

### 1.1 New Module: `slime/backends/megatron_utils/lora/`

```
slime/backends/megatron_utils/lora/
├── __init__.py
├── config.py         # LoRAConfig dataclass + CLI argument registration
├── layers.py         # TP-compatible LoRA layers (2D shared + 3D expert)
├── injection.py      # inject_lora_adapters(model, config) + freeze base
└── merge.py          # merge/unmerge LoRA weights for transfer + ref forward
```

### 1.2 Modified Existing Files

| File | Changes |
|------|---------|
| `arguments.py` | Add `--lora-rank`, `--lora-alpha`, `--lora-target-modules`, `--lora-dropout` |
| `model_provider.py` | Add `wrap_model_provider_with_lora()` in model construction pipeline |
| `actor.py` | Merge before `update_weights()`, adapter-off for ref forward |
| `model.py` | Adapter-only checkpoint save/load paths |
| `checkpoint.py` | Adapter checkpoint detection and overlay loading |

### 1.3 Data Flow

```
                    Training
                       │
    ┌──────────────────┼──────────────────┐
    │                  │                  │
    ▼                  ▼                  ▼
 Actor Forward    Ref Forward       Weight Update
 (adapter ON)     (adapter OFF)     (merge → existing path)
    │                  │                  │
    │  W_base + B·A    │  W_base only     │  W_merged = W_base + B·A·scale
    │                  │                  │  → convert_to_hf
    │                  │                  │  → FlattenedTensorBucket
    │                  │                  │  → SGLang (normal mode)
    ▼                  ▼                  ▼
 actor_log_probs  ref_log_probs    Rollout Generation
                                   (zero overhead)
```

---

## 2. LoRA Layer Design

### 2.1 TP Sharding Rules

| Base Layer Type | lora_A shape | lora_A TP | lora_B shape | lora_B TP |
|----------------|-------------|-----------|-------------|-----------|
| ColumnParallel (QKV, FC1) | `[rank, H]` | Replicated | `[out/T, rank]` | Sharded dim=0 |
| RowParallel (o_proj, FC2) | `[rank, in/T]` | Sharded dim=1 | `[out, rank]` | Replicated |

Where T = tp_size, H = hidden_size.

### 2.2 Shared Layer LoRA (2D)

```python
class LoRAColumnParallelLinear(nn.Module):
    """Wraps a ColumnParallelLinear with LoRA adapter."""
    def __init__(self, base_layer, rank, alpha, dropout=0.0):
        self.base_layer = base_layer  # frozen
        self.lora_A = nn.Parameter(zeros(rank, base_layer.input_size))
        self.lora_B = nn.Parameter(zeros(base_layer.output_size_per_partition, rank))
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # TP attributes for lora_B
        self.lora_B.tensor_model_parallel = True
        self.lora_B.partition_dim = 0

    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out

class LoRARowParallelLinear(nn.Module):
    """Wraps a RowParallelLinear with LoRA adapter."""
    def __init__(self, base_layer, rank, alpha, dropout=0.0):
        self.base_layer = base_layer  # frozen
        self.lora_A = nn.Parameter(zeros(rank, base_layer.input_size_per_partition))
        self.lora_B = nn.Parameter(zeros(base_layer.output_size, rank))
        self.scaling = alpha / rank
        # TP attributes for lora_A
        self.lora_A.tensor_model_parallel = True
        self.lora_A.partition_dim = 1

    def forward(self, x):
        base_out = self.base_layer(x)
        # lora_A @ x produces partial sum → added to base partial sum → same reduce
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out
```

### 2.3 Fused QKV LoRA

Megatron fuses Q, K, V into a single `linear_qkv` (ColumnParallel). We use separate Q/K/V adapters whose B outputs are arranged in Megatron's interleaved group layout:

```python
class LoRAFusedQKV(nn.Module):
    """Separate Q/K/V LoRA adapters on fused linear_qkv."""
    def __init__(self, base_layer, rank, alpha, args):
        self.base_layer = base_layer
        H = base_layer.input_size
        # Per-TP-rank output dimensions
        q_out = (args.num_attention_heads // tp_size) * head_dim
        kv_out = (args.num_query_groups // tp_size) * head_dim

        self.q_lora_A = nn.Parameter(zeros(rank, H))       # replicated
        self.q_lora_B = nn.Parameter(zeros(q_out, rank))    # TP sharded dim=0
        self.k_lora_A = nn.Parameter(zeros(rank, H))
        self.k_lora_B = nn.Parameter(zeros(kv_out, rank))
        self.v_lora_A = nn.Parameter(zeros(rank, H))
        self.v_lora_B = nn.Parameter(zeros(kv_out, rank))
        self.scaling = alpha / rank

    def forward(self, x):
        base_out = self.base_layer(x)
        q_corr = F.linear(F.linear(x, self.q_lora_A), self.q_lora_B) * self.scaling
        k_corr = F.linear(F.linear(x, self.k_lora_A), self.k_lora_B) * self.scaling
        v_corr = F.linear(F.linear(x, self.v_lora_A), self.v_lora_B) * self.scaling
        lora_out = interleave_qkv_corrections(q_corr, k_corr, v_corr, args)
        return base_out + lora_out
```

`interleave_qkv_corrections` rearranges [Q_all, K_all, V_all] into Megatron's per-group interleaved format: [Q_group0, K_group0, V_group0, Q_group1, K_group1, V_group1, ...].

### 2.4 Fused FC1 (SwiGLU) LoRA

```python
class LoRAFusedFC1(nn.Module):
    """Separate gate/up LoRA adapters on fused linear_fc1."""
    def __init__(self, base_layer, rank, alpha):
        H = base_layer.input_size
        half_out = base_layer.output_size_per_partition // 2

        self.gate_lora_A = nn.Parameter(zeros(rank, H))
        self.gate_lora_B = nn.Parameter(zeros(half_out, rank))
        self.up_lora_A = nn.Parameter(zeros(rank, H))
        self.up_lora_B = nn.Parameter(zeros(half_out, rank))
        self.scaling = alpha / rank

    def forward(self, x):
        base_out = self.base_layer(x)
        gate_corr = F.linear(F.linear(x, self.gate_lora_A), self.gate_lora_B) * self.scaling
        up_corr = F.linear(F.linear(x, self.up_lora_A), self.up_lora_B) * self.scaling
        lora_out = torch.cat([gate_corr, up_corr], dim=-1)  # [gate, up] order
        return base_out + lora_out
```

### 2.5 MoE Expert LoRA (3D Batched)

MoE experts use `GroupedMLP` with stacked 3D weight tensors `[num_local_experts, out, in]`. LoRA adapters follow the same stacked format using `torch.bmm` for efficient batched computation:

```python
class LoRAGroupedExpertFC1(nn.Module):
    """Per-expert LoRA on GroupedMLP fc1 (3D stacked, SwiGLU)."""
    def __init__(self, base_layer, num_local_experts, rank, alpha, args):
        H = args.hidden_size
        half_out = args.ffn_hidden_size // expt_tp_size  # per expert-TP-rank

        # 3D stacked: [num_local_experts, dim, rank]
        self.gate_lora_A = nn.Parameter(zeros(num_local_experts, rank, H))
        self.gate_lora_B = nn.Parameter(zeros(num_local_experts, half_out, rank))
        self.up_lora_A = nn.Parameter(zeros(num_local_experts, rank, H))
        self.up_lora_B = nn.Parameter(zeros(num_local_experts, half_out, rank))
        self.scaling = alpha / rank

    def forward(self, x, expert_tokens):
        """
        x: tokens already routed to experts, shape varies by expert
        expert_tokens: per-expert token assignment
        """
        base_out = self.base_layer(x, expert_tokens)
        # Batched LoRA: bmm for all local experts simultaneously
        # Implementation depends on GroupedMLP's internal dispatch
        lora_corrections = batched_expert_lora(
            x, expert_tokens,
            self.gate_lora_A, self.gate_lora_B,
            self.up_lora_A, self.up_lora_B,
            self.scaling
        )
        return base_out + lora_corrections

class LoRAGroupedExpertFC2(nn.Module):
    """Per-expert LoRA on GroupedMLP fc2 (3D stacked, RowParallel-like)."""
    def __init__(self, base_layer, num_local_experts, rank, alpha, args):
        in_per_tp = args.ffn_hidden_size // expt_tp_size
        out = args.hidden_size

        self.lora_A = nn.Parameter(zeros(num_local_experts, rank, in_per_tp))
        self.lora_B = nn.Parameter(zeros(num_local_experts, out, rank))
        self.scaling = alpha / rank
```

**Expert TP handling:** Expert LoRA follows the same TP rules as shared layers but uses `expt_tp_group` instead of `tp_group`. The `partition_dim` and `tensor_model_parallel` attributes are set identically.

**Expert merge:** Per-expert merge before weight transfer:
```python
for e in range(num_local_experts):
    W_fc1[e] += torch.cat([
        gate_lora_B[e] @ gate_lora_A[e],
        up_lora_B[e] @ up_lora_A[e]
    ], dim=0) * scaling
    W_fc2[e] += lora_B[e] @ lora_A[e] * scaling
```
After merge, the 3D stacked weight goes through the existing EP all_gather + convert_to_hf pipeline unchanged.

---

## 3. Injection and Freezing

### 3.1 Injection Pipeline

```python
def inject_lora_adapters(model, lora_config):
    """Replace target modules with LoRA-wrapped versions."""
    for layer in model.decoder.layers:
        if "q_proj" in lora_config.target_modules or "k_proj" in ... or "v_proj" in ...:
            layer.self_attention.linear_qkv = LoRAFusedQKV(
                layer.self_attention.linear_qkv, lora_config.rank, lora_config.alpha, args
            )
        if "o_proj" in lora_config.target_modules:
            layer.self_attention.linear_proj = LoRARowParallelLinear(
                layer.self_attention.linear_proj, lora_config.rank, lora_config.alpha
            )
        if "gate_proj" in ... or "up_proj" in ...:
            layer.mlp.linear_fc1 = LoRAFusedFC1(
                layer.mlp.linear_fc1, lora_config.rank, lora_config.alpha
            )
        if "down_proj" in lora_config.target_modules:
            layer.mlp.linear_fc2 = LoRARowParallelLinear(
                layer.mlp.linear_fc2, lora_config.rank, lora_config.alpha
            )
        # MoE expert LoRA
        if hasattr(layer.mlp, 'experts') and "expert" in lora_config.target_modules:
            inject_expert_lora(layer.mlp.experts, lora_config)

def freeze_base_params(model):
    """Freeze all non-LoRA parameters."""
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False
```

### 3.2 Integration in model_provider.py

```python
def wrap_model_provider_with_lora(original_provider, args):
    def wrapped(pre_process=True, post_process=True, vp_stage=None):
        model = original_provider(pre_process, post_process, vp_stage)
        if args.lora_rank > 0:
            lora_config = LoRAConfig.from_args(args)
            inject_lora_adapters(model, lora_config)
            freeze_base_params(model)
        return model
    return wrapped
```

This wraps the existing model provider, applied before `wrap_model_provider_with_freeze`.

---

## 4. Merge and Weight Transfer

### 4.1 Merge Operation

```python
def merge_lora_weights(model):
    """Merge LoRA adapters into base weights in-place. Call before weight transfer."""
    for module in model.modules():
        if isinstance(module, (LoRAColumnParallelLinear, LoRARowParallelLinear)):
            module.base_layer.weight.data += (
                module.lora_B @ module.lora_A * module.scaling
            )
        elif isinstance(module, LoRAFusedQKV):
            _merge_fused_qkv(module)
        elif isinstance(module, LoRAFusedFC1):
            _merge_fused_fc1(module)
        elif isinstance(module, (LoRAGroupedExpertFC1, LoRAGroupedExpertFC2)):
            _merge_expert_lora(module)

def unmerge_lora_weights(model):
    """Reverse merge. Call after weight transfer to restore training state."""
    # Same operations with subtraction instead of addition
```

### 4.2 Integration in actor.py

```python
def update_weights(self):
    if self.args.lora_rank > 0:
        merge_lora_weights(self.model)      # merge into base
    self.weight_updater.update_weights()     # existing path (unchanged)
    if self.args.lora_rank > 0:
        unmerge_lora_weights(self.model)     # restore for continued training
```

The existing `update_weights` path (convert_to_hf → FlattenedTensorBucket → SGLang) works without any modification.

---

## 5. Ref Model Handling (RL Optimization)

### 5.1 Current Flow (Without LoRA)

```
_switch_model("ref")
  → TensorBackuper.restore("ref")     # CPU → GPU copy of entire model
  → forward pass                       # ref_log_probs
  → _switch_model("actor")            # restore actor weights
```
Cost: 2x full model CPU→GPU copy per training step.

### 5.2 LoRA Flow

```
_disable_lora(model)                   # set all lora scaling to 0
  → forward pass                       # ref_log_probs (base model only)
_enable_lora(model)                    # restore scaling
```

Implementation:
```python
def disable_lora(model):
    for module in model.modules():
        if hasattr(module, 'scaling'):
            module._saved_scaling = module.scaling
            module.scaling = 0.0

def enable_lora(model):
    for module in model.modules():
        if hasattr(module, '_saved_scaling'):
            module.scaling = module._saved_scaling
```

Benefits:
- No CPU→GPU weight copy for ref model
- No "ref" entry in TensorBackuper (saves model_size bytes of CPU memory)
- Negligible overhead (~microseconds to toggle scaling)

### 5.3 Backward Compatibility

When `args.lora_rank == 0`, the existing ref model swap mechanism continues to work unchanged.

---

## 6. Checkpoint

### 6.1 Save (Adapter-Only)

```python
def save_lora_checkpoint(model, path, args):
    adapter_state = {}
    for name, param in model.named_parameters():
        if 'lora_' in name:
            adapter_state[name] = param.data.cpu()
    torch.save(adapter_state, path / "adapter_model.bin")
    save_lora_config(args, path / "lora_config.json")
```

### 6.2 Load (Base + Adapter Overlay)

```python
def load_lora_checkpoint(model, base_path, adapter_path, args):
    # 1. Load base model (existing path)
    load_checkpoint(model, base_path)
    # 2. Inject LoRA structure (if not already present)
    if args.lora_rank > 0:
        inject_lora_adapters(model, LoRAConfig.from_args(args))
        freeze_base_params(model)
    # 3. Overlay adapter weights
    if adapter_path:
        adapter_state = torch.load(adapter_path / "adapter_model.bin")
        model.load_state_dict(adapter_state, strict=False)
```

### 6.3 Full Checkpoint Compatibility

When `--save-full-checkpoint` is passed, merge + save full model for deployment without LoRA infrastructure.

---

## 7. Configuration

### 7.1 New CLI Arguments

```
--lora-rank              LoRA rank (default: 0, disabled)
--lora-alpha             LoRA alpha for scaling (default: 2x rank)
--lora-target-modules    Comma-separated target modules
                         (choices: q_proj, k_proj, v_proj, o_proj,
                          gate_proj, up_proj, down_proj, expert)
--lora-dropout           LoRA dropout rate (default: 0.0)
--lora-init-method       Initialization method for LoRA A (default: kaiming)
--save-adapter-only      Save only adapter weights in checkpoint (default: True when lora_rank > 0)
--adapter-load           Path to load adapter checkpoint from
```

### 7.2 Validation

- `--lora-rank > 0` implies LoRA mode
- `--lora-target-modules` defaults to `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- `--lora-rank` and `--only-train-params-name-list` are mutually exclusive
- `expert` in target_modules only valid when `--num-experts > 0`

---

## 8. Expert LoRA Specifics

### 8.1 GroupedMLP Weight Format

Megatron's `GroupedMLP` stores expert weights as 3D stacked tensors:
- `weight1` (fc1): `[num_local_experts, 2 * ffn_hidden / expt_tp, hidden_size]`
- `weight2` (fc2): `[num_local_experts, hidden_size, ffn_hidden / expt_tp]`

### 8.2 Expert LoRA Merge + Transfer

After merge, expert weights remain in 3D format. The existing transfer pipeline handles them:
1. `all_gather_param` with `expt_tp_group` → full per-expert tensors
2. EP `all_gather` → all experts on source rank
3. `convert_to_hf` → HF-named individual expert tensors
4. Transfer to SGLang → standard model weights

No changes needed in `convert_to_hf` or `update_weight/` for expert LoRA (merge makes it transparent).

### 8.3 Expert LoRA Parameter Count

With 256 experts, rank=64, ffn_hidden=2048, hidden=7168:
- Per expert fc1 LoRA: 2 × (64 × 7168 + half_ffn × 64) × 2B ≈ 2-3 MB
- Per expert fc2 LoRA: (64 × half_ffn + 7168 × 64) × 2B ≈ 1-2 MB
- 256 experts total: ~800 MB - 1.2 GB
- Combined with shared LoRA (~1.3 GB): total ~2-2.5 GB adapter

This is manageable — ~0.3% of total model size.

---

## 9. Testing Strategy

1. **Unit tests**: LoRA layer forward/backward correctness, merge/unmerge roundtrip
2. **TP correctness**: Compare LoRA output across TP=1 vs TP>1 (must be numerically identical)
3. **Merge equivalence**: Verify merged model output == LoRA model output
4. **Ref model**: Verify adapter-off output == base model output
5. **Checkpoint roundtrip**: Save adapter → load adapter → verify identical outputs
6. **Integration**: End-to-end training run with LoRA on a small model
