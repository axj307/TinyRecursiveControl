# TRM Replication Review for TinyRecursiveControl

This document critically evaluates how TinyRecursiveControl (TRC) reproduces the Tiny Recursive Models (TRM) recursive reasoning for control tasks and outlines concrete gaps and fixes.

## Summary

- Intent: implement TRM-style recursive reasoning for control, in both flat and hierarchical forms.
- Verdict: TRC captures the core TRM loop and the two-level z_H/z_L hierarchy with shared L_level, H_cycles/L_cycles, gradient truncation, and process supervision. A few fidelity gaps remain (pre/post‑norm handling, error embedding dimension, ACT halting integration) and minor doc/code mismatches.

## What’s Implemented Well

- Two modes that mirror TRM:
  - Single-latent recursive refinement with shared reasoning across iterations (src/models/recursive_reasoning.py:135).
  - Two-level hierarchy (z_H, z_L) with shared L_level, scheduled L_cycles then H_step, and optional gradient truncation (src/models/recursive_reasoning.py:406, 496).
- Shared L_level applied to both z_H and z_L (weight sharing) (src/models/recursive_reasoning.py:422).
- Process supervision that rewards per-iteration cost reduction using differentiable dynamics and an LQR-like cost (src/training/process_supervision.py:117, 171).
- Optional value predictor (cost head) aligned with TRM’s Q-like auxiliary signal (src/models/value_predictor.py:15).
- Differentiable dynamics implemented for double integrator, Van der Pol, and rocket landing (src/environments/torch_dynamics.py:52, 111, 180).
- Factory presets for TRM-style configs (SwiGLU, RMSNorm, post‑norm, truncation) (src/models/tiny_recursive_control.py:520).

## Gaps vs TRM (Critical)

1) Pre‑norm vs Post‑norm toggle is not actually applied
- In `RecursiveReasoningBlock.forward`, both branches apply the same residual+norm order, so `norm_position` does not change behavior (src/models/recursive_reasoning.py:107–126).
- Impact: undermines TRM-style ablations (SwiGLU + RMSNorm + post‑norm) fidelity.
- Fix: implement genuine pre/post-norm flows for both attention and FFN:
  - Pre‑norm attention: `out = attn(norm1(z)); z = z + out`.
  - Post‑norm attention: `out = attn(z); z = norm1(z + out)`.
  - Pre‑norm FFN: `out = ffn(norm2(z)); z = z + out`.
  - Post‑norm FFN: `out = ffn(z); z = norm2(z + out)`.

2) Error embedding dimension hard-coded to 2
- `error_embedding = nn.Linear(2, latent_dim)` in both refinement modules (src/models/recursive_reasoning.py:198, 441).
- Breaks for problems where `state_dim != 2` (e.g., rocket landing state_dim=7) when using error feedback.
- Fix: parameterize with `state_dim` from config and set `error_embedding = nn.Linear(state_dim, latent_dim)`.

3) ErrorEncoder is unused
- `ErrorEncoder` is instantiated in the main model but error signals are embedded via separate linear layers in the refinement modules (src/models/tiny_recursive_control.py:100 vs src/models/recursive_reasoning.py:198/441).
- Fix: either remove `ErrorEncoder` completely or route `trajectory_error` through it and share it across modes.

4) ACT-style halting defined but not wired
- `AdaptiveRecursiveControl` exists but is unused in forward/training (src/models/recursive_reasoning.py:248).
- TRM’s ACT/Q halting is a key fidelity component. With a value head already present, wiring a halting decision per iteration would complete parity.

5) Attention over a single-token latent
- The latent is treated as length-1 sequence; MHA degenerates to a per-sample linear map (low utility).
- Fix (optional): default `use_attention=False` for vector-latent, or represent the control plan as a short sequence to make attention meaningful.

6) High-level input injection asymmetry
- Low-level uses `z_H + z_initial + control_context`; high-level uses only `z_L` (src/models/recursive_reasoning.py:508, 517).
- Fix (optional): inject `z_initial` (and possibly control/error context) into z_H as well for closer parity with TRM’s multi-source inputs.

## Doc/Code Mismatches

- Pendulum: README and scripts claimed support; no environment exists. Removed pendulum references in this change.
- Process supervision dynamics: double integrator and Van der Pol integrated; rocket landing dynamics exist and can be added to the PS script if desired.
- Deprecated simulators duplicated in supervised_trainer; new torch_dynamics module is the canonical place.

## Recommendations (Prioritized)

1) Implement true pre/post‑norm in `RecursiveReasoningBlock`.
2) Make error embedding dimension dynamic (use `state_dim`).
3) Integrate or remove `ErrorEncoder` for consistency.
4) Wire optional halting (use `AdaptiveRecursiveControl` or a value-based stopping heuristic).
5) Consider disabling MHA for single-token latent or switch to a short sequence latent.
6) Optionally inject `z_initial` into the z_H update.
7) Expand process supervision to rocket landing by adding it to `create_dynamics_function()` in `scripts/train_trc_process_supervision.py`.

## Key Code References

- Two-level recursion scaffold: src/models/recursive_reasoning.py:406, 496, 505–518, 521–531
- Reasoning block and norm-position bug: src/models/recursive_reasoning.py:107–126
- Error embedding hard-coded to 2 dims: src/models/recursive_reasoning.py:198, 441
- ErrorEncoder unused in forward: src/models/tiny_recursive_control.py:100
- Halting module (unused): src/models/recursive_reasoning.py:248
- Process supervision (loss and value head): src/training/process_supervision.py:117, 171; src/models/value_predictor.py:15
- Differentiable dynamics: src/environments/torch_dynamics.py:52 (DI), 111 (VDP), 180 (Rocket)

## Next Steps

- If desired, I can open a small follow-up patch implementing: (1) true pre/post‑norm behavior, (2) dynamic error embedding, and (3) pendulum references cleanup already included here.

