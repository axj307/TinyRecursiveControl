# Multi-Problem Refactoring Setup Complete ✅

## What Was Done

1. ✅ **Committed current codebase** (commit: `e851390`)
2. ✅ **Created new branch** `refactor/multi-problem`
3. ✅ **Set up git worktree** at `/orcd/home/002/amitjain/project/TinyRecursiveControl-refactor`
4. ✅ **Created detailed TODO document** in worktree

## Directory Structure

```
Main repo:     /orcd/home/002/amitjain/project/TinyRecursiveControl (branch: main)
Worktree:      /orcd/home/002/amitjain/project/TinyRecursiveControl-refactor (branch: refactor/multi-problem)
```

## Next Steps for You

### 1. Navigate to Worktree

```bash
cd /orcd/home/002/amitjain/project/TinyRecursiveControl-refactor
```

### 2. Read the Detailed TODO

```bash
cat REFACTORING_TODO.md
# OR
less REFACTORING_TODO.md
```

This file contains:
- Complete implementation details for each file
- Code templates ready to copy/paste
- Architecture explanations
- Step-by-step instructions
- Quick reference checklists

### 3. Start Implementing

Follow the phases in order:

**Phase 1: Environment Abstraction** (5 files)
- `src/environments/base.py`
- `src/environments/double_integrator.py`
- `src/environments/pendulum.py`
- `src/environments/__init__.py`
- `src/environments/metadata.py`

**Phase 2: Configuration System** (4 files)
- `configs/problems/double_integrator.yaml`
- `configs/problems/pendulum.yaml`
- `configs/training/default.yaml`
- `src/config/loader.py`

**Phase 3: Data Generation** (2 files)
- Update `src/data/lqr_generator.py`
- Create `scripts/generate_dataset.py`

**Phase 4: Training & Evaluation** (2 files)
- Update `scripts/train_trc.py`
- Update `src/evaluation/evaluator.py`

**Phase 5: SLURM Scripts** (2 files)
- `slurm/double_integrator_pipeline.sbatch`
- `slurm/pendulum_pipeline.sbatch`

**Phase 6: Documentation & Testing**
- `docs/ADDING_NEW_PROBLEMS.md`
- Update `README.md`
- Test both problems
- Merge back to main

### 4. Testing as You Go

After each phase, test imports:

```bash
# Test Phase 1
python -c "from src.environments import get_problem; print(get_problem('double_integrator'))"

# Test Phase 2
python -c "from src.config import get_config; print(get_config('double_integrator'))"

# etc.
```

### 5. Commit Frequently

```bash
git add src/environments/
git commit -m "Phase 1: Add environment abstraction layer"

git add configs/
git commit -m "Phase 2: Add configuration system"

# etc.
```

### 6. When Done, Merge Back

```bash
cd /orcd/home/002/amitjain/project/TinyRecursiveControl
git merge refactor/multi-problem
git worktree remove ../TinyRecursiveControl-refactor
```

## Key Design Decisions

1. **Separate SLURM scripts per environment** (as requested)
   - `double_integrator_pipeline.sbatch`
   - `pendulum_pipeline.sbatch`
   - Easy to customize per problem

2. **Simple registry pattern** (not over-engineered)
   - Just a dict in `__init__.py`
   - Easy to understand and extend

3. **YAML configs** (inspired by llm-control + episodic-transformer-memory-ppo)
   - Hierarchical: problem + training
   - Clean separation of concerns

4. **Abstract base class** (inspired by llm-control)
   - Explicit interface requirements
   - Type-safe
   - Paper-friendly

5. **TRM-inspired metadata** (inspired by TinyRecursiveModels)
   - Unified dataset metadata schema
   - Tracks all relevant info

## Architecture Overview

```
Environment Registry → Problem Instance → Config → Data Generation
                                             ↓
                                       Training Script
                                             ↓
                                    TRC Model (unchanged!)
                                             ↓
                                        Evaluation
```

## Getting Help

- **Detailed TODO**: `/orcd/home/002/amitjain/project/TinyRecursiveControl-refactor/REFACTORING_TODO.md`
- **Reference codebases**:
  - `/home/amitjain/project/Unsloth/Language_Reasoning/llm-control`
  - `/home/amitjain/project/Transformer_RL/episodic-transformer-memory-ppo`
  - `TinyRecursiveModels/` (in your repo)

## Quick Start

```bash
# 1. Navigate to worktree
cd /orcd/home/002/amitjain/project/TinyRecursiveControl-refactor

# 2. Open TODO
cat REFACTORING_TODO.md

# 3. Start with Phase 1
mkdir -p src/environments
touch src/environments/base.py

# 4. Copy templates from REFACTORING_TODO.md and implement!
```

---

**Happy coding! The detailed TODO has everything you need to implement this independently. 🚀**

---

## Original Codebase

Your original working codebase is safe at:
- `/orcd/home/002/amitjain/project/TinyRecursiveControl`
- Branch: `main`
- Commit: `e851390`

You can always return to it if needed, and the worktree won't affect it.
