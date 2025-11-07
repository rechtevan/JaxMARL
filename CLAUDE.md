# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Context

**Fork Owner**: rechtevan (not upstream FLAIROx)
**Purpose**: Code examination, bug fixes, enhancements, security improvements, test development, and coverage improvements
**GitHub Issues**: Create issues in rechtevan's repository, not upstream

## Local Development Conventions

**`.local/` Directory**: Used for AI-generated analysis, scripts, reports, and other files that should NOT be committed to git. This directory is in `.gitignore`.

Use `.local/` for:
- Code analysis reports
- Test coverage reports and summaries
- Security scan results
- Performance benchmarking results
- Build logs for review
- Temporary scripts, diagrams, and visualizations
- Any AI-generated content not part of the codebase
- Experimental code snippets and prototypes

**Examples:**
- `.local/coverage-report.html` - Coverage analysis
- `.local/security-scan.txt` - Security findings
- `.local/performance-analysis.md` - Performance metrics
- `.local/test-plan.md` - Test strategy documents

## Project Overview

JaxMARL is a Multi-Agent Reinforcement Learning (MARL) library in JAX that combines ease-of-use with GPU-enabled efficiency. It provides JAX-native implementations of MARL environments and baseline algorithms, enabling thorough evaluation of MARL methods with end-to-end JIT compilation.

## Installation & Setup

**Environment-only installation:**
```bash
pip install jaxmarl
```

**Development installation (for running algorithms):**
```bash
git clone https://github.com/FLAIROx/JaxMARL.git && cd JaxMARL
pip install -e .[algs]
export PYTHONPATH=./JaxMARL:$PYTHONPATH
```

**Development with testing:**
```bash
pip install -e .[dev]
```

**Docker (recommended for fastest start):**
```bash
make build  # Build the Docker container
make run    # Run the container interactively
```

## Testing

**Run all tests:**
```bash
pytest ./tests/
```

**Run tests in Docker:**
```bash
make test
```

**Run specific environment tests:**
```bash
pytest tests/mpe/
pytest tests/smax/
pytest tests/overcooked/
```

## Running Baselines

All baseline algorithms use Hydra for configuration management. Config files are located in `baselines/<ALGORITHM>/config/`.

**Run IPPO:**
```bash
python baselines/IPPO/ippo_rnn_smax.py
python baselines/IPPO/ippo_ff_mpe.py
```

**Run MAPPO:**
```bash
python baselines/MAPPO/mappo_rnn_smax.py
python baselines/MAPPO/mappo_ff_hanabi.py
```

**Run Q-Learning variants (IQL, VDN, QMIX, etc.):**
```bash
python baselines/QLearning/qmix_rnn.py
python baselines/QLearning/vdn_ff.py
python baselines/QLearning/shaq.py
```

**Enable wandb logging:**
Edit the config file (e.g., `baselines/IPPO/config/ippo_rnn_smax.yaml`) and set wandb parameters.

## Architecture

### Core Environment API

JaxMARL follows a hybrid PettingZoo/Gymnax-inspired API with JAX-first design:

- **Environment registration:** `jaxmarl/registration.py` - Central registry using `make(env_id, **kwargs)`
- **Base class:** `jaxmarl/environments/multi_agent_env.py` - `MultiAgentEnv` abstract class
- **Key methods:**
  - `reset(key)` → `(obs_dict, state)`
  - `step(key, state, actions_dict)` → `(obs_dict, state, rewards_dict, dones_dict, infos_dict)`
  - All methods are `@jax.jit` decorated for performance

**Important conventions:**
- Actions, observations, rewards, and dones are dictionaries keyed by agent name
- Done dictionary includes special `"__all__"` key indicating episode termination
- Parallel structure: all agents act at each timestep (async games use dummy actions)
- Auto-reset on episode end (controllable via `reset_state` parameter)

### Environment Structure

All environments are in `jaxmarl/environments/`:
- `mpe/` - Multi-Agent Particle Environments (communication-oriented tasks)
- `smax/` - Simplified Multi-Agent Challenge (StarCraft-like without game engine)
- `overcooked/` and `overcooked_v2/` - Cooperative cooking tasks
- `mabrax/` - Multi-agent continuous control (Brax-based)
- `hanabi/` - Cooperative card game (partially observable)
- `storm/` - Spatial-temporal matrix games
- `jaxnav/` - 2D navigation for differential drive robots
- `coin_game/` - Social dilemma grid world
- `switch_riddle/` - Simple communication game (debugging)

### Baseline Algorithms

Located in `baselines/`, following CleanRL's single-file implementation philosophy:

**IPPO (Independent PPO):**
- Shared parameters between agents
- Single network architecture (FF or RNN variants)
- Files: `baselines/IPPO/ippo_{rnn|ff|cnn}_{env}.py`

**MAPPO (Multi-Agent PPO):**
- Centralized value function
- Files: `baselines/MAPPO/mappo_{rnn|ff}_*.py`

**Q-Learning Family:**
- IQL, VDN, QMIX, TransfQMIX, SHAQ, PQN-VDN
- All in `baselines/QLearning/`
- Uses flashbax for replay buffers

**Common patterns in baselines:**
- Hydra config management (one config per environment/algorithm combo)
- Uses `ScannedRNN` for recurrent architectures (GRU-based)
- Training state managed via Flax `TrainState`
- Vectorized environments with `jax.vmap`

### Wrappers

Located in `jaxmarl/wrappers/`:
- `baselines.py` - Logging wrappers (SMAXLogWrapper, MPELogWrapper, etc.)
- `gymnax.py` - Gymnax compatibility
- `transformers.py` - State/observation transformations

### Key JAX Patterns

- **Full JIT compilation:** All environment methods are jitted
- **Functional state:** Environment state is explicit (no hidden state)
- **Pure functions:** All randomness via explicit PRNG keys
- **Vectorization:** Heavy use of `vmap` for parallel agent processing
- **Scan for RNNs:** `nn.scan` decorator for efficient recurrent layers

## Development Guidelines

### Adding a New Environment

Requirements from CONTRIBUTING.md:
1. Implement `MultiAgentEnv` interface
2. Unit tests demonstrating correctness (pytest format)
3. If porting from existing implementation, add tests showing transition correspondence
4. Training results for IPPO and MAPPO over 20 seeds
5. Config files saved to `baselines/`
6. README explaining the environment

### Adding a New Algorithm

Requirements from CONTRIBUTING.md:
1. Single-file implementation following CleanRL philosophy
2. Hydra config file in `baselines/<ALGORITHM>/config/`
3. Performance results on ≥3 environments with ≥20 seeds per result
4. Compare to existing implementations if applicable
5. README with implementation details and usage

### Testing Requirements

- Use pytest format
- Place tests in `tests/<environment_name>/`
- Test environment API compliance
- Test correspondence to reference implementations where applicable
- All tests must pass in Docker CI (see `.github/workflows/`)

## Environment IDs

Use these strings with `make(env_id)`:

**MPE:** `MPE_simple_v3`, `MPE_simple_tag_v3`, `MPE_simple_world_comm_v3`, `MPE_simple_spread_v3`, `MPE_simple_crypto_v3`, `MPE_simple_speaker_listener_v4`, `MPE_simple_push_v3`, `MPE_simple_adversary_v3`, `MPE_simple_reference_v3`, `MPE_simple_facmac_v1` (and 3a, 6a, 9a variants)

**SMAX:** `SMAX`, `HeuristicEnemySMAX`, `LearnedPolicyEnemySMAX`

**MABrax:** `ant_4x2`, `halfcheetah_6x1`, `hopper_3x1`, `humanoid_9|8`, `walker2d_2x3`

**STORM:** `storm`, `storm_2p`, `storm_np`

**Others:** `hanabi`, `overcooked`, `overcooked_v2`, `coin_game`, `jaxnav`, `switch_riddle`

## Dependencies

- JAX ≤0.4.38 (ensure correct JAX installation for your accelerator)
- Flax (neural networks)
- Optax (optimizers)
- Distrax (distributions)
- Flashbax 0.1.0 (replay buffers)
- Hydra ≥1.3.2 (config management)
- Brax 0.10.3 (for MABrax environments)
- Python ≥3.10

## Code Style Notes

- Baselines are single-file implementations (no shared modules between algorithms)
- Environment-specific hyperparameters go in Hydra config files
- Use `functools.partial` with `jax.jit` for static arguments
- PRNG key splitting is explicit everywhere
- Network architectures defined inline with Flax Linen

## Licensing

**Project License**: Apache License 2.0
**Fork Copyright**: Booz Allen Hamilton (for contributions to rechtevan/JaxMARL)
**Original Copyright**: 2023 FLAIR (upstream FLAIROx/JaxMARL)

### License Requirements

**IMPORTANT**: All code and contributions to this fork (rechtevan/JaxMARL) must:

- Be licensed under **Apache License 2.0** (same as project)
- Acknowledge **Booz Allen Hamilton copyright** for contributions to this fork
- Be compatible with Apache License 2.0 terms

**Specific requirements:**
- **All code written**: Source code, tests, documentation → Apache 2.0 license, Booz Allen Hamilton copyright
- **Runtime dependencies**: Must use Apache 2.0, MIT, BSD, or similar permissive licenses
- **Dev dependencies**: Can use any OSI-approved license (pytest, ruff, mypy, etc.)
- **Avoid**: GPL, LGPL, AGPL for runtime dependencies (copyleft incompatible)

### When Suggesting Dependencies

Before recommending a new dependency, verify:
1. **Check license**: Use MIT, Apache 2.0, or BSD licensed packages
2. **Runtime vs dev**: Stricter requirements for runtime dependencies
3. **Document in pyproject.toml**: Add appropriate license classifiers

### Approved Tool Licenses (Current)

All currently recommended quality improvement tools are Apache 2.0 compatible:

| Tool | License | Type | Status |
|------|---------|------|--------|
| pytest-cov | MIT | Dev | ✅ |
| Ruff | MIT | Dev | ✅ |
| MyPy | MIT | Dev | ✅ |
| pre-commit | MIT | Dev | ✅ |
| CodeQL | GitHub ToS | CI | ✅ |
| Codecov | Apache 2.0 | CI | ✅ |

### License Headers and Copyright Attribution

**POLICY: Industry Standard Approach (Option A)**

This repository follows industry standard practices for multi-company open source contributions:

✅ **NOTICE file** - Lists major contributors (FLAIR, Booz Allen Hamilton)
✅ **Git history** - Provides detailed attribution (use corporate email)
✅ **LICENSE file** - Apache 2.0 covers all code
❌ **NO copyright headers** - Not required, matches existing codebase

This is the same approach used by Kubernetes, TensorFlow, Linux Kernel, and hundreds of other multi-company Apache 2.0 projects.

#### General Rule: DO NOT ADD HEADERS

**Default approach for all work:**
- **Do NOT add copyright headers** to existing files when modifying them
- **Do NOT add copyright headers** to new files
- **Use git commits** with Booz Allen Hamilton email for attribution
- The LICENSE + NOTICE files provide legal coverage
- Git history provides detailed attribution record
- This maintains consistency with industry standard practice

#### Exception: When Headers Are Required

Only add copyright headers if:
1. Creating substantial new components (new algorithms, new environments)
2. Legal/compliance requires it for your organization
3. Upstream files already have headers (preserve and append)

**For NEW substantial files** (if headers required):
```python
# Copyright [YEAR] Booz Allen Hamilton
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

**For files WITH existing headers** (preserve and append):
```python
# Copyright 2023 FLAIR
# Copyright 2025 Booz Allen Hamilton
#
# Licensed under the Apache License, Version 2.0...
```

**For files WITHOUT existing headers** (most files):
- **Do NOT add headers** - just modify the code
- Git history will show: `git log --follow <file>`
- LICENSE file provides legal coverage

#### Copyright Attribution via Git

To verify contributors to any file:
```bash
# See all contributors to a file
git log --format="%an <%ae>" <file> | sort | uniq -c | sort -rn

# See detailed history
git log --follow <file>
```

**Key Points:**
- **LICENSE file** (Apache 2.0) covers ALL code in the repository
- **Git history** provides complete attribution record
- **No headers** is the norm for this codebase - keep it that way
- **Only add headers** when legally required or for substantial new work
- **Never remove** existing headers if present
