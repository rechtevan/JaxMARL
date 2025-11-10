# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Repository Context

**Fork Owner**: rechtevan (not upstream FLAIROx) **Purpose**: Code examination,
bug fixes, enhancements, security improvements, test development, and coverage
improvements **GitHub Issues**: Create issues in rechtevan's repository, not
upstream

**Important for Fork Contributors**: If you fork this repository to your own
account, set your fork as the default repository for GitHub CLI to avoid
creating issues in the upstream repository:

```bash
# Set YOUR fork as default (replace YOUR_USERNAME)
cd /path/to/JaxMARL
gh repo set-default YOUR_USERNAME/JaxMARL

# Verify it's set correctly
gh repo view --json nameWithOwner
# Should show: "nameWithOwner": "YOUR_USERNAME/JaxMARL"
```

This ensures `gh issue create` and other commands work with YOUR fork, not
upstream.

## Local Development Conventions

**`.local/` Directory**: Used for AI-generated analysis, scripts, reports, and
other files that should NOT be committed to git. This directory is in
`.gitignore`.

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

JaxMARL is a Multi-Agent Reinforcement Learning (MARL) library in JAX that
combines ease-of-use with GPU-enabled efficiency. It provides JAX-native
implementations of MARL environments and baseline algorithms, enabling thorough
evaluation of MARL methods with end-to-end JIT compilation.

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

**Run tests with coverage:**

```bash
# Generate coverage report with terminal output
pytest --cov=jaxmarl --cov-report=term ./tests/

# Generate coverage with HTML report
pytest --cov=jaxmarl --cov-report=html ./tests/

# Generate coverage with JSON output
pytest --cov=jaxmarl --cov-report=json ./tests/

# Generate all report types
pytest --cov=jaxmarl --cov-report=term --cov-report=html --cov-report=json ./tests/
```

**Coverage reports location:**

- HTML reports: `.local/htmlcov/index.html`
- JSON reports: `.local/coverage.json`
- Configuration: `.coveragerc`

**Coverage Scope (What's Measured)**:

Coverage targets (80-90%+) apply to **core production code only**:

- ✅ Environment implementations (`jaxmarl/environments/*/`)
- ✅ Wrappers (`jaxmarl/wrappers/`)
- ✅ Registration system (`jaxmarl/registration.py`)
- ✅ Multi-agent base classes (`jaxmarl/environments/multi_agent_env.py`)

**Excluded from coverage requirements** (per `.coveragerc`):

- ❌ Visualization utilities (`*/viz/*`, `*visualizer.py`, `*_viz.py`)
- ❌ Interactive modules (`*/interactive.py`, `*/manual_game*.py`)
- ❌ Pretrained models (`*/pretrained/*`)
- ❌ Experimental code (`*/gridworld/*`)
- ❌ Test files (`*/tests/*`, `*_test.py`)
- ❌ Debug/development code (`if __name__ == "__main__"`)

**Rationale**: Coverage focuses on core functionality that's used in production
(environments, algorithms). Visualization, interactive tools, and experimental
code are tested manually and don't need automated coverage tracking.

## Code Quality and Linting

### Code Documentation Standards

**Python uses docstrings, not Doxygen** (Doxygen is for C/C++).

**Recommended Style: Google-style docstrings** (common in ML/JAX projects):

```python
def step(self, key: chex.PRNGKey, state: State, actions: dict) -> tuple:
    """Execute one environment step with given actions.

    Args:
        key: JAX random key for stochastic operations
        state: Current environment state
        actions: Dictionary mapping agent names to actions

    Returns:
        tuple: (observations, state, rewards, dones, infos)

    Raises:
        ValueError: If actions dict doesn't match agent names
    """
```

**Alternative: NumPy-style** (scientific computing standard):

```python
def step(self, key, state, actions):
    """
    Execute one environment step.

    Parameters
    ----------
    key : chex.PRNGKey
        JAX random key for stochastic operations
    state : State
        Current environment state
    actions : dict
        Agent names mapped to actions

    Returns
    -------
    observations : dict
        Agent observations after step
    state : State
        Updated environment state
    """
```

**Documentation Tools**:

```bash
# Audit docstring coverage
pip install interrogate
interrogate jaxmarl/ --verbose

# Check docstring style
pip install pydocstyle
pydocstyle jaxmarl/

# Generate API docs (optional)
pip install sphinx sphinx-rtd-theme
sphinx-quickstart docs/
sphinx-build -b html docs/ docs/_build/
```

**Documentation Requirements**:

- All public environment APIs must have docstrings
- All `__init__`, `reset()`, `step()` methods documented
- State dataclasses documented with field descriptions
- Module-level docstrings for each file
- Type hints in function signatures

**Status**: See issue #35 for documentation improvement plan

### Python Code

**Static analysis tools** (configured in `pyproject.toml`):

- **Ruff** - Fast Python linter and formatter (MIT license)
  - Combines functionality of multiple tools (flake8, isort, pyupgrade, etc.)
  - Configured for JAX/ML code patterns
  - Line length: 88 characters
  - Auto-fix capabilities for many issues
- **MyPy** - Static type checking (MIT license)
  - Python 3.10+ type checking
  - Configured to ignore missing imports for ML libraries
  - Gradual typing approach (not strict mode)
- **Pre-commit hooks** - Automatic checks before commit (MIT license)
  - Runs Ruff and MyPy automatically
  - Configuration in `.pre-commit-config.yaml`

**Run Python checks locally**:

```bash
# Install dev dependencies
pip install -e .[dev]

# Run Ruff linter
ruff check .

# Run Ruff linter with auto-fix
ruff check --fix .

# Run Ruff formatter (check only)
ruff format --check .

# Run Ruff formatter (auto-fix)
ruff format .

# Run MyPy type checking
mypy jaxmarl baselines

# Run all checks via pre-commit
pre-commit run --all-files
```

**Using Pre-commit Hooks**:

Pre-commit hooks automatically run quality checks before each commit, ensuring
code quality and consistency.

```bash
# One-time setup: Install pre-commit hooks
pip install -e .[dev]
pre-commit install

# Now hooks run automatically on 'git commit'
# To skip hooks (not recommended): git commit --no-verify

# Run hooks manually on all files
pre-commit run --all-files

# Run specific hook on all files
pre-commit run ruff --all-files
pre-commit run mypy --all-files
pre-commit run mdformat --all-files

# Run hooks on staged files only
pre-commit run

# Update hook versions
pre-commit autoupdate

# Uninstall hooks (removes from .git/hooks)
pre-commit uninstall
```

**Pre-commit Hook Features**:

- **Ruff** - Lints and formats Python code, auto-fixes most issues
- **Ruff-format** - Formats Python code to consistent style
- **MyPy** - Type checks main library code (excludes tests/baselines)
- **mdformat** - Formats markdown files with GFM support
- **markdownlint-cli2** - Lints markdown for style and correctness
- **Standard hooks** - Trailing whitespace, EOF fixer, YAML/TOML validation

Configuration files:

- `.pre-commit-config.yaml` - Hook definitions and versions
- `pyproject.toml` - Tool configurations (ruff, mypy, mdformat)
- `.mdformat.toml` - Markdown formatting options
- `.markdownlint-cli2.yaml` - Markdown linting rules

### Markdown Documentation

**Markdown linting and formatting** (configured in `pyproject.toml` and
`.pymarkdown.json`):

- **mdformat** - Auto-formatter for markdown files (MIT license)

  - Supports GitHub Flavored Markdown (GFM)
  - Auto-fixes formatting issues
  - Integrates with pre-commit hooks
  - Line length: 88 characters

- **pymarkdownlnt** - Markdown linter (MIT license)

  - 46 built-in rules for markdown quality
  - GFM compliant
  - Catches syntax errors and inconsistencies

**Files checked**:

- README.md, CLAUDE.md, CONTRIBUTING.md, NOTICE
- All .md files in repository
- Ensures consistent documentation style

**Run markdown checks locally**:

```bash
# Install dev dependencies (includes markdown tools)
pip install -e .[dev]

# Format markdown files (auto-fix)
mdformat .

# Check markdown formatting (no changes)
mdformat --check .

# Lint markdown files
pymarkdown scan .
```

**Configuration files**:

- `pyproject.toml` - Contains [tool.ruff], [tool.mypy], and [tool.mdformat]
  sections
- `.pymarkdown.json` - PyMarkdown linter rules and settings
- `.pre-commit-config.yaml` - Pre-commit hook configuration

## CI/CD and Security

### Automated Workflows

The repository uses GitHub Actions for continuous integration and security
scanning:

**CodeQL Security Scanning** (GitHub Default):

- Automated security vulnerability detection for Python code
- Runs automatically on push to main
- Results viewable in GitHub Security tab → Code scanning alerts
- Uses GitHub's default security query suite

**Static Analysis** (`.github/workflows/static-analysis.yml`):

- **Ruff Linting**: Checks Python code for style issues and potential bugs
- **Ruff Formatting**: Verifies code formatting consistency
- **MyPy Type Checking**: Validates type hints and catches type errors
- **Markdown Formatting**: Ensures consistent markdown file formatting
- **Markdown Linting**: Checks markdown files for syntax and style issues
- All jobs run in parallel for fast feedback
- Runs on: push to main, pull requests

**Test Coverage** (`.github/workflows/coverage.yml`):

- Runs pytest with coverage reporting
- Uploads results to Codecov
- Runs on: push to main, pull requests

**Docker Tests** (`.github/workflows/docker-tests.yml`):

- Validates tests pass in Docker environment
- Ensures reproducible test execution
- Runs on: all pushes and pull requests

**Viewing Security Results:**

```bash
# CodeQL findings appear in:
# - GitHub Security tab → Code scanning alerts
# - Pull request checks (if issues found)
# - Automatic scans on push to main
```

**Security Best Practices:**

- All dependencies must be Apache 2.0 compatible (see Licensing section)
- CodeQL scans for common Python security issues:
  - SQL injection vulnerabilities
  - Command injection risks
  - Path traversal issues
  - Unsafe deserialization
  - And more...

## Running Baselines

All baseline algorithms use Hydra for configuration management. Config files are
located in `baselines/<ALGORITHM>/config/`.

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

**Enable wandb logging:** Edit the config file (e.g.,
`baselines/IPPO/config/ippo_rnn_smax.yaml`) and set wandb parameters.

## Architecture

### Core Environment API

JaxMARL follows a hybrid PettingZoo/Gymnax-inspired API with JAX-first design:

- **Environment registration:** `jaxmarl/registration.py` - Central registry
  using `make(env_id, **kwargs)`
- **Base class:** `jaxmarl/environments/multi_agent_env.py` - `MultiAgentEnv`
  abstract class
- **Key methods:**
  - `reset(key)` → `(obs_dict, state)`
  - `step(key, state, actions_dict)` →
    `(obs_dict, state, rewards_dict, dones_dict, infos_dict)`
  - All methods are `@jax.jit` decorated for performance

**Important conventions:**

- Actions, observations, rewards, and dones are dictionaries keyed by agent name
- Done dictionary includes special `"__all__"` key indicating episode
  termination
- Parallel structure: all agents act at each timestep (async games use dummy
  actions)
- Auto-reset on episode end (controllable via `reset_state` parameter)

### Environment Structure

All environments are in `jaxmarl/environments/`:

- `mpe/` - Multi-Agent Particle Environments (communication-oriented tasks)
- `smax/` - Simplified Multi-Agent Challenge (StarCraft-like without game
  engine)
- `overcooked/` and `overcooked_v2/` - Cooperative cooking tasks
- `mabrax/` - Multi-agent continuous control (Brax-based)
- `hanabi/` - Cooperative card game (partially observable)
- `storm/` - Spatial-temporal matrix games
- `jaxnav/` - 2D navigation for differential drive robots
- `coin_game/` - Social dilemma grid world
- `switch_riddle/` - Simple communication game (debugging)

### Baseline Algorithms

Located in `baselines/`, following CleanRL's single-file implementation
philosophy:

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
3. If porting from existing implementation, add tests showing transition
   correspondence
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

**MPE:** `MPE_simple_v3`, `MPE_simple_tag_v3`, `MPE_simple_world_comm_v3`,
`MPE_simple_spread_v3`, `MPE_simple_crypto_v3`,
`MPE_simple_speaker_listener_v4`, `MPE_simple_push_v3`,
`MPE_simple_adversary_v3`, `MPE_simple_reference_v3`, `MPE_simple_facmac_v1`
(and 3a, 6a, 9a variants)

**SMAX:** `SMAX`, `HeuristicEnemySMAX`, `LearnedPolicyEnemySMAX`

**MABrax:** `ant_4x2`, `halfcheetah_6x1`, `hopper_3x1`, `humanoid_9|8`,
`walker2d_2x3`

**STORM:** `storm`, `storm_2p`, `storm_np`

**Others:** `hanabi`, `overcooked`, `overcooked_v2`, `coin_game`, `jaxnav`,
`switch_riddle`

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

- Baselines are single-file implementations (no shared modules between
  algorithms)
- Environment-specific hyperparameters go in Hydra config files
- Use `functools.partial` with `jax.jit` for static arguments
- PRNG key splitting is explicit everywhere
- Network architectures defined inline with Flax Linen

## Licensing

**Project License**: Apache License 2.0 **Fork Copyright**: Booz Allen Hamilton
(for contributions to rechtevan/JaxMARL) **Original Copyright**: 2023 FLAIR
(upstream FLAIROx/JaxMARL)

### License Requirements

**IMPORTANT**: All code and contributions to this fork (rechtevan/JaxMARL) must:

- Be licensed under **Apache License 2.0** (same as project)
- Acknowledge **Booz Allen Hamilton copyright** for contributions to this fork
- Be compatible with Apache License 2.0 terms

**Specific requirements:**

- **All code written**: Source code, tests, documentation → Apache 2.0 license,
  Booz Allen Hamilton copyright
- **Runtime dependencies**: Must use Apache 2.0, MIT, BSD, or similar permissive
  licenses
- **Dev dependencies**: Can use any OSI-approved license (pytest, ruff, mypy,
  etc.)
- **Avoid**: GPL, LGPL, AGPL for runtime dependencies (copyleft incompatible)

### When Suggesting Dependencies

Before recommending a new dependency, verify:

1. **Check license**: Use MIT, Apache 2.0, or BSD licensed packages
2. **Runtime vs dev**: Stricter requirements for runtime dependencies
3. **Document in pyproject.toml**: Add appropriate license classifiers

### Approved Tool Licenses (Current)

All currently recommended quality improvement tools are Apache 2.0 compatible:

| Tool       | License    | Type | Status |
| ---------- | ---------- | ---- | ------ |
| pytest-cov | MIT        | Dev  | ✅     |
| Ruff       | MIT        | Dev  | ✅     |
| MyPy       | MIT        | Dev  | ✅     |
| pre-commit | MIT        | Dev  | ✅     |
| CodeQL     | GitHub ToS | CI   | ✅     |
| Codecov    | Apache 2.0 | CI   | ✅     |

### License Headers and Copyright Attribution

**POLICY: Industry Standard Approach (Option A)**

This repository follows industry standard practices for multi-company open
source contributions:

✅ **NOTICE file** - Lists major contributors (FLAIR, Booz Allen Hamilton) ✅
**Git history** - Provides detailed attribution (use corporate email) ✅
**LICENSE file** - Apache 2.0 covers all code ❌ **NO copyright headers** - Not
required, matches existing codebase

This is the same approach used by Kubernetes, TensorFlow, Linux Kernel, and
hundreds of other multi-company Apache 2.0 projects.

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
