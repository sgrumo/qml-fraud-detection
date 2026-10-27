# AGENTS.md

Guidelines for AI agents (and humans) working in this repository. Adapted from
[minimaxir's AGENTS.md](https://gist.github.com/minimaxir/10b780671ee5d695b4369b987413b38f),
with project-specific rules for a quantum machine learning thesis on fraud
detection.

## Project context

- Package `qfd` (`src/qfd/`) compares classical and quantum fraud detectors.
- Three stages, one per subpackage, sharing `data/`, `evaluation/` and the
  device-agnostic `quantum/` core:
  1. `classical/`: classical ML baselines.
  2. `simulation/`: PennyLane simulators (exact, shot-based, noisy).
  3. `hardware/`: AWS Braket, Google Quantum Engine (Cirq), D-Wave Leap.
- Experiments are YAML files in `configs/<stage>/`, run with `qfd run <config>`.
- Background and the benchmarking protocol live in `docs/`; follow
  `docs/06-datasets-and-benchmarking.md` for every experiment.

## Core principles

- Optimise for algorithmic efficiency first, then for reuse. Write the minimum
  code that solves the problem.
- Prefer a small, well-maintained library over custom code (scikit-learn,
  PennyLane, Ocean, polars).
- Clarity and maintainability over cleverness.

## Environment and tools

- **Use conda for environments** (`environment.yml`, env name `qfd`). Do not
  use `uv`, `venv` or global pip installs.
- Declare every Python dependency once, in `pyproject.toml`. Conda installs
  only Python and compiled libraries that need OpenMP (xgboost, lightgbm).
- Hardware SDKs are optional extras: `.[braket]`, `.[google]`, `.[dwave]`.
  Import them lazily inside `qfd.hardware` so the core install works without them.
- Use `polars` instead of `pandas` for tabular I/O.
- Use `orjson` instead of `json`.
- Use `tqdm` for progress bars in notebooks.
- Use `logging` (module-level `logger = logging.getLogger(__name__)`), never
  `print`, for reporting.

## Style

- PEP 8, 4-space indentation, 88-character lines (enforced by Ruff).
- `snake_case` functions and variables, `PascalCase` classes,
  `UPPER_CASE` constants. Scikit-learn style `X` / `K` matrix names are allowed.
- No emoji in code or comments. No comments that restate the code.

## Documentation and typing

- Every public module, class, function and method has a Google-style docstring
  with `Args`, `Returns` and `Raises` where relevant.
- Every signature is fully type-annotated. Avoid `Any`; use the aliases in
  `qfd.types` (`FloatArray`, `IntArray`, `Params`) and protocols.
- `mypy --strict` must pass.

## Error handling

- Never swallow exceptions; no bare `except:`. Catch specific exceptions and
  re-raise with context (`raise ... from exc`).
- Use context managers for files and resources.
- Error messages say what is wrong and how to fix it.

## Function and class design

- One responsibility per function or class.
- At most five parameters; group related settings in a frozen dataclass
  (see `TrainingConfig`, `ExperimentConfig`).
- No mutable default arguments. Return early.
- Prefer composition and protocols (`qfd.types.Detector`,
  `qfd.quantum.kernels.QuantumKernel`) over deep inheritance.

## Testing

- pytest, one test file per module area in `tests/`, Arrange-Act-Assert.
- New code ships with unit tests. Mock external services and hardware.
- Markers: `slow` for long simulations, `hardware` for tests that submit paid
  jobs. `hardware` tests are deselected by default; never run them unasked.
- No commented-out tests.

## Imports and dependencies

- No wildcard imports. Order: standard library, third party, local (Ruff `I`).
- Optional-extra imports go inside functions (with `# noqa: PLC0415`) or behind
  an `ImportError` that explains which extra to install.

## Benchmarking and experiments

- Never run benchmarks in parallel with each other; timing is part of the results.
- Never tune or select on the test period. Fit every transformation on the
  training period only (`qfd.data.build_preprocessor`).
- Compare like with like: classical and quantum models get the same samples,
  the same reduced features and the same tuning budget.
- Report AUPRC at natural prevalence as the primary metric; accuracy is never
  a headline metric.
- Disable caches that make runs depend on each other.
- Do not "fix" disappointing results by changing the protocol after seeing
  test numbers. Record deviations instead.

## Quantum hardware

- Hardware runs cost money. Never submit hardware jobs, run `pytest -m hardware`,
  or pass `--allow-hardware` without explicit approval from the user for that run.
- Keep `max_hardware_circuits` in hardware configs conservative.

## Security

- Secrets live in `.env` (git-ignored); see `.env.example`.
- Read credentials from environment variables only. Never hard-code, print or
  log tokens, keys or URLs containing them.
- Never commit raw data (`data/raw/` is git-ignored).

## Version control

- Clear, imperative commit messages describing why the change was made.
- No commented-out code, debug prints or credentials in commits.
- Before committing: `ruff check`, `ruff format --check`, `mypy` and
  `pytest` pass; new public functions have docstrings and type hints.
