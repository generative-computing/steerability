# Installation

The toolkit uses [uv](https://docs.astral.sh/uv/) as the package manager (Python 3.12+). For Mac/Linux, `uv` is installed via:

=== "standalone installer"
    ```bash
    curl -Ls https://astral.sh/uv/install.sh | sh
    ```

=== "Homebrew"
    ```bash
    brew install astral-sh/uv/uv
    ```

For Windows, `uv` can be installed (using PowerShell 7+) via:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

See the uv page for details and other installation options.


## Installing the toolkit

Once `uv` is installed, install the `steerability` package via:

```commandline
uv venv --python 3.12 && uv pip install .
```

The above creates a `.venv` (if missing), installs `steerability` (in non-editable mode), and installs all dependencies
listed under `[project.dependencies]` in the `pyproject.toml` file. Activate the environment by running `source .venv/bin/activate`.
Note that on Windows, you may need to split the installation script into two separate commands (instead of chained via `&&`).
To install an optional extra from `[project.optional-dependencies]`, e.g., `eval`, append it in quotes and
square brackets to the `install` command as follows:

```commandline
uv venv --python 3.12 && uv pip install '.[eval]'
```

By default, pipelines load and run the model in process (via Hugging Face `transformers`). The optional extras are
grouped in three tiers:

- Backends: `vllm`, the vLLM execution backends (offline engine or server) plus the `vllm_hook_plugins` core. This
  extra pulls in `trl[vllm]` such that the resolved vLLM version stays inside TRL's supported range.
- Workflows: `eval`, the Inspect AI evaluation stack and the plotting utilities in `evaluation/plotting.py`.
- Method-specific: `merging`, the MergeKit structural control, isolated because MergeKit pins an older pydantic than
  Inspect requires.

Constrained decoding on the Hugging Face backend uses xgrammar, which is a core dependency; on vLLM backends the
constraint lowers to native structured outputs.

The umbrella `all` extra currently installs `eval`; it is the stable name for every extra that can share one
environment. Install `merging` and `vllm` by name, e.g., `uv pip install '.[vllm]'`. Note that `merging` cannot share
an environment with `eval`; `pyproject.toml` declares this as a `[tool.uv]` conflict.

Contributors install with `uv sync --extra all`, which creates the environment, installs the toolkit in editable mode,
and adds the `dev` dependency group (pytest, pre-commit, notebook tooling). Add `--group docs` to build the
documentation site.

The vLLM boot environment (applied by the offline engine, and returned by `serve_environment()` for a server you
launch) defaults the FlashInfer sampler off via `VLLM_USE_FLASHINFER_SAMPLER=0`. This avoids a JIT kernel compile at
boot, which fails on a node whose CUDA toolkit does not match the installed torch build. The native sampler is
greedy-equivalent. To use FlashInfer instead, install its prebuilt kernels for your CUDA version from
`https://flashinfer.ai/whl/cu1XX` (`flashinfer-jit-cache`, and optionally `flashinfer-cubin`) and set
`VLLM_USE_FLASHINFER_SAMPLER=1`.

## Accessing Hugging Face models

Inference is facilitated by Hugging Face. Authenticate once with `hf auth login` (the `huggingface_hub` CLI), or
export `HF_TOKEN=hf_***` in the environment that runs the pipeline.

Some Hugging Face models (e.g. `meta-llama/Meta-Llama-3.1-8B-Instruct`) are behind an access gate. To gain access:

1. Request access on the model's Hub page with the account whose token you use.
2. Wait for approval (you'll receive an email).
3. (Re-)authenticate locally with `hf auth login`.

Once you have completed the above steps, please see our [quickstart](quickstart.md) guide to get up and running!
