![Steerability](https://github.com/IBM/steerability/raw/main/docs/assets/logo_slim_darkmode.png#gh-dark-mode-only)
![Steerability](https://github.com/IBM/steerability/raw/main/docs/assets/logo_slim_lightmode.png#gh-light-mode-only)

[//]: # (to add: arxiv; pypi; ci)
[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://ibm.github.io/steerability/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue)
[![GitHub License](https://img.shields.io/github/license/IBM/steerability)](https://github.com/IBM/steerability/blob/main/LICENSE)

---

The Steerability toolkit is an open source Python package for steering large language models.

The toolkit enables the development and evaluation of a wide range of steering methods via reusable components across four model control surfaces (input, structure, state, and output). Features include modular abstractions for the construction of steering methods, functionality for composition of steering methods into [steering pipelines](docs/concepts/steering_pipelines.md), and evaluation of steering pipelines on [Inspect](https://inspect.aisi.org.uk) tasks (including measurement of steering side effects).

To get started, please see the documentation at <https://ibm.github.io/steerability/> and the [example notebooks](examples/index.md).

## Installation

The toolkit uses [uv](https://docs.astral.sh/uv/) as the package manager (Python 3.12+). After installing `uv` and cloning the repo,
install the toolkit by running:

```commandline
uv venv --python 3.12 && uv pip install .
```

By default, pipelines load and run the model *in process* (via Hugging Face `transformers`). The toolkit additionally provides
support for inference through vLLM via [vLLM-Hook](https://github.com/IBM/vLLM-Hook). To enable this,
install the extra with `uv pip install ".[vllm]"`. The `eval` extra adds the Inspect AI evaluation stack; `all` installs
every extra that can share one environment.

## Contributing

We welcome contributions, particularly in the form of new steering methods (controls) and evaluation tasks, along with bug reports,
documentation improvements, and new features. See the [contribution guidelines](CONTRIBUTING.md) and the tutorial on
[adding a steering method](./docs/tutorials/add_new_steering_method.md).

## Reference

If you find the toolkit useful in your work, please cite the following:
```bibtex
@article{miehling2026aisteerability360,
  title = {AI Steerability 360: A Toolkit for Steering Large Language Models},
  author = {Miehling, Erik and Ramamurthy, Karthikeyan Natesan and Venkateswaran, Praveen and Ko, Irene and Dognin, Pierre and Singh, Moninder and Pedapati, Tejaswini and Balakrishnan, Avinash and Riemer, Matthew and Wei, Dennis and Vejsbjerg, Inge and Daly, Elizabeth M. and Varshney, Kush R.},
  journal = {arXiv preprint arXiv:2603.07837},
  year = {2026}
}
```

## IBM ❤️ Open Source AI

The Steerability toolkit has been brought to you by IBM.
