![AISteer360](https://github.com/IBM/AISteer360/raw/main/docs/assets/logo_wide_darkmode.png#gh-dark-mode-only)
![AISteer360](https://github.com/IBM/AISteer360/raw/main/docs/assets/logo_wide_lightmode.png#gh-light-mode-only)

[//]: # (to add: arxiv; pypi; ci)
[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://ibm.github.io/AISteer360/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)
[![GitHub License](https://img.shields.io/github/license/IBM/AISteer360)](https://github.com/IBM/AISteer360/blob/main/LICENSE)

---

The AI Steerability 360 toolkit is an open source Python package for steering large language models.

The toolkit enables the development and evaluation of a wide range of steering methods through an expressive library of
reusable components across four model control surfaces (input, structure, state, and output). Features include modular abstractions for the
construction of steering methods, functionality for composition of steering methods into [steering pipelines](docs/concepts/steering_pipelines.md),
and benchmarking of pipelines on custom use cases and metrics (including measurement of steering side effects).

To get started, please see the documentation at <https://ibm.github.io/AISteer360/> and the [example notebooks](examples/index.md).

## Installation

The toolkit uses [uv](https://docs.astral.sh/uv/) as the package manager (Python 3.11+). After installing `uv` and cloning the repo,
install the toolkit by running:

```commandline
uv venv --python 3.11 && uv pip install .
```

By default, pipelines load and run the model *in process* (via Hugging Face `transformers`). The toolkit additionally provides
support for inference through vLLM (either offline engine or server) via [vLLM-Hook](https://github.com/IBM/vLLM-Hook). To enable this,
install the extra with `uv pip install ".[vllm]"`.

## Contributing

We welcome contributions, particularly new steering methods (controls), use cases, and metrics, along with bug reports,
documentation improvements, and new features. See the [contribution guidelines](CONTRIBUTING.md) and the tutorials on
[adding a steering method](./docs/tutorials/add_new_steering_method.md),
[adding a use case](./docs/tutorials/add_new_use_case.md), and
[adding a metric](./docs/tutorials/add_new_metric.md).

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

The AI Steerability 360 toolkit has been brought to you by IBM.
