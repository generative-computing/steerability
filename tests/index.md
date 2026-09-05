# Tests

Tests evaluate control implementations across a set of control arguments, models (listed in `tests/utils/ci_models.yaml`),
and the available devices on the machine.

The test tree is organized by area:

- `tests/controls/` covers individual steering controls
- `tests/core/` covers the pipeline, registry, and other core functionality
- `tests/internals/` covers the `core/internals` substrate and probes
- `tests/evaluation/` covers the Inspect AI evaluation stack (provider, collator, solver, scorer adapter, suite, runner)
- `tests/utils/` holds shared test utilities

## Executing tests

Running tests requires that the toolkit is installed with the `all` extra and the `dev` dependency group. First, run:
```commandline
uv sync --extra all
```
To execute tests for all controls, run:
```commandline
pytest tests/controls/
```
Tests for single controls are executed by specifying the file name, e.g.,:
```commandline
pytest tests/controls/test_pasta.py
```

## Adding your own control test

To add your own test for a steering method, please follow the pattern in the existing test files. Specifically:

1. Create a test file named `test_your_control.py` in the `tests/controls/` directory

2. Define test parameters for your control using a grid dictionary and `build_param_grid()`:
```python
YOUR_CONTROL_GRID = {
    "param_1": [value_1_1, value_1_2],
    "param_2": [value_2_1, value_2_2, value_2_3],
    # ... other parameters your control accepts
}

@pytest.mark.parametrize("conf", build_param_grid(YOUR_CONTROL_GRID))
def test_your_control(model_and_tokenizer, device: torch.device, conf: dict):
```

3. Follow the standard test pattern:
   - Move model to target device
   - Initialize your control with parameters from `conf`
   - Create a `SteeringPipeline` with your control
   - Call `pipeline.steer()` to set up the model
   - Prepare prompt and any runtime kwargs (if necessary)
   - Call `pipeline.generate()`
   - Specify necessary assertions

Executing the test will automatically run your test across all available models and devices with every combination of parameters in your grid.
