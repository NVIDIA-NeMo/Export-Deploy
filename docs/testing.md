# Test NeMo Export-Deploy

This guide outlines how to test NeMo Export and Deploy using unit and functional tests, detailing steps for local or Docker-based execution, dependency setup, and metric tracking to ensure effective and reliable testing.

## Unit Tests

:::{important}
Unit tests require at least a GPU to test the full suite.
:::

```sh
# Run the unit tests using local GPUs
pytest -s tests/unit_tests/
```

## Functional Tests

:::{important}
Functional tests may require multiple GPUs to run. See each script to understand the requirements.
:::

Will be updated later!
