# Documentation Development

- [Documentation Development](#documentation-development)
  - [Build the Documentation](#build-the-documentation)
  - [Live Building](#live-building)
  - [Run Tests in Python Docstrings](#run-tests-in-python-docstrings)
  - [Write Tests in Python Docstrings](#write-tests-in-python-docstrings)
  - [Documentation Version](#documentation-version)


## Build the Documentation

The following sections describe how to set up and build the NeMo Export and Deploy documentation.

Switch to the documentation source folder and generate HTML output.

```sh
python3 -m venv docs-env
source docs-env/bin/activate
pip install sphinx sphinx-autobuild sphinx-autodoc2 sphinx-copybutton myst_parser nvidia-sphinx-theme
sphinx-autobuild docs docs/_build/html

```

* The resulting HTML files are generated in a `_build/html` folder that is created under the project `docs/` folder.
* The generated python API docs are placed in `apidocs` under the `docs/` folder.


## Documentation Version

The three files below control the version switcher. Before you attempt to publish a new version of the documentation, update these files to match the latest version numbers.

* docs/versions1.json
* docs/project.json
* docs/conf.py

