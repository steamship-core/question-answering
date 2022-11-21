# Developing

## Basic Information

* 🐍 The project targets Python 3.8
* ✍️ Code formatting is performed with black and isort
* ♻️ Continuous integration with Steamship can be configured with GitHub Actions
* ✅ Code linting is automated via pre-commit hooks: bandit, darglint, flake8, mypy, pre-commit-hooks, pydocstyle, pygrep-hooks, pyupgrade, safety, and shellcheck
* 📋 Testing is automated via Pyunit
* 🧑‍💻 We recommend PyCharm as a development environment

## Development Setup

### Set up virtual environment

First make sure you have Python3.8 -- the officially supported version

We highly recommend using virtual environments for development. 
Set up your virtual environment using the following commands:

```
python3.8 -m venv .venv
source .venv/bin/activate
python3.8 -m pip install -r requirements.txt
python3.8 -m pip install -r requirements.dev.txt
```

This will install the required dependencies (runtime and development) and register the project source tree with your virtual environment so that `import steamship` statements will resolve correctly.

### Set up pre-commit-hooks

We use pre-commit hooks to validate coding standards before submission to code review. To make sure your code is always validated before each commit, please install the required git hook scripts as follows: 
```bash
pre-commit install
```

Once completed the pre-commit hooks wil run automatically on `git commit`. 

When pre-commit hooks make file modifications, the `git commit` command that triggered them will fail and need to be run again. Simply run the command multiple times until it succeeds.

You can run the pre-commit hooks manually via:
```bash
pre-commit run --all-files
```

### Set your IDE to use proper Docstrings

Steamship uses PyCharm for Python development. 

In PyCharm:

* Navigate to Preferences -> Tools -> Python Integrated Tools -> Docstring Format
* Select "NumPy" as the Docstring Format.

## Testing

### Configuring Test Credentials

The tests include integration tests that are intended to be performed against a running Steamship server. 