# Bayesian and Attentive Aggregation for Cooperative Multi-Agent Deep Reinforcement Learning

This repository contains the code base for my master's thesis. Please not that this code is "researcher"-level quality and not software-engineer quality so it doesn't really hold up to the standards I would usually hold myself to.

The original code was based on https://github.com/ALRhub/deep_rl_for_swarms/.

## installation

The main code is in /playground.

You need Python 3.8.

To get the dependencies and create the virtual env, run `poetry install`.

## utils

poetry run optuna dashboard --storage 'sqlite:///file:optuna.sqlite3?vfs=unix-dotfile&uri=true' --study-name=testify2

## jupyter setup

```bash
poetry run python -m ipykernel install --user --name poetry-env
poetry run jupyter labextension install jupyterlab-plotly @jupyter-widgets/jupyterlab-manager plotlywidget
```

## nix setup

before running `poetry install`, install build deps for pillow:

nix-shell zsh -p zlib.dev pkg-config libjpeg glib

## compiling stuff not working? (especially on bwunicluster)

`env |grep isystem` should not be empty

try `conda install -c anaconda gxx_linux-64`
