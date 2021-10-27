#!/bin/bash -eux

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
$script_dir/../cmake-build-release/main_multi_evo _input.json
export PIPENV_PIPFILE=${script_dir}/Pipfile
which python
pyenv versions
which pipenv
pipenv run python ${script_dir}/plot_timeseries.py
