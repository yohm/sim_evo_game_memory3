#!/bin/bash -eux

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
$script_dir/../cmake-build-release/main_mem1 _input.json
export PIPENV_PIPFILE=${script_dir}/Pipfile
pipenv run python ${script_dir}/plot_timeseries.py
pipenv run python ${script_dir}/plot_frequency.py
