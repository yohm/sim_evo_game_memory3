#!/bin/bash -eux

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
if [ $# -eq 1 ]; then
  $script_dir/../cmake-build-release/main_multi_evo $@
else
  $script_dir/../cmake-build-release/main_multi_evo _input.json
fi
export PIPENV_PIPFILE=${script_dir}/Pipfile
pipenv run python ${script_dir}/plot_timeseries.py
