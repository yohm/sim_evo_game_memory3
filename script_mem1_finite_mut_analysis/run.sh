#!/bin/bash -eux

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
$script_dir/../cmake-build-release/main_mem1_finite_mut_analysis $*
export PIPENV_PIPFILE=${script_dir}/Pipfile
pipenv run python ${script_dir}/plot.py
