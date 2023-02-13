#!/bin/zsh

source /opt/homebrew/anaconda3/bin/activate

black --line-length 65 "$1"

jupyter nbconvert "$1" \
  --to markdown \
  --TagRemovePreprocessor.enabled=True \
  --TagRemovePreprocessor.remove_cell_tags remove_cell \
  --TagRemovePreprocessor.remove_all_outputs_tags remove_output \
  --TagRemovePreprocessor.remove_input_tags remove_input
