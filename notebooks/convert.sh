#!/bin/zsh

source /opt/homebrew/anaconda3/bin/activate base

for arg; do
  black --preview --line-length 65 "$arg"
  jupyter nbconvert "$arg" \
    --to markdown \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags remove_cell \
    --TagRemovePreprocessor.remove_all_outputs_tags remove_output \
    --TagRemovePreprocessor.remove_input_tags remove_input
done