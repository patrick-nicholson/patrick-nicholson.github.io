#!/bin/zsh

source /opt/homebrew/anaconda3/bin/activate base

for arg; do
  cd "${arg:h}" || exit
  cp "${arg:t}" nb.ipynb
  black --preview --line-length 65 nb.ipynb
  jupyter nbconvert nb.ipynb \
    --to markdown \
    --NbConvertApp.output_files_dir "${arg:t:r}_files" \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags remove_cell \
    --TagRemovePreprocessor.remove_all_outputs_tags remove_output \
    --TagRemovePreprocessor.remove_input_tags remove_input
  mv nb.md "${arg:t:r}.md"
  rm nb.ipynb
done