#!/bin/zsh

micromamba activate

for arg; do
  cd "${arg:h}" || exit
  cp "${arg:t}" nb.ipynb
  black --preview --line-length 65 nb.ipynb
  jupyter nbconvert nb.ipynb \
    --to markdown \
    --NbConvertApp.output_files_dir "../public/${arg:t:r}_files" \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags remove_cell \
    --TagRemovePreprocessor.remove_all_outputs_tags remove_output \
    --TagRemovePreprocessor.remove_input_tags remove_input
  mv nb.md "../_posts/${arg:t:r}.md"
  rm nb.ipynb
done