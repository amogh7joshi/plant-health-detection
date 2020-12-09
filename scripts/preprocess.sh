#!/usr/bin/env bash

cd .. || exit

if [ -d "./data" ]; then
  # shellcheck disable=SC2164
  cd data
else
  raise error "The data directory does not exist. Please re-download the repository and try again."
fi

if [ -d "./dataset" ]; then
  # shellcheck disable=SC2164
  cd dataset
else
  raise error "The dataset directory does not exist. Please re-download the repository and try again."
fi

if [ -d "./plant_leaves" ]; then
  # shellcheck disable=SC2164
  cd plant_leaves

  # Remove existing files.
  rm -f X_train.pickle
  rm -f X_validation.pickle
  rm -f X_test.pickle

  rm -f y_train.pickle
  rm -f y_validation.pickle
  rm -f y_test.pickle

  cd .. || exit
fi

cd .. || exit

# Process Data
echo "Would you like to process everything at once, just the healthy images,
just the diseased images, or just the final processing? [e|h|d|f]"
read input
if [[ "$input" = "e" ]]; then
  # Full processing.
  python preprocess.py -m split
  python preprocess.py -m save
elif [[ "$input" = "h" ]]; then
  # Only preprocess healthy images.
  python preprocess.py -m split-healthy
  exit
elif [[ "$input" = "d" ]]; then
  # Only preprocess diseased images.
  python preprocess.py -m split-diseased
elif [[ "$input" = "f" ]]; then
  # Save Data
  python preprocess.py -m save
fi

# Remove Final Unnecessary Files
if [[ "$input" = "e" ]] || [[ "$input" = "f" ]]; then
  echo "Would you like to get rid of the remaining files? [y|n]"
  read in2
  if [[ "$in2" = "y" ]]; then
    rm -f diseased.pickle
    rm -f healthy.pickle
  else
    echo "Process Complete."
  fi
fi

exit

