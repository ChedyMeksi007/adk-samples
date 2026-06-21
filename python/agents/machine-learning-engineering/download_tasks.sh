#!/bin/bash
# Download MLE-Bench Lite tasks from Kaggle and set up task folders
# These are the tasks used in the e2e benchmark (test_e2e_benchmark.py)

TASKS_DIR="./machine_learning_engineering/tasks"

TASKS=(
  # tabular regression
  "california-housing-prices"
  # image classification
  "aerial-cactus-identification"
  "leaf-classification"
  # image to image
  "denoising-dirty-documents"
  # text classification
  "detecting-insults-in-social-commentary"
  "spooky-author-identification"
  "jigsaw-toxic-comment-classification-challenge"
  "random-acts-of-pizza"
  # tabular
  "nomad2018-predict-transparent-conductors"
  # seq -> seq
  "text-normalization-challenge-english-language"
  "text-normalization-challenge-russian-language"
)

success=0
failed=0
skipped=0

for task in "${TASKS[@]}"; do
  echo "============================================"
  echo "Task: $task"
  echo "============================================"

  task_dir="$TASKS_DIR/$task"

  # Skip if folder exists and has files (besides task_description.txt)
  if [ -d "$task_dir" ]; then
    file_count=$(find "$task_dir" -type f ! -name "task_description.txt" | wc -l)
    if [ "$file_count" -gt 0 ]; then
      echo "SKIP: already has $file_count data files"
      skipped=$((skipped + 1))
      echo ""
      continue
    fi
  fi

  mkdir -p "$task_dir"

  # Download competition data
  uv run kaggle competitions download -c "$task" -p "$task_dir" 2>&1
  dl_status=$?

  if [ $dl_status -ne 0 ]; then
    echo ""
    echo "FAILED: $task"
    echo "  Possible fixes:"
    echo "  1. Accept rules: https://www.kaggle.com/competitions/$task/rules"
    echo "  2. Check if competition exists: https://www.kaggle.com/competitions/$task"
    echo "  3. Check your kaggle.json credentials"
    echo ""
    failed=$((failed + 1))
    continue
  fi

  # Unzip all zip files
  for zip in "$task_dir"/*.zip; do
    if [ -f "$zip" ]; then
      echo "Unzipping: $(basename "$zip")"
      unzip -o -q "$zip" -d "$task_dir"
      rm "$zip"
    fi
  done

  # Unzip nested zips (some competitions have train.csv.zip etc.)
  for zip in "$task_dir"/*.zip; do
    if [ -f "$zip" ]; then
      echo "Unzipping nested: $(basename "$zip")"
      unzip -o -q "$zip" -d "$task_dir"
      rm "$zip"
    fi
  done

  # Show what was downloaded
  echo "Files:"
  ls "$task_dir" | head -15
  file_count=$(find "$task_dir" -type f | wc -l)
  echo "($file_count files total)"
  echo ""

  success=$((success + 1))
done

echo "============================================"
echo "SUMMARY"
echo "============================================"
echo "  Downloaded: $success"
echo "  Skipped:    $skipped"
echo "  Failed:     $failed"
echo ""
echo "Next step: create task_description.txt for each task."
echo "============================================"
