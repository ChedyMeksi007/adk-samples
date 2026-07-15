All requested issues have been resolved:

- Added robust installation and import handling for `pandas`, `scikit-learn`, and `skrub`.
- Implemented a version-tolerant import pattern for the `skb` namespace.
- Built a Skrub DataOps graph (`skrub.var` → `mark_as_X/Y` → `apply(TableVectorizer)` → `apply(LogisticRegression)` → `make_learner`).
- Performed a stratified train/validation split, trained the model, and printed the validation log-loss (`Final Validation Performance: ...`).
- Retrained on the full training data, generated probability predictions for the test set, and saved the submission CSV to `./final/submission.csv` with columns `id,EAP,HPL,MWS`.

The provided script is self-contained, follows the required DataOps structure, and includes the mandatory performance print statement. No further actions are needed.