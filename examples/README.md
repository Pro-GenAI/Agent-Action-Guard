# Examples

This folder contains runnable examples that show how to use the Action Classifier.

- `example_query.py` — a small script that loads `HarmActEval_dataset.json` and classifies the first few actions.

Notes:

- The example first tries to load the real `ActionClassifier()` (which requires `torch` and `sentence-transformers` and a trained model file). If that fails it automatically falls back to a lightweight demo classifier so you can run the example without the heavy dependencies.
- To run with the real model, ensure `emb_nn_model.pt` is present in the repository root and the dependencies in `requirements.txt` are installed.
