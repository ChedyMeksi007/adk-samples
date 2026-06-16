"""Defines the prompts for the submission agent."""

ADD_TEST_FINAL_INSTR = """# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you need to come up with an excellent solution in Python.
- We will now provide a task description and a Python solution that is built as a **skrub DataOps plan** (the deferred `.skb` computation graph).
- What you have to do on the solution is just loading test samples and create a submission file.

# Task description
{task_description}

# Python solution
```python
{code}
```

# Your task
- Load the test samples and create a submission file.
- The solution is a skrub DataOps plan — call `learner = <final_pred>.skb.make_learner()`, fit it on the training environment, then predict on the test environment (`learner.predict({{"data": test_df, ...}})`) using the same variable names declared with `skrub.var(...)`.
- All the provided data is already prepared and available in the `./input` directory. There is no need to unzip any files.
- Test data is available in the `./input` directory.
- Save the test predictions in a `submission.csv` file. Put the `submission.csv` into `./final` directory.
- You should not drop any test samples. Predict the target value for all test samples.
- This is a very easy task because the only thing to do is to load test samples and then replace the validation samples with the test samples. Then you can even use the full training set!

# Required
- Do not modify the given Python solution code too much. Try to integrate test submission with minimal changes.
- Maintain the skrub DataOps graph structure (`skrub.var` -> `.skb.apply(...)` -> `.skb.make_learner`) — do not replace it with a scikit-learn `Pipeline`/`make_pipeline` or with manual preprocessing.
- There should be no additional headings or text in your response.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not forget the ./final/submission.csv file.
- Do not use exit() function in the Python code.
- Do not use try: and except: or if else to ignore unintended behavior."""

ADD_TEST_FINAL_INSTR_DEFAULT = """# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you need to come up with an excellent solution in Python.
- We will now provide a task description and a Python solution.
- What you have to do on the solution is just loading test samples and create a submission file.

# Task description
{task_description}

# Python solution
```python
{code}
```

# Your task
- Load the test samples and create a submission file.
- All the provided data is already prepared and available in the `./input` directory. There is no need to unzip any files.
- Test data is available in the `./input` directory.
- Save the test predictions in a `submission.csv` file. Put the `submission.csv` into `./final` directory.
- You should not drop any test samples. Predict the target value for all test samples.
- This is a very easy task because the only thing to do is to load test samples and then replace the validation samples with the test samples. Then you can even use the full training set!

# Required
- Do not modify the given Python solution code too much. Try to integarte test submission with minimal changes.
- There should be no additional headings or text in your response.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not forget the ./final/submission.csv file.
- Do not use exit() function in the Python code.
- Do not use try: and except: or if else to ignore unintended behavior."""
