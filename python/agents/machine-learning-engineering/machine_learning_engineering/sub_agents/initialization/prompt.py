"""Defines the prompts for the initialization agent."""

SUMMARIZATION_AGENT_INSTR = """# Task description
{task_description}

# Your task
- Summarize this task description.
- Your summary will be used for searching recent effective models for {task_type}.

# Requirement
- We will directly use your response, so be simple and concise.
"""

MODEL_RETRIEVAL_INSTR = """# Competition
{task_summary}

# Your task
- List {num_model_candidates} recent effective models and their example codes to win the above competition.
- All solutions must be built as a **skrub DataOps plan** (the deferred `.skb` computation graph), NOT a plain scikit-learn `Pipeline`.

# Requirement
- The example code should be concise and simple.
- You must provide an example code, i.e., do not just mention GitHubs or papers.
- Each example must build a skrub DataOps plan: declare inputs with `skrub.var(...)`, mark features/target with `.skb.mark_as_X()` / `.skb.mark_as_y()`, transform with `.skb.apply(skrub.TableVectorizer())` (or encoders like `GapEncoder`, `MinHashEncoder`, `DatetimeEncoder`, `StringEncoder`), add the estimator with `.skb.apply(model, y=y)`, and export with `.skb.make_learner()`.
- Do NOT use `sklearn.pipeline.Pipeline` / `make_pipeline`; the whole flow must live in the skrub DataOps graph.

Use this JSON schema:
Model = {{'model_name': str, 'example_code': str}}
Return: list[Model]"""

MODEL_RETRIEVAL_INSTR_DEFAULT = """# Competition
{task_summary}

# Your task
- List {num_model_candidates} recent effective models and their example codes to win the above competition.

# Requirement
- The example code should be concise and simple.
- You must provide an example code, i.e., do not just mention GitHubs or papers.

Use this JSON schema:
Model = {{'model_name': str, 'example_code': str}}
Return: list[Model]"""


MODEL_EVAL_INSTR = """# Introduction
- You are a Kaggle grandmaster attending a competition.
- We will now provide a task description and a model description.
- You need to implement your Python solution using the provided model.
- **You must build your solution as a skrub DataOps pipeline.**

# Task description
{task_description}

# Model description
{model_description}

# skrub DataOps requirements
You MUST structure your solution as a skrub **DataOps plan** — the deferred `.skb` computation graph — NOT a plain scikit-learn `Pipeline`/`make_pipeline`:
1. Declare the inputs as DataOps variables with `skrub.var("name", value)`. Load the raw dataframe into a variable, then derive features and target from it.
2. Mark the design matrix and target inside the graph with `.skb.mark_as_X()` and `.skb.mark_as_y()` — do NOT split into NumPy arrays by hand.
3. Apply preprocessing inside the graph with `.skb.apply(skrub.TableVectorizer())`. `TableVectorizer` auto-detects column types (OneHotEncoder for low-cardinality, GapEncoder for high-cardinality text, DatetimeEncoder for dates). For text-heavy columns customize it, e.g. `skrub.TableVectorizer(high_cardinality=skrub.MinHashEncoder())` or apply `skrub.GapEncoder` / `skrub.StringEncoder` / `skrub.DatetimeEncoder` to specific columns.
4. Add the supervised estimator inside the graph with `.skb.apply(model, y=y)`.
5. Export the whole graph as a learner with `.skb.make_learner()`; fit it with `learner.fit(env)` and predict with `learner.predict(env)`, where `env` is the dict of variable values.
6. Evaluate with `predictions.skb.cross_validate(...)` or by holding out via `.skb.train_test_split()` — keep everything inside the DataOps graph.
7. Do NOT use `sklearn.pipeline.Pipeline`, `make_pipeline`, or `skrub.tabular_pipeline`; all transformation and modeling must be expressed as `.skb` operations.

Example skeleton:
```python
import pandas as pd
import skrub
from sklearn.linear_model import LogisticRegression  # or the model from the description

df = pd.read_csv("./input/train.csv")
data = skrub.var("data", df)                       # declare the input
X = data.drop(columns=[target_col]).skb.mark_as_X()
y = data[target_col].skb.mark_as_y()

X_vec = X.skb.apply(skrub.TableVectorizer())       # preprocessing inside the graph
pred  = X_vec.skb.apply(model, y=y)                # estimator inside the graph

cv = pred.skb.cross_validate(scoring=metric_name)  # evaluate the plan
print(f"Final Validation Performance: {{cv['test_score'].mean()}}")

learner = pred.skb.make_learner()                  # export for fit/predict
learner.fit({{"data": df}})
```

# Your task
- Implement the solution in Python as a skrub DataOps plan as described above.
- You must use the model as described in the model description, applied inside the DataOps graph via `.skb.apply(model, y=y)`.
- This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.
- Propose an evaluation metric that is reasonable for this task.
- All the provided data is already prepared and available in the `./input` directory. There is no need to unzip any files.
- Do not include other models that are not directly related to the model described.
- All the necessary libraries are installed (skrub, scikit-learn, pandas, numpy, etc.).
- The code should implement the proposed solution and print the value of the evaluation metric computed on a hold-out validation set.
- Only use the provided train data in the `./input` directory.

# Required
- There should be no additional headings or text in your response.
- Print out or return a final performance metric in your answer in a clear format with the exact words: 'Final Validation Performance: {{final_validation_score}}'.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not use exit() function in the Python code.
- Do not use try: and except: or if else to ignore unintended behavior.
"""

MODEL_EVAL_INSTR_DEFAULT = """# Introduction
- You are a Kaggle grandmaster attending a competition.
- We will now provide a task description and a model description.
- You need to implement your Python solution using the provided model.

# Task description
{task_description}

# Model description
{model_description}

# Your task
- Implement the solution in Python.
- You must use the model as described in the model description.
- This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.
- Propose an evaluation metric that is reasonable for this task.
- All the provided data is already prepared and available in the `./input` directory. There is no need to unzip any files.
- Do not include other models that are not directly related to the model described.
- Use PyTorch rather than TensorFlow. Use CUDA if you need. All the necessary libraries are installed.
- The code should implement the proposed solution and print the value of the evaluation metric computed on a hold-out validation set.
- Only use the provided train data in the `./input` directory.

# Required
- There should be no additional headings or text in your response.
- Print out or return a final performance metric in your answer in a clear format with the exact words: 'Final Validation Performance: {{final_validation_score}}'.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not use exit() function in the Python code.
- Do not use try: and except: or if else to ignore unintended behavior.
"""

CODE_INTEGRATION_INSTR = """# Introduction
- You are a Kaggle grandmaster attending a competition.
- We will now provide a base solution and an additional reference solution.
- Both solutions are built as **skrub DataOps plans** (the deferred `.skb` computation graph).
- You need to implement your Python solution by integrating reference solution to the base solution while maintaining the skrub DataOps graph structure.

# Base solution
```python
{base_code}
```

# Reference solution
```python
{reference_code}
```

# Your task
- Implement the solution in Python as a skrub DataOps plan.
- You have to integrate the reference solution to the base solution.
- Your code base should be the base solution.
- Try to train the additional model of the reference solution as a separate branch in the same DataOps graph (its own `.skb.apply(model, y=y)`).
- When integrating, share the `.skb.apply(skrub.TableVectorizer())` preprocessing between branches where possible, or give each branch its own preprocessing, then ensemble the prediction DataOps.
- Ensemble by combining the prediction DataOps (e.g. average the per-model `.skb.apply(..., y=y)` outputs inside the graph), keeping everything expressed as `.skb` operations. Do NOT fall back to `sklearn.pipeline.Pipeline`/`make_pipeline`.
- The solution design should be relatively simple.
- The code should implement the proposed solution and print the value of the evaluation metric computed on a hold-out validation set.
- Only use the provided train data in the `./input` directory.

# Required
- There should be no additional headings or text in your response.
- Print out or return a final performance metric in your answer in a clear format with the exact words: 'Final Validation Performance: {{final_validation_score}}'.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not use exit() function in the Python code.
- Do not use try: and except: or if else to ignore unintended behavior."""

CODE_INTEGRATION_INSTR_DEFAULT = """# Introduction
- You are a Kaggle grandmaster attending a competition.
- We will now provide a base solution and an additional reference solution.
- You need to implement your Python solution by integrating reference solution to the base solution.

# Base solution
```python
{base_code}
```

# Reference solution
```python
{reference_code}
```

# Your task
- Implement the solution in Python.
- You have to integrate the reference solution to the base solution.
- Your code base should be the base solution.
- Try to train additional model of the reference solution.
- When integrating, try to keep code with similar functionality in the same place (e.g., all preprocessing should be done and then all training).
- When integrating, ensemble the models.
- The solution design should be relatively simple.
- The code should implement the proposed solution and print the value of the evaluation metric computed on a hold-out validation set.
- Only use the provided train data in the `./input` directory.

# Required
- There should be no additional headings or text in your response.
- Print out or return a final performance metric in your answer in a clear format with the exact words: 'Final Validation Performance: {{final_validation_score}}'.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not use exit() function in the Python code.
- Do not use try: and except: or if else to ignore unintended behavior."""

CHECK_DATA_USE_INSTR = """I have provided Python code for a machine learning task (attached below):
# Solution Code
```python
{code}
```

# Task description
{task_description}

# Your task
If the above solution code does not use the information provided, try to incorporate all. Do not bypass using try-except.
DO NOT USE TRY and EXCEPT; just occur error so we can debug it!
See the task description carefully, to know how to extract unused information effectively.
When improving the solution code by incorporating unused information:
- **Maintain the skrub DataOps graph structure** (`skrub.var` → `.skb.mark_as_X`/`mark_as_y` → `.skb.apply(...)` → `.skb.make_learner`). Do not replace it with a scikit-learn `Pipeline`/`make_pipeline` or with manual preprocessing.
- Use skrub components inside the graph via `.skb.apply(...)` (e.g., `skrub.GapEncoder`, `skrub.MinHashEncoder`, `skrub.DatetimeEncoder`) to incorporate additional columns or data types.
- DO NOT FORGET to print out 'Final Validation Performance: {{final_validation_score}}' as in original solution code.

Response format:
Option 1: If the code did not use all the provided information, your response should be a single markdown code block (wrapped in ```) which is the improved code block. There should be no additional headings or text in your response.
Option 2: If the code used all the provided information, simply state that "All the provided information is used."
"""

CHECK_DATA_USE_INSTR_DEFAULT = """I have provided Python code for a machine learning task (attached below):
# Solution Code
```python
{code}
```

# Task description
{task_description}

# Your task
If the above solution code does not use the information provided, try to incorporate all. Do not bypass using try-except.
DO NOT USE TRY and EXCEPT; just occur error so we can debug it!
See the task description carefully, to know how to extract unused information effectively.
When improving the solution code by incorporating unused information, DO NOT FORGET to print out 'Final Validation Performance: {{final_validation_score}}' as in original solution code.

Response format:
Option 1: If the code did not use all the provided information, your response should be a single markdown code block (wrapped in ```) which is the improved code block. There should be no additional headings or text in your response.
Option 2: If the code used all the provided information, simply state that "All the provided information is used."
"""
