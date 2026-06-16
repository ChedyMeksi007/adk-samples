"""Defines the prompts for the ensemble agent."""

INIT_ENSEMBLE_PLAN_INSTR = """# Introduction
- You are a Kaggle grandmaster attending a competition.
- We will now provide {num_solutions} Python Solutions that are built as **skrub DataOps plans** (the deferred `.skb` computation graph).
- Your task is to propose a plan to ensemble the {num_solutions} solutions to achieve the best performance.

{python_solutions}

# Your task
- Suggest a plan to ensemble the {num_solutions} skrub DataOps plans. You should concentrate how to merge, not the other parts like hyperparameters.
- Combine the predictions inside the DataOps graph — e.g. average or blend the per-solution prediction DataOps (`.skb.apply(model, y=y)` outputs), or stack them by feeding their predictions into a meta-estimator via `.skb.apply(...)`.
- Do NOT introduce `sklearn.pipeline.Pipeline`/`make_pipeline` or `VotingRegressor`/`VotingClassifier`; keep the ensemble expressed as `.skb` operations on the existing DataOps plans.
- The suggested plan should be novel, effective, and easy to implement.
- All the provided data is already prepared and available in the `./input` directory. There is no need to unzip any files.

# Response format
- Your response should be an outline/sketch of your proposed solution in natural language.
- There should be no additional headings or text in your response.
- Plan should not modify the original solutions too much since execution error can occur."""

INIT_ENSEMBLE_PLAN_INSTR_DEFAULT = """# Introduction
- You are a Kaggle grandmaster attending a competition.
- We will now provide {num_solutions} Python Solutions used for the competiton.
- Your task is to propose a plan to ensemble the {num_solutions} solutions to achieve the best performance.

{python_solutions}

# Your task
- Suggest a plan to ensemble the {num_solutions} solutions. You should concentrate how to merge, not the other parts like hyperparameters.
- The suggested plan should be easy to novel, effective, and easy to implement.
- All the provided data is already prepared and available in the `./input` directory. There is no need to unzip any files.

# Respone format
- Your response should be an outline/sketch of your proposed solution in natural language.
- There should be no additional headings or text in your response.
- Plan should not modify the original solutions too much since exeuction error can occur."""

ENSEMBLE_PLAN_IMPLEMENT_INSTR = """# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you need to ensemble {num_solutions} Python Solutions that are built as **skrub DataOps plans** for better performance based on the ensemble plan.
- We will now provide the Python Solutions and the ensemble plan.

{python_solutions}

# Ensemble Plan
{plan}

# Your task
- Implement the ensemble plan with the provided skrub DataOps solutions.
- Unless mentioned in the ensemble plan, do not modify the original Python Solutions too much.
- Maintain the skrub DataOps graph structure (`skrub.var` → `.skb.apply(...)` → `.skb.make_learner`) in each solution; do not convert them to `sklearn.pipeline.Pipeline`/`make_pipeline`.
- All the provided data is already prepared and available in the `./input` directory. There is no need to unzip any files.
- The code should implement the proposed solution and print the value of the evaluation metric computed on a hold-out validation set.

# Response format required
- Your response should be a single markdown code block (wrapped in ```) which is the ensemble of {num_solutions} Python Solutions.
- There should be no additional headings or text in your response.
- Do not modify original Python Solutions especially the submission part due to formatting issue of submission.csv.
- Do not subsample or introduce dummy variables. You have to provide full new Python Solution using the {num_solutions} provided solutions.
- Print out or return a final performance metric in your answer in a clear format with the exact words: 'Final Validation Performance: {{final_validation_score}}'.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Do not modify the original codes too much and implement the plan since new errors can occur."""

ENSEMBLE_PLAN_IMPLEMENT_INSTR_DEFAULT = """# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you need to ensemble {num_solutions} Python Solutions for better performance based on the ensemble plan.
- We will now provide the Python Solutions and the ensemble plan.

{python_solutions}

# Ensemble Plan
{plan}

# Your task
- Implement the ensemble plan with the provided solutions.
- Unless mentioned in the ensemble plan, do not modify the origianl Python Solutions too much.
- All the provided data is already prepared and available in the `./input` directory. There is no need to unzip any files.
- The code should implement the proposed solution and print the value of the evaluation metric computed on a hold-out validation set.

# Response format required
- Your response should be a single markdown code block (wrapped in ```) which is the ensemble of {num_solutions} Python Solutions.
- There should be no additional headings or text in your response.
- Do not modify original Python Solutions especially the submission part due to formatting issue of submission.csv.
- Do not subsample or introduce dummy variables. You have to provide full new Python Solution using the {num_solutions} provided solutions.
- Print out or return a final performance metric in your answer in a clear format with the exact words: 'Final Validation Performance: {{final_validation_score}}'.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Do not modify the original codes too much and implement the plan since new errors can occur."""

ENSEMBLE_PLAN_REFINE_INSTR = """# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you have to ensemble {num_solutions} Python Solutions that are built as **skrub DataOps plans** for better performance.
- We will provide the Python Solutions and the ensemble plans you have tried.

{python_solutions}

# Ensemble plans you have tried

{prev_plans_and_scores}

# Your task
- Suggest a better plan to ensemble the {num_solutions} skrub DataOps solutions. You should concentrate how to merge, not the other parts like hyperparameters.
- Maintain the skrub DataOps graph structure (`skrub.var` → `.skb.apply(...)` → `.skb.make_learner`) in each solution.
- Combine predictions inside the DataOps graph (average/blend the per-solution prediction DataOps, or stack via a meta-estimator `.skb.apply(...)`). Do NOT use `sklearn.pipeline.Pipeline`/`make_pipeline` or `VotingRegressor`/`VotingClassifier`.
- The suggested plan must be easy to implement, novel, and effective.
- The suggested plan should be differ from the previous plans you have tried and should receive a {criteria} score.

# Response format
- Your response should be an outline/sketch of your proposed solution in natural language.
- There should be no additional headings or text in your response.
- Plan should not modify the original solutions too much since execution error can occur."""

ENSEMBLE_PLAN_REFINE_INSTR_DEFAULT = """# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you have to ensemble {num_solutions} Python Solutions for better performance.
- We will provide the Python Solutions and the ensemble plans you have tried.

{python_solutions}

# Ensemble plans you have tried

{prev_plans_and_scores}

# Your task
- Suggest a better plan to ensemble the {num_solutions} solutions. You should concentrate how to merge, not the other parts like hyperparameters.
- The suggested plan must be easy to implement, novel, and effective.
- The suggested plan should be differ from the previous plans you have tried and should receive a {criteria} score.

# Response format
- Your response should be an outline/sketch of your proposed solution in natural language.
- There should be no additional headings or text in your response.
- Plan should not modify the original solutions too much since execution error can occur."""
