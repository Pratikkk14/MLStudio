# Experiment Design and Evaluation

This document outlines how experiments are designed, executed, and compared.

## Core Rules

1. **Isolation**: Always split data into train, cross-validation, and final untouched test sets.
2. **Fair Comparison**: Every candidate model must be evaluated using the exact same target column, identical data split, and matching primary metric.
3. **Resource Awareness**: Restrict the number of models trained, hyperparameter tuning trials, and overall computation time based on the active budget.
