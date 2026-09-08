# ML Studio Architecture

This document describes the high-level architecture of ML Studio.

## Overview

ML Studio is a stateful, agentic AutoML system for tabular datasets. It operates on a feedback loop of observation, decision-making, and execution.

## System Components

1. **Reasoning Layer (Agent)**: Orchestrates workflows, makes decisions on what actions to take next, interprets results, and handles human-in-the-loop overrides.
2. **Execution Layer (ML Engine)**: Handles deterministic ML pipeline construction, training, hyperparameter tuning, and data validation/profiling.
3. **State Management**: Persists the current state of the workflow, dataset statistics, warnings, and experimental history.
