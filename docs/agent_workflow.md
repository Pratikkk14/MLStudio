# Agent Workflow Design

This document details the stateful agent workflow in ML Studio.

## Workflow Loop

The agent follows an execution loop:
1. **Observe**: Read the current state (metadata, data profile, experimental results).
2. **Reason**: Analyze metrics, class balance, outlier details, and compute budget constraints.
3. **Decide**: Formulate structured next actions (e.g. preprocessing selection, baseline models training, model tuning).
4. **Execute**: Call the corresponding deterministic execution tools.
5. **Update State**: Record outputs and update current workflow phase.
