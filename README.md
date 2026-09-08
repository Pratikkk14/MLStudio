# ML Studio

Stateful Agentic AutoML for tabular machine learning.

ML Studio is a university-level machine learning project implementing a stateful, agentic AutoML system. Rather than executing a hardcoded sequence of steps, ML Studio coordinates ML workflows via an AI agent that observes intermediate outputs, reasons about constraints, and dynamically decides the next course of action.

---

## Key Features (Design Scope)

- **State-aware agent loop**: Continuous Observe-Reason-Decide-Act workflow cycle.
- **Structured agent decisions**: Decisions conform to schema parameters.
- **Deterministic tool execution**: Pure ML processing runs locally using Scikit-Learn / Pandas.
- **Data profiling & validation**: Heuristic identification of problems, targets, and leakage warnings.
- **Explainability**: Summarizes reasoning for all major pipeline decisions.

---

## Directory Structure

```text
ml-studio/
├── README.md
├── ML_STUDIO_SPEC.md
├── requirements.txt
├── .env.example
├── .gitignore
├── src/
│   ├── main.py              # CLI Entry point
│   ├── agent/               # Agent reasoning layer
│   ├── data/                # Data loading, validation, profiling
│   ├── preprocessing/       # Feature engineering & scaling
│   ├── models/              # Model training & tuning
│   ├── evaluation/          # Validation & evaluation metrics
│   ├── tools/               # Agent-facing ML tools
│   └── utils/               # Log & config utilities
├── tests/                   # Python unit tests
├── datasets/                # Tabular datasets storage
└── outputs/                 # Output models, reports, and logs
```

---

## Setup & Installation

1. Create a virtual environment and activate it:
   ```bash
   python -m venv myvenv
   myvenv\Scripts\activate
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy environment file and configure variables:
   ```bash
   copy .env.example .env
   ```

---

## Running the Application

To check if the CLI helper is set up correctly:
```bash
python -m src.main --help

# Run complete AutoML workflow on any CSV/Excel dataset
python -m src.main --dataset datasets/synthetic_dataset.xlsx

# Optional: Run with explicit target and metric
python -m src.main --dataset datasets/synthetic_dataset.xlsx --target churn --metric f1

# Optional: Run in interactive mode with human-in-the-loop checkpoints
python -m src.main --dataset "E:\Downloads\Dry_Bean_Dataset.xlsx" --interactive

# Run automated unit tests
pytest
```