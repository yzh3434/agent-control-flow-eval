# Agent Control-Flow Evaluation

Compares four hand-written LLM agent control flows — Direct, CoT, ReAct, Reflexion —
on GSM8K, measuring success rate, average interaction rounds, and token cost.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then put your DeepSeek API key in .env
python scripts/download_gsm8k.py
```

## Run

```bash
python scripts/run_experiment.py --controllers direct cot react reflexion --difficulty easy hard
python scripts/make_report_table.py
```

## Layout

- `src/llm.py` — DeepSeek client
- `src/tools/` — tool interface + calculator
- `src/envs/` — datasets, difficulty, grading
- `src/controllers/` — the four control flows
- `src/eval/` — runner, metrics, trajectory schema
- `scripts/` — download data, run experiments, build report table
