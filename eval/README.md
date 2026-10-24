# Evaluation sets

| File | Questions | Answerable | Out of scope | Used for tuning? |
|---|---:|---:|---:|---|
| `dev.json` | 23 | 11 | 12 | **Yes.** `CROSS_ENCODER_THRESHOLD = 0.01` was chosen on this set. |
| `heldout.json` | 60 | 54 | 6 | No. Written after the threshold was fixed. |
| `adversarial.json` | 30 | 0 | 30 | No. Topically adjacent questions the corpus does not answer. |

Numbers reported for the project should come from `heldout` and
`adversarial`. `dev` numbers are optimistic by construction.

## Scoring

Each question has `expect_grounded` and, for answerable questions,
`expect_keywords_any`: substrings (usually the source URL) that must
appear in a returned source's title, URL or text.

- **accuracy**: answerable and grounded with a matching top-5 source, or
  out of scope and refused.
- **hit@5 / MRR**: over answerable questions only.
- **refusal_accuracy**: over out-of-scope questions only.
- **jurisdiction_violations / audience_violations**: returned sources outside
  the allowed jurisdictions or audience. Must be 0.

## How the held-out set was built

- Every answerable question was checked to have at least one target document
  in the corpus that is reachable under its audience and jurisdiction.
- Every adversarial question was checked to contain a key term that does not
  appear anywhere in the corpus (e.g. *pulmonary hypertension*, *renal cell
  carcinoma*, *CCPA*). Three drafted questions were dropped because the
  corpus does mention their topic (sleep apnea, A1C/prediabetes, Social
  Security).

## Running

```bash
python -m uvicorn api.main:app --port 8000 &
python run_eval.py --json-out eval/results/latest.json
```

CI runs this on every pull request (`.github/workflows/eval.yml`) and posts
the per-set summary on the run.
