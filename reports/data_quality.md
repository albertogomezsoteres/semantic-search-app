# Data quality report

Source: `statsbomb/open-data` at commit `4b73468fc5`

| Check | Result | Detail |
|---|---|---|
| Required fields present | pass | 9 fields |
| Values in valid ranges | pass | 0 <= xg <= 1, 0 <= x <= 120, 0 <= y <= 80 |
| Shot IDs unique | pass | 0 duplicates |
| Text fits the embedding model | pass | longest 97 words |
| Shot counts per competition plausible | pass | FIFA World Cup 2022: 1494 shots / 64 matches |
| Goals reconcile with final scores | pass | 64/64 matches reconcile |
