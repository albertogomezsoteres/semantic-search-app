# Data quality report

Source: `statsbomb/open-data` at commit `4b73468fc5`

| Check | Result | Detail |
|---|---|---|
| Required fields present | pass | 10 fields |
| Values in valid ranges | pass | 0 <= xg <= 1, 0 <= x <= 120, 0 <= y <= 80 |
| Shot IDs unique | pass | 0 duplicates |
| Text fits the embedding model | pass | longest 105 words |
| Shot counts per competition plausible | pass | FIFA World Cup 2018: 1706 shots / 64 matches; FIFA World Cup 2022: 1494 shots / 64 matches; UEFA Euro 2020: 1289 shots / 51 matches; UEFA Euro 2024: 1340 shots / 51 matches; Women's World Cup 2019: 1314 shots / 52 matches; Women's World Cup 2023: 1678 shots / 64 matches |
| Goals reconcile with final scores | pass | 346/346 matches reconcile |
