"""Stats: small SQL summaries (DuckDB) over the shot metadata stored in the index."""

from __future__ import annotations

import duckdb
import pandas as pd

# Each query takes one parameter: a list of tournaments (an empty list means all).
TOP_SCORERS = """
SELECT player, team, COUNT(*) AS goals, ROUND(SUM(xg), 2) AS xg
FROM shots
WHERE is_goal AND NOT is_shootout
  AND (len($tournaments) = 0 OR list_contains($tournaments, tournament))
GROUP BY player, team
ORDER BY goals DESC, xg ASC, player
LIMIT 10
"""

GOALS_BY_BODY_PART = """
SELECT body_part, COUNT(*) AS goals,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
FROM shots
WHERE is_goal AND NOT is_shootout
  AND (len($tournaments) = 0 OR list_contains($tournaments, tournament))
GROUP BY body_part
ORDER BY goals DESC
"""

FINISHING_BY_ZONE = """
SELECT zone, COUNT(*) AS shots, SUM(is_goal::INT) AS goals,
       ROUND(100.0 * SUM(is_goal::INT) / COUNT(*), 1) AS conversion_pct,
       ROUND(AVG(xg), 3) AS avg_xg
FROM shots
WHERE NOT is_shootout
  AND (len($tournaments) = 0 OR list_contains($tournaments, tournament))
GROUP BY zone
ORDER BY avg_xg DESC
"""

QUERIES = {
    "Top scorers": TOP_SCORERS,
    "Goals by body part": GOALS_BY_BODY_PART,
    "Finishing by zone": FINISHING_BY_ZONE,
}


def run(sql: str, shots: pd.DataFrame, tournaments: list[str] | None = None) -> pd.DataFrame:
    """Run one of the queries above against a DataFrame of shot metadata."""
    con = duckdb.connect()
    con.register("shots", shots)
    try:
        return con.execute(sql, {"tournaments": list(tournaments or [])}).df()
    finally:
        con.close()
