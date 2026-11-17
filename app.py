"""Football Moments Search: a Gradio app over the prebuilt Chroma index.

This file is only the UI. Search logic lives in src/football_search/search.py,
and the index in chromadb/ is built offline with `make all` (never re-embedded here).
"""

from __future__ import annotations

import base64
import html
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Arc, Rectangle  # noqa: E402

from football_search import stats  # noqa: E402
from football_search.config import configure_logging, load_config  # noqa: E402
from football_search.search import Filters, Result, Searcher, load_manifest  # noqa: E402

configure_logging()
log = logging.getLogger("app")

CFG = load_config()
MANIFEST = load_manifest(CFG.chroma_dir)
SEARCHER = Searcher.from_manifest(CFG)  # models and index load once, at startup
SHOTS = pd.DataFrame(SEARCHER.collection.get(include=["metadatas"])["metadatas"])
TOURNAMENTS = sorted(SHOTS["tournament"].unique())
TEAMS = sorted(SHOTS["team"].unique())

# (query, goals only). Goal examples switch on the filter: embeddings find the right kind
# of shot well but are weaker at telling scored from missed, which is what filters are for.
EXAMPLES = [
    ["penalty saved by the goalkeeper", False],
    ["headed goal from a corner", True],
    ["long-range shot into the top corner", True],
    ["first-time volley after a cross", True],
    ["chip over the goalkeeper", True],
    ["Messi penalty in the final", False],
]

OUTCOME_STYLE = {
    "Goal": ("Goal", "#15803d"),
    "Saved": ("Saved", "#1d4ed8"),
    "Saved to Post": ("Saved onto post", "#1d4ed8"),
    "Saved Off Target": ("Saved wide", "#1d4ed8"),
    "Post": ("Hit the woodwork", "#b45309"),
    "Blocked": ("Blocked", "#6b7280"),
    "Off T": ("Off target", "#6b7280"),
    "Wayward": ("Wayward", "#6b7280"),
}

CSS = """
.card {border: 1px solid var(--border-color-primary); border-radius: 10px;
       padding: 12px 14px; margin-bottom: 10px; background: var(--block-background-fill);}
.card-head {display: flex; justify-content: space-between; gap: 8px; flex-wrap: wrap;
            font-size: 0.85rem; color: var(--body-text-color-subdued);}
.card-title {font-weight: 600; font-size: 1.02rem; margin: 4px 0 6px;}
.badge {display: inline-block; color: white; border-radius: 999px; padding: 1px 9px;
        font-size: 0.78rem; font-weight: 600; margin-right: 6px;}
.card-text {font-size: 0.92rem; line-height: 1.45;}
.card-foot {font-size: 0.8rem; color: var(--body-text-color-subdued); margin-top: 6px;}
.footer {font-size: 0.82rem; color: var(--body-text-color-subdued); text-align: center;}
.footer img {height: 28px; display: block; margin: 6px auto;}
"""


# --- Rendering ------------------------------------------------------------------------


def result_card(rank: int, r: Result) -> str:
    m = r.metadata
    label, color = OUTCOME_STYLE.get(m.get("outcome", ""), (m.get("outcome", ""), "#6b7280"))
    when = "Shootout" if m.get("is_shootout") else f"{m.get('minute')}'"
    esc = html.escape
    return f"""
<div class="card">
  <div class="card-head">
    <span>#{rank} · {esc(m.get("tournament", ""))} · {esc(m.get("stage", ""))} ·
      {esc(m.get("home", ""))} vs {esc(m.get("away", ""))} · {esc(m.get("match_date", ""))}</span>
    <span>similarity {r.score:.2f}</span>
  </div>
  <div class="card-title"><span class="badge" style="background:{color}">{esc(label)}</span>
    {esc(m.get("player", ""))} ({esc(m.get("team", ""))}) · {when}</div>
  <div class="card-text">{esc(m.get("description", r.document))}</div>
  <div class="card-foot">xG {m.get("xg", 0):.2f} · {esc(m.get("body_part", ""))} ·
    {esc(m.get("technique", ""))} · {esc(m.get("play_pattern", ""))}</div>
</div>"""


def pitch_plot(results: list[Result]):
    """Half pitch (attacking upwards) with each result's shot location, numbered by rank."""
    fig, ax = plt.subplots(figsize=(4.6, 4.1))
    line = {"color": "#9ca3af", "lw": 1.2, "fill": False}
    ax.add_patch(Rectangle((0, 60), 80, 60, **line))
    ax.add_patch(Rectangle((18, 102), 44, 18, **line))  # penalty area
    ax.add_patch(Rectangle((30, 114), 20, 6, **line))  # six-yard box
    ax.add_patch(Rectangle((36, 120), 8, 1.5, **line))  # goal
    ax.add_patch(Arc((40, 108), 20, 20, theta1=217, theta2=323, color="#9ca3af", lw=1.2))
    ax.plot(40, 108, "o", color="#9ca3af", ms=2)
    for rank, r in enumerate(results, start=1):
        x, y = r.metadata.get("x"), r.metadata.get("y")
        if x is None or y is None:
            continue
        goal = r.metadata.get("is_goal")
        ax.scatter(y, x, s=170, color="#15803d" if goal else "white",
                   edgecolors="#15803d" if goal else "#374151", lw=1.4, zorder=3)  # fmt: skip
        ax.text(y, x, str(rank), ha="center", va="center", fontsize=7, zorder=4,
                color="white" if goal else "#111827")  # fmt: skip
    ax.set_xlim(-1, 81)
    ax.set_ylim(58, 123)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Shot locations (filled = goal)", fontsize=9, color="#6b7280")
    fig.tight_layout()
    return fig


def logo_html() -> str:
    data = base64.b64encode((ROOT / "assets" / "statsbomb-logo.png").read_bytes()).decode()
    m = MANIFEST
    info = (
        f"Index: {m['rows']:,} shots from {m['matches']} matches "
        f"({', '.join(m['tournaments'])}) · embeddings {m['embedding']['model']}"
        + (f" + reranker {m['rerank']['model']}" if m.get("rerank") else "")
        + f" · built {m['built_at'][:10]} from StatsBomb commit {m['source']['commit'][:7]}"
    )
    return f"""
<div class="footer">
  Data: StatsBomb Open Data<br>
  <a href="https://github.com/statsbomb/open-data" target="_blank">
    <img src="data:image/png;base64,{data}" alt="StatsBomb"></a><br>
  {html.escape(info)}
</div>"""


# --- Callbacks ------------------------------------------------------------------------


def run_search(query, tournaments, team, goals_only, min_xg, top_k):
    query = (query or "").strip()
    if not query:
        return "<p>Type what you want to find, or click an example.</p>", pitch_plot([])
    filters = Filters(
        tournaments=tuple(tournaments or ()),
        team=team or "",
        goals_only=bool(goals_only),
        min_xg=float(min_xg or 0),
    )
    results = SEARCHER.search(query, top_k=int(top_k), filters=filters)
    if not results:
        return "<p>No shots match these filters. Try removing some.</p>", pitch_plot([])
    cards = "".join(result_card(i, r) for i, r in enumerate(results, start=1))
    return cards, pitch_plot(results)


def run_stats(name, tournaments):
    return stats.run(stats.QUERIES[name], SHOTS, tournaments)


# --- Layout ---------------------------------------------------------------------------

with gr.Blocks(title="Football Moments Search") as demo:
    gr.Markdown(
        "# Football Moments Search\n"
        f"Search {MANIFEST['rows']:,} shots from major international tournaments in plain "
        "English. Each shot is described in words, embedded, and stored in a Chroma vector "
        "database; filters narrow the results by tournament, team, goals, or xG "
        "(expected goals: how likely a shot was to go in)."
    )
    with gr.Tab("Search"):
        with gr.Row():
            query = gr.Textbox(
                label="What are you looking for?",
                placeholder="e.g. headed goal from a corner in extra time",
                value=EXAMPLES[0][0],
                scale=5,
            )
            button = gr.Button("Search", variant="primary", scale=1)
        with gr.Accordion("Filters", open=False):
            with gr.Row():
                tournaments = gr.Dropdown(TOURNAMENTS, multiselect=True, label="Tournament")
                team = gr.Dropdown([""] + TEAMS, value="", label="Team")
            with gr.Row():
                goals_only = gr.Checkbox(label="Goals only")
                min_xg = gr.Slider(0, 1, value=0, step=0.05, label="Minimum xG")
                top_k = gr.Slider(1, CFG.max_top_k, value=CFG.top_k, step=1, label="Results")
        gr.Examples(EXAMPLES, inputs=[query, goals_only], label="Try an example")
        with gr.Row():
            with gr.Column(scale=3):
                results_html = gr.HTML()
            with gr.Column(scale=2):
                pitch = gr.Plot(label="Pitch")
        inputs = [query, tournaments, team, goals_only, min_xg, top_k]
        button.click(run_search, inputs, [results_html, pitch])
        query.submit(run_search, inputs, [results_html, pitch])
        demo.load(run_search, inputs, [results_html, pitch])  # show results on first visit
    with gr.Tab("Stats (SQL)"):
        gr.Markdown(
            "Quick summaries computed with DuckDB SQL over the same shot data. "
            "The queries are in `src/football_search/stats.py`."
        )
        with gr.Row():
            stat_name = gr.Radio(list(stats.QUERIES), value="Top scorers", label="Summary")
            stat_tournaments = gr.Dropdown(TOURNAMENTS, multiselect=True, label="Tournament")
        stat_table = gr.Dataframe(interactive=False)
        for control in (stat_name, stat_tournaments):
            control.change(run_stats, [stat_name, stat_tournaments], stat_table)
        demo.load(run_stats, [stat_name, stat_tournaments], stat_table)
    gr.HTML(logo_html())


if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft())
