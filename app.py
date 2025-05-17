import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objs as go
import os

# ─────────────────────────  LOAD CAPTION  ─────────────────────────
with open("interactive_mds_caption.md", encoding="utf-8") as fh:
    CAPTION_MD = fh.read()

# ───────────────────────  DATA  ────────────────────────
def load_coords(stim, group, dim):
    fn = f"mds_results/coords_{stim}_stimuli_{group}_listeners_{dim}dim.csv"
    return pd.read_csv(fn)

stimuli_types   = ["all", "can", "eng", "mixed"]
listener_groups = ["all", "can", "eng"]
dims            = ["2d", "3d"]

coords = {
    d: {s: {g: load_coords(s, g, int(d[:-1])) for g in listener_groups}
        for s in stimuli_types}
    for d in dims
}

# ──────────────────  AUXILIARIES  ──────────────────────
def audio_for(row, stim):
    if stim == "mixed":
        return [f"{row['speaker']}_Can_utt1.wav", f"{row['speaker']}_Can_utt2.wav",
                f"{row['speaker']}_Eng_utt1.wav", f"{row['speaker']}_Eng_utt2.wav"]
    return [f"{row['speaker']}_{row['language']}_utt1.wav",
            f"{row['speaker']}_{row['language']}_utt2.wav"]

palette = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

def color_map(speakers):
    uniq = sorted(speakers.unique())
    return {spk: palette[i % len(palette)] for i, spk in enumerate(uniq)}

sym_map  = {"can": "circle", "eng": "square", "mixed": "diamond"}
name_map = {"can": "Cantonese", "eng": "English", "mixed": "Mixed"}

# ──────────────────  FIGURE BUILDER  ───────────────────
def build_figure(df, dim, stim):
    df = df.copy()
    df["label"] = df["speaker"] + "_" + (df["language"].str.capitalize()
                                         if stim != "mixed" else "Mixed")

    cm    = color_map(df["speaker"])
    size  = 12 if dim == "2d" or (dim == "3d" and stim == "mixed") else 6
    traces = []

    langs = ["mixed"] if stim == "mixed" else ["can", "eng"]
    for lang in langs:
        sub = df if lang == "mixed" else df[df["language"].str.lower() == lang]
        if sub.empty:
            continue

        hov = [f"{r['label']}<br>Speaker: {r['speaker']}<br>"
               f"Language: {name_map[lang]}<br>"
               f"Coords: ({r['dim1']:.3f}, {r['dim2']:.3f}"
               f"{', ' + format(r['dim3'], '.3f') if dim == '3d' else ''})"
               for _, r in sub.iterrows()]

        marker = dict(size=size,
                      color=[cm[s] for s in sub['speaker']],
                      symbol=sym_map[lang],
                      line=dict(width=1, color='black'))

        custom = [audio_for(r, stim) for _, r in sub.iterrows()]

        trace = (go.Scatter if dim == "2d" else go.Scatter3d)(
            x=sub['dim1'], y=sub['dim2'],
            **({} if dim == "2d" else dict(z=sub['dim3'])),
            mode="markers+text",
            marker=marker,
            text=sub['label'],
            textposition="bottom center",
            textfont=dict(size=24 if dim == "3d" and stim == "mixed" else 12),
            hoverinfo="text",
            hovertext=hov,
            customdata=custom,
            name=name_map[lang])

        traces.append(trace)

    layout = go.Layout(
        template="plotly_white",
        showlegend=False,
        height=600,
        margin=dict(l=40, r=40, t=40, b=40),
        font=dict(family="Arial, sans-serif")
    )

    if dim == "2d":
        layout.update(
            xaxis=dict(title="Dimension 1", range=[-0.8, 0.8]),
            yaxis=dict(title="Dimension 2", range=[-0.8, 0.8]))
    else:
        layout.update(scene=dict(
            xaxis=dict(title="Dimension 1", range=[-0.8, 0.8]),
            yaxis=dict(title="Dimension 2", range=[-0.8, 0.8]),
            zaxis=dict(title="Dimension 3", range=[-0.8, 0.8])
        ))

    return go.Figure(traces, layout)

# ─────────────────────  DASH APP  ──────────────────────
TAB_W = "900px"          # width for all tab rows
SIDE_W = "380px"         # sidebar width for caption

app = dash.Dash(__name__, suppress_callback_exceptions=True)

# ---------- LEFT-HAND interactive panel ----------
interactive_panel = html.Div([
    html.H1("Perceived voice similarity",
            style={"fontFamily": "Arial, sans-serif"}),

    dcc.Tabs(id="dim-tabs", value="2d", children=[
        dcc.Tab(label="2D MDS", value="2d"),
        dcc.Tab(label="3D MDS", value="3d")
    ], style={"width": TAB_W, "margin": "20px auto",
              "fontFamily": "Arial, sans-serif"}),

    html.Div(id="stim-tabs-wrapper", children=[
        dcc.Tabs(id="stim-tabs", value="all", children=[
            dcc.Tab(label="All stimuli",            value="all"),
            dcc.Tab(label="Cantonese stimuli",      value="can"),
            dcc.Tab(label="English stimuli",        value="eng"),
            dcc.Tab(label="Mixed-language stimuli", value="mixed")
        ])
    ]),

    html.Div(id="listener-tabs-wrapper"),

    html.Div(id="audio-output",
             style={"marginTop": "20px", "fontFamily": "Arial, sans-serif"})
], style={"flex": "1 1 auto", "paddingRight": "20px"})

# ---------- RIGHT-HAND caption panel ----------
caption_panel = html.Div(
    dcc.Markdown(CAPTION_MD,
                 style={"whiteSpace": "pre-wrap",
                        "overflowY": "auto",
                        "height": "95vh",
                        "padding": "20px",
                        "fontFamily": "Arial, sans-serif"}),
    style={"flex": f"0 0 {SIDE_W}",
           "borderLeft": "1px solid #ccc",
           "backgroundColor": "#f9f9f9"}
)

app.layout = html.Div([interactive_panel, caption_panel],
                      style={"display": "flex"})

# ───────  build stimulus-set layer  ───────
@app.callback(Output("stim-tabs-wrapper", "children"),
              Input("dim-tabs", "value"))
def make_stim_tabs(dim):
    return dcc.Tabs(id="stim-tabs", value="all", children=[
        dcc.Tab(label="All stimuli",            value="all"),
        dcc.Tab(label="Cantonese stimuli",      value="can"),
        dcc.Tab(label="English stimuli",        value="eng"),
        dcc.Tab(label="Mixed-language stimuli", value="mixed")
    ], style={"width": TAB_W, "margin": "0 auto",
              "fontFamily": "Arial, sans-serif"})

# ───────  build listener-group layer  ───────
@app.callback(Output("listener-tabs-wrapper", "children"),
              Input("dim-tabs",  "value"),
              Input("stim-tabs", "value"))
def make_listener_tabs(dim, stim):
    fig = lambda g: build_figure(coords[dim][stim][g], dim, stim)
    return dcc.Tabs(children=[
        dcc.Tab(label="All listeners",
                children=[dcc.Graph(id="g-all", figure=fig("all"))]),
        dcc.Tab(label="Cantonese-English listeners",
                children=[dcc.Graph(id="g-can", figure=fig("can"))]),
        dcc.Tab(label="English listeners",
                children=[dcc.Graph(id="g-eng", figure=fig("eng"))])
    ], style={"width": TAB_W, "margin": "0 auto",
              "fontFamily": "Arial, sans-serif"})

# ───────  audio playback  ───────
@app.callback(Output("audio-output", "children"),
              Input("g-all", "clickData"),
              Input("g-can", "clickData"),
              Input("g-eng", "clickData"))
def play_audio(cd_all, cd_can, cd_eng):
    click = next((c for c in (cd_all, cd_can, cd_eng)
                  if c and c.get("points")), None)
    if not click:
        return "Click a point to hear example audio"

    files = click["points"][0]["customdata"]
    players = [html.Audio(src=f"/assets/{f}", controls=True,
                          style={"width": "400px", "margin": "10px auto"})
               for f in files if os.path.exists(f"./assets/{f}")]
    return players or "Audio files not found"

# ─────────────────  run  ──────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=True)