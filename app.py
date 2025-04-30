import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import os

# ---------- 1.  Load coordinates ----------
coords_all = pd.read_csv("coords_all_participants.csv")
coords_c   = pd.read_csv("coords_cantonese_english_participants.csv")
coords_e   = pd.read_csv("coords_english_participants.csv")

# ---------- 2.  Original trial file (audio lookup) ----------
df_full = pd.read_csv("mds_data.csv")
df_full['label1'] = df_full['stim1'].apply(lambda f: f.split('_')[0] + '_' + f.split('_')[1])
df_full['label2'] = df_full['stim2'].apply(lambda f: f.split('_')[0] + '_' + f.split('_')[1])

# ---------- 3.  Axis limits (identical for all plots) ----------
global_xy = pd.concat([coords_all[['x','y']], coords_c[['x','y']], coords_e[['x','y']]])
x_min, x_max = global_xy['x'].min() - .5, global_xy['x'].max() + .5
y_min, y_max = global_xy['y'].min() - .5, global_xy['y'].max() + .5

# ---------- 4.  Helpers ----------
def audio_for(label):
    rows = df_full[(df_full['label1'] == label) | (df_full['label2'] == label)]
    if rows.empty:
        return None
    r = rows.iloc[0]
    return r['stim1'] if label in r['label1'] else r['stim2']

def fig_from_df(df, title):
    can_mask = df['language'] == 'can'
    eng_mask = df['language'] == 'eng'
    spk_col  = df['speaker'].astype('category').cat.codes

    def make_trace(mask, symbol):
        return dict(
            x=df.loc[mask, 'x'],  y=df.loc[mask, 'y'],
            mode="markers+text",
            marker=dict(symbol=symbol, size=15, color=spk_col[mask],
                        colorscale="Viridis", line=dict(width=1, color="black")),
            text=df.loc[mask, 'label'],
            textposition="top center",
            customdata=[audio_for(lb) for lb in df.loc[mask, 'label']],
            hovertext=[f"{r.label}<br>Speaker: {r.speaker}<br>Language: {r.language}"
                       for _, r in df[mask].iterrows()],
            hoverinfo="text"
        )

    return dict(
        data=[make_trace(can_mask, "circle"), make_trace(eng_mask, "square")],
        layout=dict(
            title=title, showlegend=False,
            xaxis=dict(title="dimension 1", range=[x_min, x_max]),
            yaxis=dict(title="dimension 2", range=[y_min, y_max]),
            margin=dict(l=60, r=60, b=60, t=60),
            font=dict(family="Arial, sans-serif"),   # global font
            width=800, height=600
        )
    )

# ---------- 5.  Dash app ----------
app = dash.Dash(__name__)

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "textAlign": "center"},
    children=[
        html.H1("Perceived voice similarity"),
        html.P(["Interactive 2D maps of perceived voice similarity from multidimensional scaling (MDS)",
                html.Br(),
                "Click a point to see details and play audio"]),
        dcc.Tabs(
            style={"width": "860px", "margin": "0 auto"},   # center the tab headers
            children=[
                dcc.Tab(label="All listeners", children=[
                    html.Div(                          # <-- centering wrapper
                        children=[
                            dcc.Graph(
                                id="g-all",
                                figure=fig_from_df(coords_all, "all listeners"),
                                style={"margin": "0 auto"}  # center the plot
                            ),
                            html.Div(id="aud-all")
                        ],
                        style={"display": "flex",
                            "flexDirection": "column",
                            "alignItems": "center"}
                    )
                ]),
                dcc.Tab(label="Cantonese-English listeners", children=[
                    html.Div(
                        children=[
                            dcc.Graph(
                                id="g-c",
                                figure=fig_from_df(coords_c, "cantonese-english listeners"),
                                style={"margin": "0 auto"}
                            ),
                            html.Div(id="aud-c")
                        ],
                        style={"display": "flex",
                            "flexDirection": "column",
                            "alignItems": "center"}
                    )
                ]),
                dcc.Tab(label="English listeners", children=[
                    html.Div(
                        children=[
                            dcc.Graph(
                                id="g-e",
                                figure=fig_from_df(coords_e, "english listeners"),
                                style={"margin": "0 auto"}
                            ),
                            html.Div(id="aud-e")
                        ],
                        style={"display": "flex",
                            "flexDirection": "column",
                            "alignItems": "center"}
                    )
                ]),
            ]
        )   
    ])

# ---------- 6.  Callbacks ----------
def audio_player(click):
    if not click or click["points"][0]["customdata"] is None:
        return html.Div("click a point to play audio")
    wav = click["points"][0]["customdata"]
    if not wav or not os.path.exists(f"./assets/{wav}"):
        return html.Div("audio not found")
    return html.Audio(src=f"/assets/{wav}", controls=True, autoPlay=True, style={"width": "400px"})

@app.callback(Output("aud-all", "children"), Input("g-all", "clickData"))
def callback_all(c): return audio_player(c)

@app.callback(Output("aud-c",   "children"), Input("g-c",  "clickData"))
def callback_c(c):   return audio_player(c)

@app.callback(Output("aud-e",   "children"), Input("g-e",  "clickData"))
def callback_e(c):   return audio_player(c)

# ---------- 7.  Run ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=True)