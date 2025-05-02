import dash
from dash import dcc, html, Input, Output
import pandas as pd
import os
import plotly.graph_objs as go

# ---------- Load coordinate CSVs ----------
def load_coords(group, dim):
    path = f"coords/coords_{group}_{dim}dim.csv"
    return pd.read_csv(path)

coords = {
    "2d": {
        "all": load_coords("all", 2),
        "can": load_coords("can", 2),
        "eng": load_coords("eng", 2)
    },
    "3d": {
        "all": load_coords("all", 3),
        "can": load_coords("can", 3),
        "eng": load_coords("eng", 3)
    }
}

# ---------- Audio lookup ----------
df_full = pd.read_csv("item_list.csv")
df_full['label1'] = df_full['stim1'].apply(lambda f: f.split('_')[0] + '_' + f.split('_')[1])
df_full['label2'] = df_full['stim2'].apply(lambda f: f.split('_')[0] + '_' + f.split('_')[1])

def audio_for(label):
    rows = df_full[(df_full['label1'] == label) | (df_full['label2'] == label)]
    if rows.empty:
        return None
    r = rows.iloc[0]
    return r['stim1'] if label in r['label1'] else r['stim2']

# ---------- Helper to build figures ----------
def build_2d_figure(df):
    unique_spk = sorted(df['speaker'].unique())
    color_map = {s: f"hsl({i * 360 / len(unique_spk)},50%,50%)" for i, s in enumerate(unique_spk)}

    traces = []
    for lang, symbol in [('can', 'circle'), ('eng', 'square')]:
        lang_df = df[df['language'] == lang]
        hovertext = [
            f"{r['label']}<br>Speaker: {r['speaker']}<br>Language: {r['language']}<br>Coords: ({r['dim1']:.2f}, {r['dim2']:.2f})"
            for _, r in lang_df.iterrows()
        ]
        traces.append(go.Scatter(
            x=lang_df['dim1'],
            y=lang_df['dim2'],
            mode='markers+text',
            marker=dict(
                size=12,
                color=[color_map[s] for s in lang_df['speaker']],
                symbol=symbol,
                line=dict(width=1, color='black')
            ),
            text=lang_df['label'],
            textposition='top center',
            customdata=[audio_for(l) for l in lang_df['label']],
            hoverinfo='text',
            hovertext=hovertext,
            name=lang
        ))

    return go.Figure(data=traces, layout=go.Layout(
        template='plotly_white',
        showlegend=False,
        xaxis=dict(title='Dimension 1', range=[-0.8, 0.8]),
        yaxis=dict(title='Dimension 2', range=[-0.8, 0.8]),
        margin=dict(l=40, r=40, t=40, b=40),
        height=600,
        font=dict(family="Arial, sans-serif")
    ))

def build_3d_figure(df):
    unique_spk = sorted(df['speaker'].unique())
    color_map = {s: f"hsl({i * 360 / len(unique_spk)},50%,50%)" for i, s in enumerate(unique_spk)}

    traces = []
    for lang, symbol in [('can', 'circle'), ('eng', 'square')]:
        lang_df = df[df['language'] == lang]
        hovertext = [
            f"{r['label']}<br>Speaker: {r['speaker']}<br>Language: {r['language']}<br>Coords: ({r['dim1']:.2f}, {r['dim2']:.2f}, {r['dim3']:.2f})"
            for _, r in lang_df.iterrows()
        ]
        traces.append(go.Scatter3d(
            x=lang_df['dim1'],
            y=lang_df['dim2'],
            z=lang_df['dim3'],
            mode='markers+text',
            marker=dict(
                size=6,
                color=[color_map[s] for s in lang_df['speaker']],
                symbol=symbol,
                line=dict(width=1, color='black')
            ),
            text=lang_df['label'],
            textposition='top center',
            customdata=[audio_for(l) for l in lang_df['label']],
            hoverinfo='text',
            hovertext=hovertext,
            name=lang
        ))

    return go.Figure(data=traces, layout=go.Layout(
        showlegend=False,
        scene=dict(
            xaxis=dict(title='Dimension 1', range=[-0.8, 0.8]),
            yaxis=dict(title='Dimension 2', range=[-0.8, 0.8]),
            zaxis=dict(title='Dimension 3', range=[-0.8, 0.8])
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=600,
        font=dict(family="Arial, sans-serif")
    ))

# ---------- Dash App ----------
app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1("Perceived voice similarity", style={"marginTop": "10px"}),
    html.P("Interactive maps from multidimensional scaling (MDS)"),
    
    html.Div([
        dcc.Tabs(id='dim-tabs', value='2d', children=[
            dcc.Tab(label='2D MDS', value='2d'),
            dcc.Tab(label='3D MDS', value='3d')
        ])
    ], style={"width": "600px", "margin": "20px auto"}),

    html.Div(id='listener-tabs-wrapper'),

    html.Div(id='audio-output', style={'marginTop': '20px'})
], style={"fontFamily": "Arial, sans-serif", "textAlign": "center"})

# ---------- Callbacks ----------
@app.callback(
    Output('listener-tabs-wrapper', 'children'),
    Input('dim-tabs', 'value')
)
def update_tabs(dim):
    fig_func = build_2d_figure if dim == '2d' else build_3d_figure
    return dcc.Tabs(children=[
        dcc.Tab(label='All listeners', children=[
            dcc.Graph(id='g-all', figure=fig_func(coords[dim]['all']))
        ]),
        dcc.Tab(label='Cantonese-English listeners', children=[
            dcc.Graph(id='g-can', figure=fig_func(coords[dim]['can']))
        ]),
        dcc.Tab(label='English listeners', children=[
            dcc.Graph(id='g-eng', figure=fig_func(coords[dim]['eng']))
        ])
    ], style={"width": "900px", "margin": "0 auto"})

@app.callback(
    Output('audio-output', 'children'),
    Input('g-all', 'clickData'),
    Input('g-can', 'clickData'),
    Input('g-eng', 'clickData')
)
def play_audio(c_all, c_can, c_eng):
    click = next((c for c in [c_all, c_can, c_eng] if c and c.get("points")), None)
    if not click:
        return html.Div("Click a point to see details and play audio")

    audio_file = click["points"][0]["customdata"]
    if not audio_file or not os.path.exists(f"./assets/{audio_file}"):
        return html.Div("Audio file not found")

    return html.Audio(src=f"/assets/{audio_file}", controls=True, autoPlay=True, style={"width": "400px"})

# ---------- Run ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=True)