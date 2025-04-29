import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
from sklearn.manifold import MDS
import os

def extract_speaker_lang(filename):
    """Extract speaker + language from filename"""
    parts = filename.split('_')
    return f"{parts[0]}_{parts[1]}" if len(parts) > 1 else filename

def get_subject_columns(df, lang_type=None):
    """Get subject columns, optionally by language type ('C' for Cantonese, 'E' for English), or all"""
    if lang_type == 'C':
        return [col for col in df.columns if col.startswith('subjC')]
    elif lang_type == 'E':
        return [col for col in df.columns if col.startswith('subjE')]
    else:
        return [col for col in df.columns if col.startswith('subj')]

def compute_mds(df, subject_cols):
    """Compute dissimilarity and perform MDS analysis based on subject columns"""
    df['label1'] = df['stim1'].apply(extract_speaker_lang)
    df['label2'] = df['stim2'].apply(extract_speaker_lang)
    labels = sorted(set(df['label1']).union(set(df['label2'])))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    dissimilarity_matrix = np.zeros((n, n))
    for _, row in df.iterrows():
        i = label_to_idx[row['label1']]
        j = label_to_idx[row['label2']]
        diss = np.mean([float(row[subj]) for subj in subject_cols])
        dissimilarity_matrix[i, j] = diss
        dissimilarity_matrix[j, i] = diss
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
    coords = mds.fit_transform(dissimilarity_matrix)
    return coords, labels

def get_plot_info(df, labels):
    """Return speaker, language, and audio filename for each point"""
    speaker = [lbl.split('_')[0] for lbl in labels]
    lang = [lbl.split('_')[1] for lbl in labels]
    audio_files = []
    for lbl in labels:
        rows = df[(df['label1'] == lbl) | (df['label2'] == lbl)]
        if not rows.empty:
            row = rows.iloc[0]
            if extract_speaker_lang(row['stim1']) == lbl:
                audio_files.append(row['stim1'])
            else:
                audio_files.append(row['stim2'])
        else:
            audio_files.append(None)
    return speaker, lang, audio_files

# Precompute overall range for all plots
df_full = pd.read_csv("mds_data.csv")

subject_cols_all = get_subject_columns(df_full)
subject_cols_c = get_subject_columns(df_full, 'C')
subject_cols_e = get_subject_columns(df_full, 'E')

coords_all, labels_all = compute_mds(df_full, subject_cols_all)
coords_c, labels_c = compute_mds(df_full, subject_cols_c)
coords_e, labels_e = compute_mds(df_full, subject_cols_e)

# Find global axis limits
all_coords = np.vstack([coords_all, coords_c, coords_e])
x_min, x_max = all_coords[:, 0].min() - 0.5, all_coords[:, 0].max() + 0.5
y_min, y_max = all_coords[:, 1].min() - 0.5, all_coords[:, 1].max() + 0.5

def get_figure(coords, labels, title):
    speakers, langs, audio_files = get_plot_info(df_full, labels)
    color_vals = pd.Series(speakers).astype('category').cat.codes
    hover_text = [
        f"{lbl}<br>Speaker: {spk}<br>Language: {lng}" for lbl, spk, lng in zip(labels, speakers, langs)
    ]

    can_idx = [i for i, l in enumerate(langs) if l == 'can']
    eng_idx = [i for i, l in enumerate(langs) if l == 'eng']

    figure = {
        "data": [
            {
                "x": coords[can_idx, 0],
                "y": coords[can_idx, 1],
                "mode": "markers+text",
                "marker": {
                    "symbol": "circle",
                    "size": 15,
                    "color": color_vals[can_idx],
                    "colorscale": "Viridis",
                    "line": {"width": 1, "color": "black"},
                },
                "text": [labels[i] for i in can_idx],
                "textposition": "top center",
                "customdata": [audio_files[i] for i in can_idx],
                "hoverinfo": "text",
                "hovertext": [hover_text[i] for i in can_idx],
                # no name field -> no legend entry
            },
            {
                "x": coords[eng_idx, 0],
                "y": coords[eng_idx, 1],
                "mode": "markers+text",
                "marker": {
                    "symbol": "square",
                    "size": 15,
                    "color": color_vals[eng_idx],
                    "colorscale": "Viridis",
                    "line": {"width": 1, "color": "black"},
                },
                "text": [labels[i] for i in eng_idx],
                "textposition": "top center",
                "customdata": [audio_files[i] for i in eng_idx],
                "hoverinfo": "text",
                "hovertext": [hover_text[i] for i in eng_idx],
                # no name field -> no legend entry
            },
        ],
        "layout": {
            "xaxis": {"title": "Dimension 1", "range": [x_min, x_max]},
            "yaxis": {"title": "Dimension 2", "range": [y_min, y_max]},
            "margin": dict(l=60, r=60, b=60, t=60),
            "width": 800,
            "height": 600,
            "title": title,
            "showlegend": False  # <- no legend
        }
    }
    return figure

# Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H3("MDS results (Click a point to play audio)"),
    dcc.Tabs([
        dcc.Tab(label="All participants", children=[
            dcc.Graph(id="mds-scatter-all", figure=get_figure(coords_all, labels_all, "All participants")),
            html.Div(id="audio-player-all")
        ]),
        dcc.Tab(label="Cantonese-English participants", children=[
            dcc.Graph(id="mds-scatter-c", figure=get_figure(coords_c, labels_c, "Cantonese-English participants")),
            html.Div(id="audio-player-c")
        ]),
        dcc.Tab(label="English participants", children=[
            dcc.Graph(id="mds-scatter-e", figure=get_figure(coords_e, labels_e, "English participants")),
            html.Div(id="audio-player-e")
        ]),
    ])
])

@app.callback(
    Output("audio-player-all", "children"),
    Input("mds-scatter-all", "clickData")
)
def update_audio_all(clickData):
    return play_audio_from_click(clickData)

@app.callback(
    Output("audio-player-c", "children"),
    Input("mds-scatter-c", "clickData")
)
def update_audio_c(clickData):
    return play_audio_from_click(clickData)

@app.callback(
    Output("audio-player-e", "children"),
    Input("mds-scatter-e", "clickData")
)
def update_audio_e(clickData):
    return play_audio_from_click(clickData)

def play_audio_from_click(clickData):
    if clickData is None or clickData["points"][0]["customdata"] is None:
        return html.Div("Please click a point to play audio.")
    audio_file = clickData["points"][0]["customdata"]
    if not os.path.exists(f"./assets/{audio_file}"):
        return html.Div(f"Audio file not found: {audio_file}")
    return html.Audio(src=f"/assets/{audio_file}", controls=True, autoPlay=True, style={"width": "400px"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=True)