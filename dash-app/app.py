import sys
sys.path.insert(0, "../")

import os 
from glob import glob
# import argparse

import dash
# import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
# from dash_html_components.Div import Div
# from librosa.core import audio
# import plotly.graph_objs as go
# import plotly.express as px
import numpy as np
import pandas as pd
import visualisation.plotly_figs as pf
# from pydub import AudioSegment
from datetime import datetime
import base64

##### Path
PATH_MP3 = '/home/nfarrugi/bigdisk2/mp3_sl/'
PATH_DATABASE = "/home/nfarrugi/SilentCities/database/public_final_metadata_geo_stats.csv"
PATH_TAGSITE = "/home/nfarrugi/bigdisk2/meta_silentcities/site/"

#### Initialization
database = pd.read_csv(PATH_DATABASE)

# get available site
available_site = glob(os.path.join(PATH_MP3, '*/'))
available_site = np.sort([int(i[-5:-1]) for i in available_site])
database = database[database['partID'].isin(available_site)].reset_index(drop=True)


# Initialization first fig
current_partID = available_site[0]
figmap = pf.get_map_fig(database)
figindic, data = pf.get_heatmaps(available_site[0], path=PATH_TAGSITE)
wavefig, path_current_audio = pf.get_sample_fig(available_site[0], f"{data['name'][0][:-4]}_{int(data['start'][0])}.mp3", path=PATH_MP3)
encoded_sound = base64.b64encode(open(path_current_audio, 'rb').read())


# styles
table_header_style = {
    "backgroundColor": "rgb(2,21,70)",
    "color": "white",
    "textAlign": "center",
}

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }}
app.title = "SilentCities"
server = app.server

##### Layout
app.layout = html.Div(
    className="",
    children=[
        html.Div(
            className="header",
            children=[
                # html.A(
                #     id="link silent",
                #     children=["Silent Cities project"],
                #     href="https://osf.io/h285u/",
                #     style={"color": "white"},
                # ),   
                html.H2("Silent Cities Project : Data Analysis", style={"color": "white"}),
                # html.A(
                #     id="gh-link",
                #     children=["Source code Silent Cities"],
                #     href="https://github.com/brain-bzh/SilentCities",
                #     style={"color": "white", "border": "solid 1px white"},
                # ),    
            ],
            style={"backgroundColor": "rgb(2,21,70)", "textAlign": "center"}
        ),
        
    html.Div(
        className='tot',
            children=[
                html.Div(
                    className="row",
                    children=[
                        html.Div(
                            className="one-third column alpha",
                            children=[
                                html.Div(className="row",children = [
                                dcc.Graph(id='map', figure = figmap),
                                html.H4(children=['Site informations :'],style={"color": "blue"}),
                                html.H6(
                                    id="map-info",
                                    children=[f"Site: {database['partID'][0]}", html.Br(),
                                            f"Country: {database['country'][0]}", html.Br(),
                                            f"City : { database['city'][0]}",html.Br(),
                                            f"Recorder : {database['recorder'][0]}",html.Br(),
                                            f"Stat : {database['s_statsmean'][0]:.3f}", html.Br(),
                                            f"Information : NA"],
                                    
                                    style={"color": "black", "border": "solid 1px white"},
                                ),    
                                ])
                            ]
                        ),
                        html.Div(
                            className="two-thirds column omega",
                            children=[dcc.Graph(id='heatmap', figure = figindic,  style={
                        "width": "100%"})]
                        ) 
                    ]
                ),
                html.Div(
                    className="row",
                    children=[
                    html.H6(
                                    id="wav-info",
                                    children=[f"Wave file informations : file name {data['name'][0]}, date : {datetime.strptime(data['datetime'][0], '%Y%m%d_%H%M%S')}, Geophony : {data['geophony'][0]}, Biophony {data['biophony'][0]}, Anthropophony {data['anthropophony'][0]}"],
                                    style={"color": "blue", "border": "solid 1px white"},
                                ),  
                    html.Audio(id="player", src='data:audio/mpeg;base64,{}'.format(encoded_sound.decode()), controls=True, style={
                        "width": "100%"}),
                    dcc.Graph(id='spactrogram', figure = wavefig, style={
                        "width": "100%"})
                    ]
                )
            ]
        )
    ]
)



##### call back

@app.callback([Output('heatmap', 'figure'),
                Output('map-info', 'children')],
    [Input('map', 'clickData')])
def Update_heatmap(clickData):
    global data
    global database
    global current_partID
    current_partID = clickData['points'][0]['customdata'][0]
    idx =  clickData['points'][0]['pointNumber']

    figindic, data = pf.get_heatmaps(current_partID, path=PATH_TAGSITE)


    text = [f"Site: {database['partID'][idx]}", html.Br(),
            f"Country: {database['country'][idx]}", html.Br(),
            f"City : { database['city'][idx]}",html.Br(),
            f"Recorder : {database['recorder'][idx]}",html.Br(),
            f"Stat : {database['s_statsmean'][idx]:.3f}", html.Br(),
            f"Information : NA"]
    print(text)
    return figindic, text



@app.callback([Output('spactrogram', 'figure'),
               Output('player', 'src'),
               Output('wav-info', 'children')],
    [         Input('heatmap', 'clickData')])
def Update_audio(clickData):
    global data
    global current_partID

    x = clickData['points'][0]['x']
    idx = data.index[data["datetime"]==x][0]

    
    wavefig, path_current_audio = pf.get_sample_fig(current_partID, f"{str(data['name'][idx])[:-4]}_{int(data['start'][idx])}.mp3", path=PATH_MP3)
    
    encoded_sound = base64.b64encode(open(path_current_audio, 'rb').read())
    src = 'data:audio/mpeg;base64,{}'.format(encoded_sound.decode())
    text = [f"Wave file informations : file name {str(data['name'][idx])[:-4]}_{int(data['start'][idx])}.mp3, date : {datetime.strptime(data['datetime'][idx], '%Y%m%d_%H%M%S')}, Geophony : {data['geophony'][idx]}, Biophony {data['biophony'][idx]}, Anthropophony {data['anthropophony'][idx]}"]

    return wavefig, src, text


    
if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1',port=os.getenv("PORT", "8051"))

