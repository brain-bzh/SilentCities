import plotly.express as px
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description='Silent City map generator')
parser.add_argument('--database', type=str, help='path to database_process (pkl)')
parser.add_argument('--loc_info', type=str, help='path to location file (pkl)')
parser.add_argument('--html_name', default='maps.html',type=str, help='Name of the html output file')
args = parser.parse_args()

df = pd.read_pickle(args.loc_info)
df2 = pd.read_pickle(args.database)

df['moy'] = None
for idx, k in enumerate(df['moy']):
    if None in [df2['mar'][idx], df2['apr'][idx], df2['may'][idx], df2['jun'][idx], df2['jul'][idx], df2['aug'][idx]] :
        df['moy'][idx] = 0
    else:
        df['moy'][idx] = float((df2['mar'][idx]+df2['apr'][idx]+df2['may'][idx]+df2['jun'][idx]+df2['jul'][idx]+df2['aug'][idx])/5)


df['mar'] = df2['mar']
df['apr'] = df2['apr']
df['may'] = df2['may']
df['jun'] = df2['jun']
df['jul'] = df2['jul']
df['aug'] = df2['aug']
df['site_ID'] = df2['partID']
df['device_name'] = df2['The recording equipment you will be using:']

fig = px.scatter_mapbox(df, lat="lat", lon="lon", color = 'device_name', size = np.stack(df['moy']), hover_data=['device_name', 'moy', 'mar','apr', 'may', 'jun', 'jul', 'aug','site_ID'], zoom=2)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(legend=dict(
    title='Device',
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.write_html(f'{args.html_name}.html')
fig.show()