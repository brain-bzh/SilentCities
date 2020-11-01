from plotly.subplots import make_subplots
import plotly.graph_objects as go
import argparse
import pandas as pd
from datetime import datetime

parser = argparse.ArgumentParser(
    description='Silent Cities export ecoacoustics indicators as interactive graph')
parser.add_argument('--path', type=str, help='path to pkl process')
parser.add_argument('--savename',default=None, type=str, help='save html')
args = parser.parse_args()

df = pd.read_pickle(args.path)
for idx, k in enumerate(df['datetime']) :
    df['datetime'][idx] = datetime.strptime(k, '%Y%m%d_%H%M%S')


indic = ['dB', 'ndsi', 'aci', 'nbpeaks', 'BI', 'EVN', 'ACT', 'EAS', 'ECV', 'EPS']
indic_filt = ['dB_filt', 'ndsi_filt', 'aci_filt', 'nbpeaks_filt', 'BI_filt', 'EVN_filt', 'ACT_filt', 'EAS_filt', 'ECV_filt', 'EPS_filt']
indic_tot = []
for k in range(20):
    if k%2 == 0:
        indic_tot.append(indic[int(k/2)])
    else :
        indic_tot.append(indic_filt[int((k-1)/2)])


fig = make_subplots(rows=10, cols=2, subplot_titles=indic_tot,shared_xaxes='all')
for idx, k in enumerate(indic): 
    fig.add_trace(go.Scatter(x=df['datetime'], y=df[k]),
              row=idx+1, col=1)
    fig.add_trace(go.Scatter(x=df['datetime'], y=df[k+'_filt']),
              row=idx+1, col=2)
fig.update(layout_showlegend=False)
if args.savename is not None:
    fig.write_html(args.savename+'.html')
fig.show()
