import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import datetime
import numpy as np
from tqdm import tqdm
from audioset_tagging_cnn.config import labels
import argparse


parser = argparse.ArgumentParser(
    description='Export pkl to hdf5 files')
parser.add_argument('--file', type=str, help='pkl file')
parser.add_argument('--out', type=str, help='output name')


args = parser.parse_args()

df = pd.read_pickle(args.file)
for idx, k in enumerate(df['datetime']) :
    df['datetime'][idx] = datetime.datetime.strptime(k, '%Y%m%d_%H%M%S')

keys_ = list(df.keys())[:-3]
df = pd.DataFrame(df)
df_indic = df[keys_]
df_indic.to_csv(f'{args.out}_indicateurs.csv')
df_indic.to_hdf(f'{args.out}_indicateurs.h5', key='df_indic', mode='w', complevel=9)

clipwise = np.zeros((len(df['clipwise_output']), 527))
for k in tqdm(range(len(df))):
    clipwise[k,:] = df['clipwise_output'][k]

DF = pd.DataFrame(clipwise, columns=labels)
DF['datetime'] = df['datetime']
DF['name'] = df['name']
DF.to_hdf(f'{args.out}_proba.h5', key='DF', mode='w', complevel=9)
DF.to_csv(f'{args.out}_proba.csv')

# fig = go.Figure(go.Scatter(x = df['datetime'], y = df['BI']))
# fig.add_trace(go.Scatter(x = df['datetime'], y = df['BI_filt']))
# # fig = px.scatter(df, x = 'datetime', y = 'aci')
# fig.show()
