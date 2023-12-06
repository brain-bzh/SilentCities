import sys
import pandas as pd
import numpy as np
from audioset_tagging_cnn.config import labels
import datetime
from matplotlib import pyplot as plt 
import plotly.graph_objects as go

color_plot = {'Silence' : '#000000',
              'Vehicle' : '#c72323',
              'Rain' : '#2b6fc3',
              'Bee, wasp, etc.' : '#22c51e',
              'Wind noise (microphone)' : '#1e36c5',
              'Caw' : '#00e54b',
              'Bird vocalization, bird call, bird song' : '#4edc00',
              'Applause' : '#d42adf'}

fewlabels = ['Applause',
'Bird vocalization, bird call, bird song',
'Caw',
'Bee, wasp, etc.',
'Wind noise (microphone)','Rain','Vehicle','Silence']

def subset_probas(Df,search_labels):
    allprobas = np.stack(Df['clipwise_output'])

    ind_list = []
    for curlabel in search_labels:
        ind_list.append(int(np.argwhere([c==curlabel for c in labels])))

    return allprobas[:,ind_list]


def average_proba_over_freq(Df,freq_str='D',subset_labels=fewlabels):
    """
    Calculates probability density estimates over a configurable frequency

    arguments :
    Df : DataFrame, output of tag_silentcities
    freq_str : Frequency over which to calculate probability density estimate (default : days)
    subset_labels : Subset of labels from the Audioset Ontology to be used for the estimate. 
    default labels are :
    'Applause','Bird vocalization, bird call, bird song','Chirp, tweet','Pigeon, dove',
    'Caw','Bee, wasp, etc.','Wind noise (microphone)','Rain','Vehicle','Emergency vehicle','Rail transport',
    'Aircraft','Silence'

    outputs : 
    
    probas_agg : Probability Density estimates of the subset of labels calculate according to the frequency specified

    """
    #Let's use the datetime (Timestamp) as a new index for the dataframe
    ts = pd.DatetimeIndex([datetime.datetime.strptime(dd,"%Y%m%d_%H%M%S") for dd in Df['datetime']])

    

    # Let's add the Labels from the shortlist as entries in the Df. Will be easier to manipulate them afterwards
    
    prob = subset_probas(Df,subset_labels)

    newDf = pd.DataFrame()

    
    for f,curlabel in enumerate(subset_labels):
        newDf[curlabel] = prob[:,f]

    newDf.index = ts

    # Now let's create a period range to easily compute statistics over days (frequency can be changed by changing the freq_str argument)
    prng = pd.period_range(start=ts[0],end=ts[-1], freq=freq_str).astype('datetime64[ns]')
    

    # And now create the final DataFrame that averages probabilities (of labels subset_labels) with the frequency defined in freq_str

    allser = dict()

    for lab in fewlabels:

        curser = pd.Series([newDf[prng[i]:prng[i+1]][lab].mean() for i in range(len(prng)-1)],index=prng[:-1])
        
        allser[lab] = curser
        
    probas_agg = pd.DataFrame(allser)

    return probas_agg


def make_interactive_pdf(Df,list_resolutions = ['0.25H','H','3H','6H','12H','D'],active_beg = 4,subset_labels=fewlabels):
    
    fig = go.Figure()

     ## which resolution is active when starting

    ### loop on resolution 

    for cur_res in list_resolutions:
        print(cur_res)

        probas_agg = average_proba_over_freq(Df,freq_str=cur_res,subset_labels=fewlabels)

        datelabel = probas_agg.index

        # Create figure

        for curcol in probas_agg.columns:

            fig.add_trace(go.Scatter(x=datelabel, y=probas_agg[curcol],name=curcol,line=dict(color=color_plot[curcol]),visible=False))

            
    nbcol = len(probas_agg.columns)

    # Make one resolution visible trace visible
    for curcol,_ in enumerate(probas_agg.columns):

        fig.data[active_beg*nbcol+curcol].visible = True


    # Create and add slider
    steps = []
    for i in range(len(list_resolutions)): 
        step = dict(
            method="restyle",
            label=list_resolutions[i],
            args=["visible", [False] * len(fig.data)],
        )
        
        for curcol,_ in enumerate(probas_agg.columns):
            step["args"][1][i*nbcol+curcol] = True  # Toggle trace to "visible"
            steps.append(step)
        
    sliders = [dict(
        active=active_beg*nbcol,
        pad={"t": len(list_resolutions)},
        #currentvalue={"prefix": "Resolution: "},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )
    return fig

filename = sys.argv[1]
savepath = filename[:-3] + 'html'

print(savepath)

Df = pd.read_pickle(sys.argv[1])
#fig = analysis.heatmap_probas(Df,search_labels=fewlabels,nbannot = 10)
#fig.write_html(pathname+'1.html')
fig = make_interactive_pdf(Df)

fig.write_html(savepath)
