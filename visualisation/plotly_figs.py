import plotly.express as px
import plotly.graph_objects as go

# mapbox_access_token = open(".mapbox_token").read()#"pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A"


# def get_map_fig(database):
#     fig = px.scatter_mapbox(database, lat="lat", lon="lng", color = "s_statsmean",hover_data=['partID', 'city', 'country', "recorder"], zoom=3, height=300, width=500)
#     fig.update_layout(
#         mapbox_style="open-street-map",autosize=True,showlegend=False,
#                 margin=go.layout.Margin(l=0, r=0, t=0, b=0))
#     fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
#     return fig



def get_map_fig(database):
    fig = px.scatter_mapbox(database, lat="lat", lon="lng", color = "s_statsmean",hover_data=['partID', 'city', 'country', "recorder"], zoom=1, height=300, width=500, size_max=10)
    fig.update_layout(
        mapbox_style="open-street-map",autosize=True,showlegend=False,
                margin=go.layout.Margin(l=0, r=0, t=0, b=0))
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, coloraxis_colorbar=dict(
        title="<b>Indicateur</b>",titleside='right',
                        thicknessmode="pixels", thickness=10,
                        lenmode="pixels", len=250,
                        yanchor="bottom", y=0.1,
                        xanchor="right", x=1.0
                        ))
    return fig
