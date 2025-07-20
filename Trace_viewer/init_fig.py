import plotly.graph_objects as go
import numpy as np
import plotly.express as px

def init_fig():

    fig = go.FigureWidget()  
    hovertemplate = "index: %{pointNumber}<br>" +  "t: %{x}<br>" + "y: %{y}<extra></extra>" 

    fig.add_trace(go.Scattergl(x = [], y=[], xaxis='x1', yaxis='y1', name='fret_g', mode='lines', line = {'color': '#013220'}, hovertemplate = hovertemplate)) #0
    fig.add_trace(go.Scattergl(x = [], y=[], xaxis='x1', yaxis='y2', name='fret_b', mode='lines', line = {'color': '#00008B'}, hovertemplate = hovertemplate)) #1

    
    fig.add_trace(go.Scattergl(x = [], y =[], xaxis='x1', yaxis='y4', name='bb', mode='lines', line = {'color': 'blue'}, hovertemplate = hovertemplate)) #2
    fig.add_trace(go.Scattergl(x = [], y =[], xaxis='x1', yaxis='y4', name='bg', mode='lines', line = {'color': 'green'}, hovertemplate = hovertemplate)) #3
    fig.add_trace(go.Scattergl(x = [], y =[], xaxis='x1', yaxis='y4', name='br', mode='lines', line = {'color': 'red'}, hovertemplate = hovertemplate)) #4
    fig.add_trace(go.Scattergl(x = [], y = [], xaxis='x1',yaxis='y4',name='tot_b', mode='lines', line = {'color': 'black'}, hovertemplate = hovertemplate)) #5


    fig.add_trace(go.Scattergl(x = [], y = [], xaxis='x1', yaxis='y3', name='gg', mode='lines', line = dict(color = 'green'), hovertemplate = hovertemplate))  #6
    fig.add_trace(go.Scattergl(x = [], y = [], xaxis='x1', yaxis='y3', name='gr', mode='lines', line = dict(color = 'red'), hovertemplate = hovertemplate)) #7
    fig.add_trace(go.Scattergl(x = [], y = [],xaxis='x1', yaxis='y3',name='tot_g', mode='lines', line = dict(color = 'black'), hovertemplate = hovertemplate)) #8

    fig.add_trace(go.Scattergl(x = [], y = [],xaxis='x1', yaxis='y3',name='rr', mode='lines', line = dict(color = 'orange'), hovertemplate = hovertemplate)) #9



    fig.add_trace(go.Scatter(x = [], y = [], xaxis='x1', yaxis='y1', name = 'fret_g_bkps', mode = 'markers', marker = dict(color='red', size=10))) #10
    fig.add_trace(go.Scatter(x = [], y = [], xaxis='x1', yaxis='y2', name='fret_b_bkps', mode='markers', marker = dict(color='red', size=10))) #11
    fig.add_trace(go.Scatter(x = [], y = [], xaxis='x1', yaxis='y4', name='b_bkps', mode='markers', marker = dict(color='red', size=10))) #12
    fig.add_trace(go.Scatter(x = [], y = [], xaxis='x1', yaxis='y3', name='g_bkps', mode='markers', marker = dict(color='red', size=10))) #13
    fig.add_trace(go.Scatter(x = [], y = [], xaxis='x1', yaxis='y3', name='r_bkps', mode='markers', marker = dict(color='red', size=10))) #14

    fig.add_trace(go.Scattergl(x = [], y = [], xaxis='x1', yaxis='y1',name='HMM_fret_g', mode='lines', line = dict(color = 'orange'))) #15

    fig.add_histogram(y = [], xaxis='x2',  yaxis='y1', name='Histogram_g', histnorm='probability density', marker = dict(color='#1f77b4', opacity=0.7)) #16
    fig.add_histogram(y = [], xaxis='x2',  yaxis='y2', name='Histogram_b', histnorm='probability density', marker = dict(color='#1f77b4', opacity=0.7)) #17
    #fig.add_trace(go.Image(z = []))


        
    fig.layout = dict(xaxis1 = dict(domain = [0, 0.9]),
                    margin = dict(t = 120),
                    hovermode = 'closest',
                    bargap = 0,
                    uirevision = True,
                    xaxis2 = dict(domain = [0.9, 1]),
                    yaxis1 = dict(domain = [0.01, 0.25]),
                    yaxis2 = dict(domain = [0.26, 0.5]),
                    yaxis3 = dict(domain = [0.51, 0.75]),
                    yaxis4 = dict(domain = [0.76, 1.00]),
                    height = (1000),
                    annotations=[
                        dict(
                            text="Trace 0",  # <-- Your title text
                            xref="paper", yref="paper",
                            x=0.5, y=1.02,  # Centered above the entire figure
                            xanchor='center', yanchor='top',
                            showarrow=False,
                            font=dict(size=20, family='Arial')
                        )
                    ],
                    )

    fig.update_layout(
        xaxis1= dict(
            showline = False,
            showgrid = False,
            showticklabels = True,
            showspikes = True,
            ticks = 'inside',
            range = (0, 360)
        ),
        
        yaxis1 = dict(
            showgrid = True,
            zeroline=True,
            showline=True,
            showticklabels=True,
            griddash='longdash',
            dtick = 0.1,
            showspikes=True,
            autorange = False,
            range = [0, 1]

        ),

        yaxis2 = dict(
            showgrid = True,
            zeroline=True,
            showline=True,
            showticklabels=True,
            griddash='longdash',
            dtick = 0.1,
            showspikes=True,
            autorange = False,
            range = [0, 1]

        ),
        
        yaxis3 = dict(
            showgrid=True,
            zeroline=True,
            showline=True,
            showticklabels=True,
            griddash = 'longdash',
            showspikes=True,
            range = (0, None),
        ),
        
        yaxis4=dict(
            showgrid=True,
            zeroline=True,
            showline=True,
            showticklabels=True,
            griddash = 'longdash',
            showspikes=True,
            range = (0, None),
        ),
        
        autosize = True,
        showlegend = False,
        # hovermode =  "x unified",
        xaxis_title='time (s)',
        yaxis_title='FRET',
        margin=dict(l=10, r=10, t=10, b=10)
    )

   

    fig_blob = px.imshow(np.zeros((9, 54)), zmin = 0, zmax = 128)
    fig_blob.update_layout(
    xaxis=dict(
        showline=True,
        range=(-0.5, 53.5),
        showticklabels = False,
        autorange =  False,
    ),
    
    yaxis=dict(
        showline=True,
        range = (-0.5, 8.5),
        showticklabels = False, 
        autorange =  False
    ),
    width = 2400, 
    height = 400,
    #autosize = True,
    uirevision = True,
    coloraxis_showscale = False,
    margin=dict(l=20, r=20, t=20, b=20)
)
    for i in range (1, 6):
        fig_blob.add_vline(x = 9 * i - 0.5)

    fig_blob.add_annotation(x = 4, y = 7, showarrow = False, text = "BB", font = dict(size=16, color = "#ADD8E6"))
    fig_blob.add_annotation(x = 13, y = 7, showarrow = False, text = "BG", font = dict(size=16, color = "#ADD8E6"))
    fig_blob.add_annotation(x = 22, y = 7, showarrow = False, text = "BR", font = dict(size=16, color = "#ADD8E6"))
    fig_blob.add_annotation(x = 31, y = 7, showarrow = False, text = "GG", font = dict(size=16, color = "#90EE90"))
    fig_blob.add_annotation(x = 40, y = 7, showarrow = False, text = "GR", font = dict(size=16, color = "#90EE90"))
    fig_blob.add_annotation(x = 49, y = 7, showarrow = False, text = "RR", font = dict(size=16, color = "#FFCCCB"))

    

    fig2 = go.FigureWidget()
    fig2.add_histogram(x = [] , name='gmm_hist', histnorm='probability density', marker=dict(color='#f7e1a0', opacity=0.7),
                    xbins = dict(start = 0,end = 1, size=0.02),
                    cumulative = dict(enabled=False))


    fig2.update_layout(
        xaxis=dict(
            showline=False,
            showgrid=False,
            showticklabels=True,
            showspikes=True,
            ticks='inside',
            dtick= 0.1,
            range=(0,1)
        ),
        yaxis=dict(rangemode = 'tozero'),
        autosize=True,
        showlegend=False,
        xaxis_title='FRET',
        margin=dict(l=10, r=10, t=10, b=10)
    )
    
    return fig, fig_blob, fig2
