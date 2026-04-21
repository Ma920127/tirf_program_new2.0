# New version
import plotly.graph_objects as go
import numpy as np
import plotly.express as px

def init_fig():

    fig = go.FigureWidget()  
    hovertemplate = "index: %{pointNumber}<br>" +  "t: %{x}<br>" + "y: %{y}<extra></extra>" 

    # --- MAIN TRACES (Increased line width to 3 for thicker traces) ---
    # Legend 1 & yaxis 1 (Domain 1)
    fig.add_trace(go.Scattergl(x=[], y=[], xaxis='x1', yaxis='y1', name='fret_g', legend='legend', mode='lines', line={'color': '#013220', 'width': 3}, hovertemplate=hovertemplate)) #0
    
    # Legend 2 & yaxis 2 (Domain 2)
    fig.add_trace(go.Scattergl(x=[], y=[], xaxis='x3', yaxis='y2', name='fret_b', legend='legend2', mode='lines', line={'color': '#00008B', 'width': 3}, hovertemplate=hovertemplate)) #1
    
    # Legend 4 & yaxis 4 (Domain 4 - Top)
    fig.add_trace(go.Scattergl(x=[], y=[], xaxis='x5', yaxis='y4', name='bb', legend='legend4', mode='lines', line={'color': 'blue', 'width': 3}, hovertemplate=hovertemplate)) #2
    fig.add_trace(go.Scattergl(x=[], y=[], xaxis='x5', yaxis='y4', name='bg', legend='legend4', mode='lines', line={'color': 'green', 'width': 3}, hovertemplate=hovertemplate)) #3
    fig.add_trace(go.Scattergl(x=[], y=[], xaxis='x5', yaxis='y4', name='br', legend='legend4', mode='lines', line={'color': 'red', 'width': 3}, hovertemplate=hovertemplate)) #4
    fig.add_trace(go.Scattergl(x=[], y=[], xaxis='x5', yaxis='y4', name='tot_b', legend='legend4', mode='lines', line={'color': 'black', 'width': 3}, hovertemplate=hovertemplate)) #5

    # Legend 3 & yaxis 3 (Domain 3)
    fig.add_trace(go.Scattergl(x=[], y=[], xaxis='x4', yaxis='y3', name='gg', legend='legend3', mode='lines', line=dict(color='green', width=3), hovertemplate=hovertemplate))  #6
    fig.add_trace(go.Scattergl(x=[], y=[], xaxis='x4', yaxis='y3', name='gr', legend='legend3', mode='lines', line=dict(color='red', width=3), hovertemplate=hovertemplate)) #7
    fig.add_trace(go.Scattergl(x=[], y=[], xaxis='x4', yaxis='y3', name='tot_g', legend='legend3', mode='lines', line=dict(color='black', width=3), hovertemplate=hovertemplate)) #8
    fig.add_trace(go.Scattergl(x=[], y=[], xaxis='x4', yaxis='y3', name='rr', legend='legend3', mode='lines', line=dict(color='orange', width=3), hovertemplate=hovertemplate)) #9

    # --- BREAKPOINTS (Hidden from legend) ---
    marker_style = dict(color='#FF0000', size=12, line=dict(width=2, color='white'))
    fig.add_trace(go.Scatter(x=[], y=[], xaxis='x1', yaxis='y1', name='fret_g_bkps', showlegend=False, mode='markers', marker=marker_style)) #10
    fig.add_trace(go.Scatter(x=[], y=[], xaxis='x3', yaxis='y2', name='fret_b_bkps', showlegend=False, mode='markers', marker=marker_style)) #11
    fig.add_trace(go.Scatter(x=[], y=[], xaxis='x5', yaxis='y4', name='b_bkps', showlegend=False, mode='markers', marker=marker_style)) #12
    fig.add_trace(go.Scatter(x=[], y=[], xaxis='x4', yaxis='y3', name='g_bkps', showlegend=False, mode='markers', marker=marker_style)) #13
    fig.add_trace(go.Scatter(x=[], y=[], xaxis='x4', yaxis='y3', name='r_bkps', showlegend=False, mode='markers', marker=marker_style)) #14

    fig.add_trace(go.Scattergl(x=[], y=[], xaxis='x1', yaxis='y1',name='HMM_fret_g', legend='legend', mode='lines', line=dict(color='orange', width=3))) #15

    # --- HISTOGRAMS ---
    hist_marker = dict(color='#1f77b4', opacity=0.7, line=dict(width=1, color='black'))
    fig.add_histogram(y=[], xaxis='x2', yaxis='y1', name='Histogram_g', showlegend=False, histnorm='probability density', marker=hist_marker) #16
    fig.add_histogram(y=[], xaxis='x2', yaxis='y2', name='Histogram_b', showlegend=False, histnorm='probability density', marker=hist_marker) #17

    # --- LAYOUT & MATPLOTLIB STYLING ---
    mpl_axis_style = dict(
        showline=True,
        linewidth=1.5,
        linecolor='black',
        mirror=True,        
        showgrid=False,
        zeroline=False,
        showspikes=False,
        ticks='inside',     
        tickwidth=1.5,
        tickcolor='black',
        ticklen=5,
    )

    legend_style = dict(
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        x=0.89,            
        xanchor="right",
        yanchor="top",
        font=dict(size=11)
    )

    fig.layout = dict(
        margin=dict(t=80, b=60, l=80, r=20),
        hovermode='closest',
        bargap=0.05,
        uirevision=True,
        height=1000,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=14, color="black"),
        
        showlegend=True,   
        
        legend=dict(**legend_style, y=0.23),
        legend2=dict(**legend_style, y=0.48),
        legend3=dict(**legend_style, y=0.73),
        legend4=dict(**legend_style, y=0.98),

        annotations=[
            dict(
                text="Trace 0",  
                xref="paper", yref="paper",
                x=0.5, y=1.02,  
                xanchor='center', yanchor='bottom',
                showarrow=False,
                font=dict(size=20, family='Arial', color='black')
            )
        ]
    )

    fig.update_layout(
        xaxis1=dict(**mpl_axis_style, domain=[0, 0.9], anchor='y1', title=dict(text='Time (s)', font=dict(size=14)), range=(0, 360)),
        xaxis3=dict(**mpl_axis_style, domain=[0, 0.9], anchor='y2', matches='x1', showticklabels=False),
        xaxis4=dict(**mpl_axis_style, domain=[0, 0.9], anchor='y3', matches='x1', showticklabels=False),
        xaxis5=dict(**mpl_axis_style, domain=[0, 0.9], anchor='y4', matches='x1', showticklabels=False),
        
        xaxis2=dict(**mpl_axis_style, domain=[0.92, 1], anchor='y1', title=dict(text='Density', font=dict(size=14))), 
        
        yaxis1=dict(**mpl_axis_style, domain=[0.01, 0.23], anchor='x1', title=dict(text='FRET', font=dict(size=14)), dtick=0.2, range=[0, 1],minallowed=-0.2, maxallowed=1.2),
        yaxis2=dict(**mpl_axis_style, domain=[0.26, 0.48], anchor='x3', title=dict(text='FRET', font=dict(size=14)), dtick=0.2, range=[0, 1],minallowed=-0.2, maxallowed=1.2),
        yaxis3=dict(**mpl_axis_style, domain=[0.51, 0.73], anchor='x4', title=dict(text='Intensity', font=dict(size=14)), rangemode='tozero'),  #range=(0, None),
        yaxis4=dict(**mpl_axis_style, domain=[0.76, 0.98], anchor='x5', title=dict(text='Intensity', font=dict(size=14)), rangemode='tozero'),   #range=(0, None),
        
        autosize=True,
    )

    # --- BLOB FIGURE ---
    fig_blob = px.imshow(np.zeros((9, 54)), zmin=0, zmax=128)
    fig_blob.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showline=True, linecolor='black', linewidth=1.5, mirror=True, range=(-0.5, 53.5), showticklabels=False, autorange=False),
        yaxis=dict(showline=True, linecolor='black', linewidth=1.5, mirror=True, range=(-0.5, 8.5), showticklabels=False, autorange=False),
        width=2400, 
        height=400,
        uirevision=True,
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    for i in range(1, 6):
        fig_blob.add_vline(x=9 * i - 0.5, line_color='white', line_width=1)

    fig_blob.add_annotation(x=4, y=7, showarrow=False, text="BB", font=dict(size=16, color="#ADD8E6"))
    fig_blob.add_annotation(x=13, y=7, showarrow=False, text="BG", font=dict(size=16, color="#ADD8E6"))
    fig_blob.add_annotation(x=22, y=7, showarrow=False, text="BR", font=dict(size=16, color="#ADD8E6"))
    fig_blob.add_annotation(x=31, y=7, showarrow=False, text="GG", font=dict(size=16, color="#90EE90"))
    fig_blob.add_annotation(x=40, y=7, showarrow=False, text="GR", font=dict(size=16, color="#90EE90"))
    fig_blob.add_annotation(x=49, y=7, showarrow=False, text="RR", font=dict(size=16, color="#FFCCCB"))

    # --- FIG 2 (GMM Histogram) ---
    fig2 = go.FigureWidget()
    fig2.add_histogram(x=[], name='gmm_hist', histnorm='probability density', marker=dict(color='#f7e1a0', opacity=0.8, line=dict(width=1.5, color='black')),
                    xbins=dict(start=0, end=1, size=0.02),
                    cumulative=dict(enabled=False))

    fig2.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showline=True, linecolor='black', linewidth=1.5, mirror=True, ticks='inside', tickwidth=1.5, tickcolor='black', ticklen=5, dtick=0.2, range=(0, 1), title='FRET'),
        yaxis=dict(showline=True, linecolor='black', linewidth=1.5, mirror=True, ticks='inside', tickwidth=1.5, tickcolor='black', ticklen=5, rangemode='tozero', title='Density'),
        autosize=True,
        showlegend=False,
        margin=dict(l=60, r=20, t=20, b=50)
    )
    
    return fig, fig_blob, fig2
