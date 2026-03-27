import plotly.graph_objects as go


def create_initial_figure(image_g, minf, maxf, radius):
    """
    Create and return the initial figure with the image and empty blob markers.
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=image_g[0],
            colorscale='gray',
            zmin=minf,
            zmax=maxf
    )
)
    fig.update_layout(
        margin=dict(r=120),
        xaxis=dict(
            showline=True,
            range=(0, image_g.shape[2]),
            autorange=False
        ),
        yaxis=dict(
            showline=True,
            range=(image_g.shape[1], 0),
            autorange=False,
            scaleanchor="x",
            scaleratio=1
        ),
        autosize=True,
        uirevision=True,
        dragmode='pan'
    )
    # Empty traces for each color channel
    fig.add_scatter(
        x=[], y=[],
        mode='markers',
        marker_symbol='square-open',
        marker=dict(
            color='rgba(135, 206, 250, 0.5)',
            size=2 * radius + 1,
            line=dict(color='MediumPurple', width=1)
        ),
        customdata = [],
        hovertemplate = '<b>No:</b> %{pointNumber}<br>' +  'X value: %{x}<br>' +'Y value: %{y}<br>'+ 'FRET_g: %{customdata}<br>' + '<extra></extra>',
        name='blobs_r'
    )
    fig.add_scatter(
        x=[], y=[],
        mode='markers',
        marker_symbol='square-open',
        marker=dict(
            color='rgba(135, 206, 250, 0.5)',
            size=2 * radius + 1,
            line=dict(color='MediumPurple', width=1)
        ),
        customdata = [],
        hovertemplate = '<b>No:</b> %{pointNumber}<br>' +  'X value: %{x}<br>' +'Y value: %{y}<br>' + 'FRET_g: %{customdata}<br>' +'<extra></extra>',
        name='blobs_g'
    )
    fig.add_scatter(
        x=[], y=[],
        mode='markers',
        marker_symbol='square-open',
        marker=dict(
            color='rgba(135, 206, 250, 0.5)',
            size=2 * radius + 1,
            line=dict(color='MediumPurple', width=1)
        ),
        hovertemplate = '<b>No:</b> %{pointNumber}<br>' +  'X value: %{x}<br>' +'Y value: %{y}<br>' + '<extra></extra>',
        name='blobs_b'
    )
    return fig
