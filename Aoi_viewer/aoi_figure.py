import plotly.graph_objects as go
import cv2
import numpy as np


def create_initial_figure(image_g, minf, maxf, radius):
    """
    Create and return the initial figure with the image and empty blob markers.
    """
    # --- 1. THE COMPRESSION TRICK ---
    original_size = image_g.shape[1] # Usually 1024
    target_size = 512                # The new, smaller size for faster drawing
    
    # Compress the massive image down to 512x512
    compressed_image = cv2.resize(
        image_g[0].astype(np.float32), 
        (target_size, target_size), 
        interpolation=cv2.INTER_AREA
    )
    
    # Create artificial X and Y coordinates that stretch from 0 up to 1024
    x_coords = np.linspace(0, original_size, target_size)
    y_coords = np.linspace(0, original_size, target_size)
    # ---------------------------------
    fig = go.Figure(
        data=go.Heatmap(
            z=compressed_image,  # Use the small image
            x=x_coords,          # Stretch it across the X axis (0 to 1024)
            y=y_coords,          # Stretch it across the Y axis (0 to 1024)
            colorscale='gray',
            zmin=minf,
            zmax=maxf
        )
    )
    fig.update_layout(
        margin=dict(r=120),
        xaxis=dict(
            showline=True,
            range=(0, original_size), # Keep axes at original size
            autorange=False
        ),
        yaxis=dict(
            showline=True,
            range=(original_size, 0), # Keep axes at original size
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
