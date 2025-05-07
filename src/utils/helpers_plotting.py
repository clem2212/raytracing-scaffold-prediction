"""
@author: Chang Yan, Sandra Haltmeier, Nezami Lab at BWH
@email: cyan3@bwh.harvard.edu
helper functions for visualization used in core_aorta
"""
import plotly.graph_objects as go

def plot_surface(surface_verts, surface_faces, color='lightblue'):
    fig = go.Figure(data=[go.Mesh3d(x=surface_verts[:, 0], y=surface_verts[:, 1], z=surface_verts[:, 2],
                                    alphahull=-1, i=surface_faces[:, 0], j=surface_faces[:, 1], k=surface_faces[:, 2],
                                    color=color, opacity=0.3)])
    fig.update_layout(width=500, height=500, scene=dict(
        xaxis=dict(showticklabels=False, title='coronal', backgroundcolor="rgba(0, 0, 0,0)", gridcolor='lightgray'),
        yaxis=dict(showticklabels=False, title='sagittal', backgroundcolor="rgba(0, 0, 0,0)", gridcolor='lightgray'),
        zaxis=dict(showticklabels=False, title='axial', backgroundcolor="rgba(0, 0, 0,0)", gridcolor='lightgray'),),
                      margin=dict(l=0, r=0, t=0, b=0))
    fig.update_layout(scene_camera=dict(eye=dict(x=2, y=2, z=0.75)), scene_aspectmode='data')
    return fig


def plot_points(points, name, color='blue'):
    fig = go.Figure()
    for i, point in enumerate(points):
        fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]], name=name[i], marker_color=color[i],
                                   mode='markers'))
    fig.update_traces(marker_size=3)
    return fig


def plot_line(line, color='blue', name=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=line[0, :], y=line[1, :], z=line[2, :], name=name, mode='lines'))
    fig.update_traces(line_color=color)
    return fig