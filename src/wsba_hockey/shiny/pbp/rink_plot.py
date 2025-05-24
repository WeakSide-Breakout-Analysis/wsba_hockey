
import numpy as np
import plotly.graph_objects as go
import io
import base64
import requests as rs
from PIL import Image

def rink(setting = "full", vertical = False):
    '''
    Function to plot rink in Plotly. Takes 2 arguments :

    setting : full (default) for full ice, offense positive half of the ice, ozone positive quarter of ice, defense for negative half of the ice, dzone for negative quarter of the ice, and neutral for the neutral zone
    vertical : True if you want a vertical rink, False (default) is for an horizontal rink

    '''

    def faceoff_circle(x, y, outer=True):
        segments = []
        theta = np.linspace(0, 2*np.pi, 300)
        if outer:
            # Outer circle
            x_outer = x + 15*np.cos(theta)
            y_outer = y + 15*np.sin(theta)
            outer_circle = go.Scatter(x=x_outer, y=y_outer, mode='lines', line=dict(width=2, color='red'), showlegend=False, hoverinfo='skip')

            segments.append(outer_circle)

        # Inner circle
        x_inner = x + np.cos(theta)
        y_inner = y + np.sin(theta)
        inner_circle = go.Scatter(x=x_inner, y=y_inner, mode='lines', fill='toself', fillcolor='rgba(255, 0, 0, 0.43)', line=dict(color='rgba(255, 0, 0, 1)', width=2), showlegend=False, hoverinfo='skip')
        
        segments.append(inner_circle)
        
        return segments  #segments
    
    fig = go.Figure()

    if vertical :
        setting_dict = {
            "full" : [-101, 101],
            "offense" : [0, 101],
            "ozone" : [25, 101],
            "defense" : [-101, 0],
            "dzone" : [-101, -25],
            "neutral" : [-25,25]
        }
        fig.update_layout(xaxis=dict(range=[-42.6, 42.6], showgrid=False, zeroline=False, showticklabels=False, constrain="domain"), yaxis=dict(range=setting_dict[setting], showgrid=False, zeroline=False, showticklabels=False, constrain="domain"),
                        showlegend=False, autosize=True, template="plotly_white")
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )
        def goal_crease(flip=1):
            x_seq = np.linspace(-4, 4, 100)
            x_goal = np.concatenate(([-4], x_seq, [4]))
            y_goal = flip * np.concatenate(([89], 83 + x_seq**2/4**2*1.5, [89]))
            goal_crease = go.Scatter(x=x_goal, y=y_goal, fill='toself', fillcolor='rgba(173, 216, 230, 0.3)', line=dict(color='red'))
            return goal_crease

        # Outer circle
        theta = np.linspace(0, 2*np.pi, 300)
        x_outer = 15 * np.cos(theta)
        y_outer = 15 * np.sin(theta)
        fig.add_trace(go.Scatter(x=x_outer, y=y_outer, mode='lines', line=dict(color='royalblue', width=2), showlegend=False, hoverinfo='skip'))
        # Inner circle
        theta2 = np.linspace(np.pi/2, 3*np.pi/2, 300)
        x_inner = 42.5 + 10 * np.cos(theta2)
        y_inner = 10 * np.sin(theta2)
        fig.add_trace(go.Scatter(x=x_inner, y=y_inner, mode='lines', line=dict(color='red', width=2), showlegend=False, hoverinfo='skip'))
        # Rink boundaries
        fig.add_shape(type='rect', xref='x', yref='y', x0=-42.5, y0=25, x1=42.5, y1=26, line=dict(color='royalblue', width=1), fillcolor='royalblue', opacity=1)
        fig.add_shape(type='rect', xref='x', yref='y', x0=-42.5, y0=-25, x1=42.5, y1=-26, line=dict(color='royalblue', width=1), fillcolor='royalblue', opacity=1)
        fig.add_shape(type='rect', xref='x', yref='y', x0=-42.5, y0=-0.5, x1=42.5, y1=0.5, line=dict(color='red', width=2), fillcolor='red')
        
        # Goal crease
        fig.add_trace(goal_crease())
        fig.add_trace(goal_crease(-1))
        # Goal lines
        goal_line_extreme = 42.5 - 28 + np.sqrt(28**2 - (28-11)**2)
        fig.add_shape(type='line', xref='x', yref='y', x0=-goal_line_extreme, y0=89, x1=goal_line_extreme, y1=89, line=dict(color='red', width=2))
        fig.add_shape(type='line', xref='x', yref='y', x0=-goal_line_extreme, y0=-89, x1=goal_line_extreme, y1=-89, line=dict(color='red', width=2))

        # Faceoff circles
        fig.add_traces(faceoff_circle(-22, 69))
        fig.add_traces(faceoff_circle(22, 69))
        fig.add_traces(faceoff_circle(-22, -69))
        fig.add_traces(faceoff_circle(22, -69))
        fig.add_traces(faceoff_circle(-22, -20, False))
        fig.add_traces(faceoff_circle(22, -20, False))
        fig.add_traces(faceoff_circle(-22, 20, False))
        fig.add_traces(faceoff_circle(22, 20, False))
        
        # Sidelines
        theta_lines = np.linspace(0, np.pi/2, 20)
        x_lines1 = np.concatenate(([-42.5], -42.5 + 28 - 28*np.cos(theta_lines), 42.5 - 28 + 28*np.cos(np.flip(theta_lines))))
        y_lines1 = np.concatenate(([15], 72 + 28*np.sin(theta_lines), 72 + 28*np.sin(np.flip(theta_lines))))
        x_lines2 = np.concatenate(([-42.5], -42.5 + 28 - 28*np.cos(theta_lines), 42.5 - 28 + 28*np.cos(np.flip(theta_lines))))
        y_lines2 = np.concatenate(([15], -72 - 28*np.sin(theta_lines), -72 - 28*np.sin(np.flip(theta_lines))))
        fig.add_trace(go.Scatter(x=x_lines1, y=y_lines1, mode='lines', line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=x_lines2, y=y_lines2, mode='lines', line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'))
        fig.add_shape(type='line', xref='x', yref='y', x0=42.5, y0=-72.5, x1=42.5, y1=72.5, line=dict(color='black', width=2))
        fig.add_shape(type='line', xref='x', yref='y', x0=-42.5, y0=-72.5, x1=-42.5, y1=72.5, line=dict(color='black', width=2))
        
        # Add goals
        goal_width = 6     # feet
        goal_depth = 4     # feet

        # Top goal
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=-goal_width / 2,
            y0=89,
            x1=goal_width / 2,
            y1=89 + goal_depth,
            line=dict(color="red", width=2),
        )
        # Bottom goal
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=-goal_width / 2,
            y0=-89 - goal_depth,
            x1=goal_width / 2,
            y1=-89,
            line=dict(color="red", width=2),
        )

    else : 
        setting_dict = {
            "full" : [-101, 101],
            "offense" : [0, 101],
            "ozone" : [25, 101],
            "defense" : [-101, 0],
            "dzone" : [-101, -25]
        }
        fig.update_layout(xaxis=dict(range=setting_dict[setting], showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(range=[-42.6, 42.6], showgrid=False, zeroline=False, showticklabels=False, constrain="domain"),
                        showlegend=True, autosize =True, template="plotly_white")
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )
        def goal_crease(flip=1):
            y_seq = np.linspace(-4, 4, 100)
            y_goal = np.concatenate(([-4], y_seq, [4]))
            x_goal = flip * np.concatenate(([89], 83 + y_seq**2/4**2*1.5, [89]))
            goal_crease = go.Scatter(x=x_goal, y=y_goal, fill='toself', fillcolor='rgba(173, 216, 230, 0.3)', line=dict(color='red'), showlegend=False, hoverinfo='skip')
            return goal_crease
        
        # Outer circle
        theta = np.linspace(0, 2 * np.pi, 300)
        x_outer = 15 * np.sin(theta)
        y_outer = 15 * np.cos(theta)
        fig.add_trace(go.Scatter(x=x_outer, y=y_outer, mode='lines', line=dict(color='royalblue', width=2), showlegend=False, hoverinfo='skip'))
        # Inner circle
        theta2 = np.linspace(3 * np.pi / 2, np.pi / 2, 300)  # Update theta2 to rotate the plot by 180 degrees
        x_inner = 10 * np.sin(theta2)  # Update x_inner to rotate the plot by 180 degrees
        y_inner = -42.5 - 10 * np.cos(theta2)  # Update y_inner to rotate the plot by 180 degrees
        fig.add_trace(go.Scatter(x=x_inner, y=y_inner, mode='lines', line=dict(color='red', width=2), showlegend=False, hoverinfo='skip'))
        
        # Rink boundaries
        fig.add_shape(type='rect', xref='x', yref='y', x0=25, y0=-42.5, x1=26, y1=42.5, line=dict(color='royalblue', width=1), fillcolor='royalblue', opacity=1)
        fig.add_shape(type='rect', xref='x', yref='y', x0=-25, y0=-42.5, x1=-26, y1=42.5, line=dict(color='royalblue', width=1), fillcolor='royalblue', opacity=1)
        fig.add_shape(type='rect', xref='x', yref='y', x0=-0.5, y0=-42.5, x1=0.5, y1=42.5, line=dict(color='red', width=2), fillcolor='red')
        # Goal crease
        fig.add_trace(goal_crease())
        fig.add_trace(goal_crease(-1))
        # Goal lines
        goal_line_extreme = 42.5 - 28 + np.sqrt(28 ** 2 - (28 - 11) ** 2)
        fig.add_shape(type='line', xref='x', yref='y', x0=89, y0=-goal_line_extreme, x1=89, y1=goal_line_extreme, line=dict(color='red', width=2))
        fig.add_shape(type='line', xref='x', yref='y', x0=-89, y0=-goal_line_extreme, x1=-89, y1=goal_line_extreme, line=dict(color='red', width=2))
        # Faceoff circles
        fig.add_traces(faceoff_circle(-69, -22))
        fig.add_traces(faceoff_circle(-69, 22))
        fig.add_traces(faceoff_circle(69, -22))
        fig.add_traces(faceoff_circle(69, 22))
        fig.add_traces(faceoff_circle(-20, -22, False))
        fig.add_traces(faceoff_circle(-20, 22, False))
        fig.add_traces(faceoff_circle(20, -22, False))
        fig.add_traces(faceoff_circle(20, 22, False))

        # Sidelines
        theta_lines = np.linspace(0, np.pi / 2, 20)
        x_lines1 = np.concatenate(([15], 72 + 28 * np.sin(theta_lines), 72 + 28 * np.sin(np.flip(theta_lines))))
        y_lines1 = np.concatenate(([-42.5], -42.5 + 28 - 28 * np.cos(theta_lines), 42.5 - 28 + 28 * np.cos(np.flip(theta_lines))))
        x_lines2 = np.concatenate(([15], -72 - 28 * np.sin(theta_lines), -72 - 28 * np.sin(np.flip(theta_lines))))
        y_lines2 = np.concatenate(([-42.5], -42.5 + 28 - 28 * np.cos(theta_lines), 42.5 - 28 + 28 * np.cos(np.flip(theta_lines))))
        fig.add_trace(go.Scatter(x=x_lines1, y=y_lines1, mode='lines', line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=x_lines2, y=y_lines2, mode='lines', line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'))
        fig.add_shape(type='line', xref='x', yref='y', x0=-72.5, y0=-42.5, x1=72.5, y1=-42.5, line=dict(color='black', width=2))
        fig.add_shape(type='line', xref='x', yref='y', x0=-72.5, y0=42.5, x1=72.5, y1=42.5, line=dict(color='black', width=2))

        # Add goals
        goal_width = 6     # feet
        goal_depth = 4     # feet

        # Right goal
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=89,
            y0=-goal_width / 2,
            x1=89 + goal_depth,
            y1=goal_width / 2,
            line=dict(color="red", width=2),
        )
        # Left goal
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=-89 - goal_depth,
            y0=-goal_width / 2,
            x1=-89,
            y1=goal_width / 2,
            line=dict(color="red", width=2),
        )

    # Add logo
    logo = Image.open(rs.get('https://f005.backblazeb2.com/file/weakside-breakout/utils/wsba.png',stream=True).raw)

    fig.add_layout_image(
        dict(
            source=logo,
            xref="x",
            yref="y",
            x=-12,
            y=12,
            sizex=24,
            sizey=24,
            sizing="stretch",
            opacity=1)
        )
    
    #Set background to transparent
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(255,255,255,255)"
    )
    return fig