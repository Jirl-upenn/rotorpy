"""
Custom animation utilities with waypoint visualization for rotorpy.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation

from rotorpy.utils.animate import _decimate_index, ClosingFuncAnimation
from rotorpy.utils.shapes import Quadrotor
from rotorpy.utils.postprocessing import unpack_sim_data
from rotorpy.world import World


def animate_with_waypoints(time, position, rotation, wind, animate_wind, world, waypoints, 
                          waypoint_history=None, filename=None, blit=False, show_axes=True, close_on_finish=False):
    """
    Custom animation function that adds red waypoint markers and highlights the current waypoint.
    Based on rotorpy.utils.animate.animate but with waypoint visualization.
    
    Parameters:
        time: (N,) array of time steps
        position: (N,M,3) array of drone positions 
        rotation: (N,M,3,3) array of drone rotations
        wind: (N,M,3) array of wind velocities
        animate_wind: bool, whether to show wind arrows
        world: World object for drawing environment
        waypoints: (K,3) array of waypoint positions
        waypoint_history: list of (time, waypoint_index) tuples from controller
        filename: str, filename to save animation
        blit: bool, use blitting for faster animation
        show_axes: bool, whether to show 3D axes
        close_on_finish: bool, close figure when done
    
    Returns:
        Animation object
    """
    # Check if there is only one drone
    if len(position.shape) == 2:
        position = np.expand_dims(position, axis=1)
        rotation = np.expand_dims(rotation, axis=1)
        wind = np.expand_dims(wind, axis=1)
    M = position.shape[1]

    # Temporal style
    rtf = 1.0  # real time factor > 1.0 is faster than real time playback
    render_fps = 30

    # Normalize the wind by the max of the wind magnitude on each axis
    wind_mag = np.max(np.linalg.norm(wind, axis=-1), axis=1)
    max_wind = np.max(wind_mag)

    if max_wind != 0:
        wind_arrow_scale_factor = 1
        wind = wind_arrow_scale_factor*wind / max_wind

    # Decimate data to render interval; always include t=0
    if time[-1] != 0:
        sample_time = np.arange(0, time[-1], 1/render_fps * rtf)
    else:
        sample_time = np.zeros((1,))
    index = _decimate_index(time, sample_time)
    time = time[index]
    position = position[index,:]
    rotation = rotation[index,:]
    wind = wind[index,:]

    # Set up axes
    fig = plt.figure('Animation with Waypoints')
    fig.clear()
    ax = fig.add_subplot(projection='3d')
    if not show_axes:
        ax.set_axis_off()

    quads = [Quadrotor(ax, wind=animate_wind, wind_scale_factor=1) for _ in range(M)]
    world_artists = world.draw(ax)

    # Add waypoint markers - all waypoints as red dots
    waypoint_artists = []
    for i, wp in enumerate(waypoints):
        scatter = ax.scatter(wp[0], wp[1], wp[2], c='red', s=100, alpha=0.7, marker='o')
        waypoint_artists.append(scatter)

    # Create a special marker for the current waypoint (larger, brighter)
    current_waypoint_artist = ax.scatter([], [], [], c='red', s=200, alpha=1.0, 
                                       marker='o', edgecolors='black', linewidth=2)
    waypoint_artists.append(current_waypoint_artist)

    title_artist = ax.set_title('t = {:.2f}, Current Waypoint: 0'.format(time[0]))

    def init():
        ax.draw(fig.canvas.get_renderer())
        return world_artists + waypoint_artists + [title_artist] + [q.artists for q in quads]

    def update(frame):
        # Get current waypoint from controller history
        current_wp_idx = 0
        if waypoint_history and len(waypoints) > 0:
            current_time = time[frame]
            # Find the most recent waypoint change before or at current time
            for hist_time, wp_idx in waypoint_history:
                if hist_time <= current_time:
                    current_wp_idx = wp_idx
                else:
                    break
        elif len(waypoints) > 0:
            # Fallback: if no waypoint history, use waypoint 0
            current_wp_idx = 0

        # Update current waypoint highlight
        if len(waypoints) > 0:
            current_wp = waypoints[current_wp_idx]
            current_waypoint_artist._offsets3d = ([current_wp[0]], [current_wp[1]], [current_wp[2]])

        title_artist.set_text('t = {:.2f}, Current Waypoint: {} [{}]'.format(
            time[frame], current_wp_idx, 
            np.round(waypoints[current_wp_idx], 2) if len(waypoints) > 0 else "N/A"))
        
        for i, quad in enumerate(quads):
            quad.transform(position=position[frame,i,:], rotation=rotation[frame,i,:,:], wind=wind[frame,i,:])
        
        return world_artists + waypoint_artists + [title_artist] + [q.artists for q in quads]

    ani = ClosingFuncAnimation(fig=fig,
                        func=update,
                        frames=time.size,
                        init_func=init,
                        interval=1000.0/render_fps,
                        repeat=False,
                        blit=blit,
                        close_on_finish=close_on_finish)

    if filename is not None:
        print('Saving Animation with Waypoints')
        if not ".mp4" in filename:
            filename = filename + ".mp4"
        ani.save(filename,
                 writer='ffmpeg',
                 fps=render_fps,
                 dpi=100)
        if close_on_finish:
            plt.close(fig)
            ani = None

    return ani


def add_waypoint_visualization(results, waypoints, video_path, waypoint_history=None):
    """
    Create a custom animation with waypoint markers and save it.
    
    Parameters:
        results: dict, simulation results from rotorpy Environment.run()
        waypoints: (K,3) array of waypoint positions
        video_path: str, path to save the animation video
        waypoint_history: list of (time, waypoint_index) tuples from controller
    """
    # Extract data directly from results dictionary
    time = results['time']
    state = results['state']
    x = state['x']
    q = state['q']
    wind = state['wind']
    
    # Convert quaternions to rotation matrices
    R = Rotation.from_quat(q).as_matrix()
    
    # Create empty world (same as in Environment class)
    wbound = 3
    world = World.empty((-wbound, wbound, -wbound, wbound, -wbound, wbound))
    
    # Create the animation with waypoints
    ani = animate_with_waypoints(time, x, R, wind, animate_wind=False, world=world, 
                                waypoints=waypoints, waypoint_history=waypoint_history, 
                                filename=video_path, close_on_finish=True)
    
    print(f"Waypoint visualization saved to: {video_path}")
    return ani