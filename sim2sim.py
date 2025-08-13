"""
Imports
"""
# The simulator is instantiated using the Environment class
import json
from rotorpy.environments import Environment

# Vehicles. Currently there is only one.
# There must also be a corresponding parameter file.
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params

# You will also need a controller (currently there is only one) that works for your vehicle.
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.controllers.isaac_hovering_controller import IsaacHoveringController

# And a trajectory generator
from rotorpy.trajectories.hover_traj import HoverTraj
from rotorpy.trajectories.point_to_point import Point2Point

# Waypoint visualization
from rotorpy.utils.waypoint_animate import add_waypoint_visualization

from scipy.spatial.transform import Rotation as R

# Reference the files above for more documentation.

# Other useful imports
import numpy as np                  # For array creation/manipulation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt     # For plotting, although the simulator has a built in plotter
from scipy.spatial.transform import Rotation  # For doing conversions between different rotation descriptions, applying rotations, etc.
import os                           # For path generation
import argparse                     # For command line argument parsing

# Custom animation function with waypoint visualization
from matplotlib.animation import FuncAnimation
from rotorpy.utils.animate import _decimate_index, ClosingFuncAnimation
from rotorpy.utils.shapes import Quadrotor

def main(args_dict):
    """
    Main function for sim2sim quadcopter simulation.
    
    Args:
        args_dict (dict): Dictionary containing CLI arguments with keys:
            - model_path (str): Path to the trained model file (.pt)
            - log_dir (str): Path to the log directory for saving results and video
    """
    model_path = args_dict.get('model_path', '/home/neo/workspace/logs/rsl_rl/quadcopter_direct/2025-06-04_17-01-47/model_4999.pt')
    log_dir_arg = args_dict.get('log_dir', None)

    waypoints = np.array([
          [ 0.0, 0.0, 1.0],
          [ 1.0, 0.0, 1.0],
          [ 1.0, 1.0, 1.0],
          [ 0.0, 1.0, 1.0]
    ])

    # An instance of the simulator can be generated as follows:
    cf2_ctbr = Multirotor(quad_params, control_abstraction='cmd_ctbr')
    # controller = SE3Control(quad_params)
    controller = IsaacHoveringController(quad_params, model_path, waypoints)
    traj = HoverTraj()
    sim_instance = Environment(vehicle    = cf2_ctbr,
                               controller = controller,
                               trajectory = traj,
                               sim_rate   = 100,
                              )

    """
    Execution
    """

    # Setting an initial state. This is optional, and the state representation depends on the vehicle used.
    # Generally, vehicle objects should have an "initial_state" attribute.
    x0 = {'x': np.array([2, 3, 0]),
          'v': np.zeros(3,),
          'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
          'w': np.zeros(3,),
          'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
          'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}
    sim_instance.vehicle.initial_state = x0

    # Set the waypoints for the controller.
    sim_instance.controller.waypoints = waypoints

    # Get the path to the logs directory
    if log_dir_arg is not None:
        log_dir = os.path.abspath(log_dir_arg)
    else:
        log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'logs'))
    os.makedirs(log_dir, exist_ok=True)
    video_path = os.path.join(log_dir, 'sim2sim.mp4')
    controller_loss_path = os.path.join(log_dir, 'results.json')

    results = sim_instance.run(t_final        = 20,       # The maximum duration of the environment in seconds
                               use_mocap      = True,     # Boolean: determines if the controller should use the motion capture estimates. 
                               terminate      = False,    # Boolean: if this is true, the simulator will terminate when it reaches the last waypoint.
                               plot           = False,     # Boolean: plots the vehicle states and commands
                               plot_mocap     = False,    # Boolean: plots the motion capture pose and twist measurements
                               plot_estimator = False,    # Boolean: plots the estimator filter states and covariance diagonal elements
                               plot_imu       = False,    # Boolean: plots the IMU measurements
                               animate_bool   = True,     # Boolean: determines if the animation of vehicle state will play.
                               animate_wind   = False,    # Boolean: determines if the animation will include a scaled wind vector to indicate the local wind acting on the UAV. 
                               verbose        = True,     # Boolean: will print statistics regarding the simulation.
                               fname          = video_path      # Filename is specified if you want to save the animation. The save location is rotorpy/data_out/. 
                              )
    
    # Add waypoint visualization to the saved video
    if results is not None:
        # Get waypoint history from controller
        waypoint_history = controller.get_waypoint_history()
        add_waypoint_visualization(results, waypoints, video_path, waypoint_history)

    controller_loss = {"controller_loss": results["controller_loss"]}
    with open(controller_loss_path, 'w') as f:
        json.dump(controller_loss, f, indent=2)
    print(f"Controller loss saved to: {controller_loss_path}")


if __name__ == "__main__":
    """
    Command line interface
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Sim2Sim quadcopter simulation')
    parser.add_argument('--model-path', 
                        type=str, 
                        default='/home/neo/workspace/logs/rsl_rl/quadcopter_direct/2025-06-04_17-01-47/model_4999.pt',
                        help='Path to the trained model file (.pt)')
    parser.add_argument('--log-dir',
                        type=str,
                        default=None,
                        help='Path to the log directory for saving results and video')
    args = parser.parse_args()
    
    # Convert argparse namespace to dictionary
    args_dict = {
        'model_path': args.model_path,
        'log_dir': args.log_dir
    }
    
    # Call main function
    main(args_dict)
