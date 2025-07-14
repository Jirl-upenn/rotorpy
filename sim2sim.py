"""
Imports
"""
# The simulator is instantiated using the Environment class
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

from scipy.spatial.transform import Rotation as R

# Reference the files above for more documentation.

# Other useful imports
import numpy as np                  # For array creation/manipulation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt     # For plotting, although the simulator has a built in plotter
from scipy.spatial.transform import Rotation  # For doing conversions between different rotation descriptions, applying rotations, etc.
import os                           # For path generation

"""
Instantiation
"""
model_path = '/home/neo/workspace/logs/rsl_rl/quadcopter_direct/2025-06-04_17-01-47/model_4999.pt'

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
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'logs'))
os.makedirs(log_dir, exist_ok=True)
video_path = os.path.join(log_dir, 'sim2sim.gif')

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
