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
from rotorpy.controllers.controller_policy import RacingPolicy

# And a trajectory generator
from rotorpy.trajectories.hover_traj import HoverTraj
from rotorpy.trajectories.point_to_point import Point2Point

from scipy.spatial.transform import Rotation as R

# Reference the files above for more documentation.

# Other useful imports
import numpy as np                  # For array creation/manipulation
import matplotlib.pyplot as plt     # For plotting, although the simulator has a built in plotter
from scipy.spatial.transform import Rotation  # For doing conversions between different rotation descriptions, applying rotations, etc.
import os                           # For path generation

"""
Instantiation
"""
model_path = '/home/neo/workspace/logs/policies/2025-06-11_01-07-04/model_9999_8192.pt'

waypoints = np.array([
      [ 0.0, 3.0, 0.75, 0.0, 0.0,  0.0],
      [-2.0, 4.5, 0.75, 0.0, 0.0, -1.57],
      [ 0.0, 6.0, 1.75, 0.0, 0.0,  3.14],
      [ 2.0, 4.5, 0.75, 0.0, 0.0,  1.57]
])
waypoints_quat = np.zeros((waypoints.shape[0], 4))

for i, waypoint_data in enumerate(waypoints):
      euler_np = waypoint_data[3:6]
      rot_from_euler = R.from_euler('xyz', euler_np)
      waypoints_quat[i, :] = rot_from_euler.as_quat(scalar_first=True)

waypoints = waypoints
waypoints_quat = waypoints_quat

gate_side = 1.0
d = 1.0 / 2
local_square = np.array([
      [0,  d,  d],
      [0, -d,  d],
      [0, -d, -d],
      [0,  d, -d]
])

# An instance of the simulator can be generated as follows:
cf2_ctbr = Multirotor(quad_params, control_abstraction='cmd_ctbr')
# controller = SE3Control(quad_params)
controller = RacingPolicy(quad_params, model_path, waypoints, waypoints_quat, gate_side)
traj = Point2Point(t_change_target=100)
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
                           fname          = None      # Filename is specified if you want to save the animation. The save location is rotorpy/data_out/. 
                          )
