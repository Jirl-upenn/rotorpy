import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.actor(x)

class PolicyControl(object):
    def __init__(self, vehicle, model_path, scale_output=True, device="cpu"):
        self.quadrotor = vehicle
        self.device = torch.device(device)
        self.obs_dim = 1 + 3 + 9 + 3 + 3 + 4
        self.action_dim = 4

        self.scale_output = scale_output

        self.last_actions = torch.zeros(self.action_dim, device=self.device)

        # Create network
        self.model = Actor(self.obs_dim, self.action_dim).to(self.device)
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        actor_state_dict = {k: v for k, v in checkpoint["model_state_dict"].items() if "actor" in k}
        self.model.load_state_dict(actor_state_dict, strict=False)

        self.model.eval()

        ######  Min/max values for scaling control outputs.

        self.rotor_speed_max = self.quadrotor['rotor_speed_max']
        self.rotor_speed_min = self.quadrotor['rotor_speed_min']

        # Compute the min/max thrust by assuming the rotor is spinning at min/max speed. (also generalizes to bidirectional rotors)
        self.max_thrust = self.quadrotor['k_eta'] * self.rotor_speed_max**2
        self.min_thrust = self.quadrotor['k_eta'] * self.rotor_speed_min**2

        # Find the maximum moment on each axis, N-m
        self.max_roll_moment = self.max_thrust * np.abs(self.quadrotor['rotor_pos']['r1'][1])
        self.max_pitch_moment = self.max_thrust * np.abs(self.quadrotor['rotor_pos']['r1'][0])
        self.max_yaw_moment = self.quadrotor['k_m'] * self.rotor_speed_max**2

        # Set the maximum body rate on each axis (this is hand selected), rad/s
        self.max_roll_br = 7.0
        self.max_pitch_br = 7.0 
        self.max_yaw_br = 3.0

    def update(self, _, state, flat_output):
        """
        Compute the control command using the neural network.
        
        Inputs:
            t, current time in seconds
            state, current state with keys:
                - x: absolute position (3,)
                - v: linear velocity (3,)
                - q: quaternion [i, j, k, w]
                - w: angular velocity (3,)
            flat_output, desired output with keys:
                - x: target position (3,)
         Output:
            control_input, dictionary with 4 controls:
                - cmd_thrust
                - cmd_moment
        """
        pos = state["x"]
        lin_vel = state["v"]
        quat = state["q"]
        ang_vel = state["w"]
        pos_des = flat_output["x"]

        altitude = pos[2]

        R = Rotation.from_quat(quat).as_matrix()
        vector_to_target = R.T @ (pos_des - pos)

        obs = torch.tensor(
            np.concatenate([
                [altitude],
                vector_to_target,
                R.flatten(),
                lin_vel,
                ang_vel,
                self.last_actions
            ]), 
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)  # Add batch dimension

        actions = self.model(obs).squeeze(0).detach().cpu().numpy()
        actions = np.clip(actions, -1, 1)
        self.last_actions = torch.tensor(actions, device=self.device)

        num_rotors = self.quadrotor['num_rotors']
        cmd_thrust = actions[0]
        cmd_w = -np.array([actions[1], actions[2], actions[3]])

        if self.scale_output:
            cmd_thrust = np.interp(cmd_thrust,
                                   [-1, 1],
                                   [num_rotors * self.min_thrust, num_rotors * self.max_thrust])

            cmd_w[0] = np.interp(cmd_w[0],
                                 [-1, 1],
                                 [-self.max_roll_br, self.max_roll_br])
            cmd_w[1] = np.interp(cmd_w[1],
                                 [-1, 1],
                                 [-self.max_pitch_br, self.max_pitch_br])
            cmd_w[2] = np.interp(cmd_w[2],
                                 [-1, 1],
                                 [-self.max_yaw_br, self.max_yaw_br])

        control_input = {'cmd_motor_speeds': np.zeros((4,)),
                         'cmd_motor_thrusts': np.zeros((4,)),
                         'cmd_thrust': cmd_thrust,
                         'cmd_moment': np.zeros((3,)),
                         'cmd_q': np.array([0, 0, 0, 1]),
                         'cmd_w': cmd_w,
                         'cmd_v': np.zeros((3,))}

        return control_input
