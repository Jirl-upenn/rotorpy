import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R

class Actor(nn.Module):
    def __init__(self, mlp_input_dim, actor_hidden_dims, num_actions, activation):
        super().__init__()
        
        layers = [nn.Linear(mlp_input_dim, actor_hidden_dims[0]), activation()]
        for i in range(len(actor_hidden_dims) - 1):
            layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i+1]))
            layers.append(activation())
        layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
        layers.append(nn.Tanh())
        
        self.actor = nn.Sequential(*layers)

    def forward(self, obs):
        return self.actor(obs)

class IsaacHoveringController:
    def __init__(self, vehicle, model_path, waypoints, scale_output=True, device="cpu"):
        self.quadrotor = vehicle
        self.device = torch.device(device)
        
        self.obs_dim = 23
        self.action_dim = 4

        self.scale_output = scale_output

        self.waypoints = waypoints
        self.proximity_threshold = 0.15

        # Create network
        self.model = Actor(self.obs_dim, [64, 64], self.action_dim, nn.ELU).to(self.device)
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        actor_state_dict = {k.replace('actor.', ''): v for k, v in checkpoint["model_state_dict"].items() if "actor" in k}
        self.model.actor.load_state_dict(actor_state_dict, strict=True)

        self.model.eval()

        ######  Min/max values for scaling control outputs.
        rotor_speed_max = self.quadrotor['rotor_speed_max']
        rotor_speed_min = self.quadrotor['rotor_speed_min']

        # Compute the min/max thrust by assuming the rotor is spinning at min/max speed. (also generalizes to bidirectional rotors)
        self.max_thrust = self.quadrotor['k_eta'] * rotor_speed_max**2
        self.min_thrust = self.quadrotor['k_eta'] * rotor_speed_min**2

        # Set the maximum body rate on each axis (this is hand selected), rad/s
        self.max_roll_br = self.max_pitch_br = 4.0
        self.max_yaw_br = 2.0

        self.idx_wp = 0
        self._previous_action = torch.zeros(1, self.action_dim, device=self.device)
        
        # From quadcopter_env.py
        self.moment_scale = 0.01
        Ixx = self.quadrotor['Ixx']
        Iyy = self.quadrotor['Iyy']
        Izz = self.quadrotor['Izz']
        Ixy = self.quadrotor.get('Ixy', 0.0)
        Ixz = self.quadrotor.get('Ixz', 0.0)
        Iyz = self.quadrotor.get('Iyz', 0.0)
        self.inertia = torch.tensor([[Ixx, Ixy, Ixz],
                                       [Ixy, Iyy, Iyz],
                                       [Ixz, Iyz, Izz]], dtype=torch.float32, device=self.device)
        # P-controller for converting moment to body rates
        self.moment_to_rate_p = 10.0

    def update(self, t, state, traj):
        """
        Compute the control command using the neural network.

        Inputs:
            state, current dictionary with state
         Output:
            control_input and observation vector:
        """
        pos_drone = torch.tensor(state['x'], dtype=torch.float32, device=self.device)
        quat_drone = torch.tensor(state['q'], dtype=torch.float32, device=self.device)
        rot_drone = torch.tensor(R.from_quat(quat_drone, scalar_first=False).as_matrix(), dtype=torch.float32, device=self.device)
        lin_vel = torch.tensor(state['v'], dtype=torch.float32, device=self.device)
        ang_vel = torch.tensor(state['w'], dtype=torch.float32, device=self.device)
        lin_vel_drone = rot_drone.T @ lin_vel

        wp_curr_pos = torch.tensor(self.waypoints[self.idx_wp, :3], dtype=torch.float32, device=self.device)

        dist_to_wp = torch.norm(pos_drone - wp_curr_pos)
        if dist_to_wp < self.proximity_threshold:
            self.idx_wp = (self.idx_wp + 1) % self.waypoints.shape[0]
            wp_curr_pos = torch.tensor(self.waypoints[self.idx_wp, :3], dtype=torch.float32, device=self.device)

        # Observation construction based on quadcopter_env.py
        # 1. absolute height (1)
        abs_height = pos_drone[2].unsqueeze(0)

        # 2. relative desired position (3)
        desired_pos_b = rot_drone.T @ (wp_curr_pos - pos_drone)

        # 3. attitude matrix (9)
        attitude_mat = rot_drone.flatten()

        # 4. linear velocity in body frame (3)
        lin_vel_b = lin_vel_drone

        # 5. angular velocity in body frame (3)
        ang_vel_b = ang_vel

        # 6. last actions (4)
        last_actions = self._previous_action.flatten()

        obs = torch.cat(
            [
                abs_height,
                desired_pos_b,
                attitude_mat,
                lin_vel_b,
                ang_vel_b,
                last_actions,
            ],
            dim=-1,
        )

        actions_tensor = self.model(obs).squeeze(0)
        actions = actions_tensor.detach().cpu().numpy()
        actions = np.clip(actions, -1, 1)
        self._previous_action = actions_tensor.detach().unsqueeze(0)

        num_rotors = self.quadrotor['num_rotors']

        if self.scale_output:
            # The policy outputs thrust and body rates, not motor speeds directly.
            # The first action is thrust, normalized between -1 and 1.
            # The policy from isaac sim has a different thrust mapping.
            # From quadcopter_env.py: self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
            # Assuming thrust_to_weight is around 2.0, this means thrust is roughly from 0 to 2*weight.
            # Let's assume a thrust-to-weight of 1.9 as in the config.
            thrust_to_weight = 1.9
            robot_mass = self.quadrotor['mass']
            gravity = 9.81
            robot_weight = robot_mass * gravity
            
            cmd_thrust = thrust_to_weight * robot_weight * (actions[0] + 1.0) / 2.0

            # The policy outputs moments, but the simulator expects body rates.
            # We can use a simple P-controller to convert moments to body rates.
            desired_moment = self.moment_scale * torch.tensor(actions[1:], dtype=torch.float32, device=self.device)
            
            # w x (I @ w)
            gyroscopic_term = torch.cross(ang_vel, self.inertia @ ang_vel)
            
            # Simplified: Solve for w_dot from tau = I @ w_dot, then integrate to get w
            # More simply, use a P-controller on the moment error.
            # Let's assume the desired body rates can be commanded from the desired moments.
            # A better way is to compute desired angular acceleration and integrate.
            # w_dot = torch.linalg.inv(self.inertia) @ (desired_moment - gyroscopic_term)
            # For simplicity and stability, let's use a proportional mapping.
            desired_body_rates = self.moment_to_rate_p * desired_moment

            roll_br = np.clip(desired_body_rates[0].item(), -self.max_roll_br, self.max_roll_br)
            pitch_br = np.clip(desired_body_rates[1].item(), -self.max_pitch_br, self.max_pitch_br)
            yaw_br = np.clip(desired_body_rates[2].item(), -self.max_yaw_br, self.max_yaw_br)
        else:
            # If not scaling, assume actions are direct commands. This part might need adjustment.
            cmd_thrust = actions[0]
            roll_br, pitch_br, yaw_br = actions[1], actions[2], actions[3]


        control_input = {'cmd_thrust': cmd_thrust,
                         'cmd_w': np.array([roll_br, pitch_br, yaw_br]),
                         'cmd_q': np.array([1, 0, 0, 0]),
                         'cmd_motor_speeds': np.array([0, 0, 0, 0]),
                         'cmd_moment': np.array([0, 0, 0, 0])}

        return control_input
