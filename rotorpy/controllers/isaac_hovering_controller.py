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
        
        self.actor = nn.Sequential(*layers)

    def forward(self, obs):
        return self.actor(obs)

class IsaacHoveringController:
    def __init__(self, vehicle, model_path, waypoints, device="cpu", scale_output=True):
        self.quadrotor = vehicle
        self.device = torch.device(device)
        
        self.obs_dim = 23
        self.action_dim = 4

        self.waypoints = waypoints
        self.proximity_threshold = 0.15
        self.wait_time_s = 1.0
        
        # Initialize loss tracking
        self.position_loss = 0.0
        self.control_loss = 0.0
        self.waypoints_achieved = 0
        self.total_loss = {
            "total": 0.0,
            "position": 0.0,
            "control": 0.0,
            "waypoints_achieved": 0
        }

        # Create network
        self.policy = Actor(self.obs_dim, [64, 64], self.action_dim, nn.ELU).to(self.device)
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        actor_state_dict = {k.replace('actor.', ''): v for k, v in checkpoint["model_state_dict"].items() if "actor" in k}
        self.policy.actor.load_state_dict(actor_state_dict, strict=True)

        self.policy.eval()

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
        
        # Timing and waypoint switching variables (matching Isaac environment logic)
        self._previous_t = 0.0
        self._last_waypoint_switch_time = 0.0
        
        # Control output scaling flag
        self.scale_output = scale_output
        
        # Waypoint history tracking for visualization
        self.waypoint_history = []  # List of (time, waypoint_index) tuples
        
    def reset_loss(self):
        """Reset all accumulated loss values."""
        self.position_loss = 0.0
        self.control_loss = 0.0
        self.waypoints_achieved = 0
        self.total_loss = {
            "total": 0.0,
            "position": 0.0,
            "control": 0.0,
            "waypoints_achieved": 0
        }

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
        
        # Apply Isaac environment waypoint switching logic
        close_to_goal = dist_to_wp < self.proximity_threshold
        slow_speed = torch.norm(lin_vel) < 0.1  # Match Isaac environment speed threshold
        time_cond = (t - self._last_waypoint_switch_time) >= self.wait_time_s
        should_switch = close_to_goal and slow_speed and time_cond
        
        if should_switch:
            self.waypoints_achieved += 1
            self.idx_wp = (self.idx_wp + 1) % self.waypoints.shape[0]
            wp_curr_pos = torch.tensor(self.waypoints[self.idx_wp, :3], dtype=torch.float32, device=self.device)
            self._last_waypoint_switch_time = t
            print(f"[INFO] Switched to waypoint {self.idx_wp}: {wp_curr_pos.cpu().numpy()} at time {t:.2f}s (Total achieved: {self.waypoints_achieved})")
        
        # Record current waypoint in history (for visualization)
        self.waypoint_history.append((float(t), int(self.idx_wp)))

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

        actions_tensor = self.policy(obs).squeeze(0)
        actions = actions_tensor.detach().cpu().numpy()
        actions = np.clip(actions, -1, 1)
        self._previous_action = actions_tensor.detach().unsqueeze(0)

        num_rotors = self.quadrotor['num_rotors']

        if self.scale_output:
            cmd_thrust = np.interp(actions[0],
                                   [-1, 1],
                                   [num_rotors * self.min_thrust, num_rotors * self.max_thrust])

            roll_br = np.interp(actions[1],
                                [-1, 1],
                                [-self.max_roll_br, self.max_roll_br])
            pitch_br = np.interp(actions[2],
                                 [-1, 1],
                                 [-self.max_pitch_br, self.max_pitch_br])
            yaw_br = np.interp(actions[3],
                               [-1, 1],
                               [-self.max_yaw_br, self.max_yaw_br])
        else:
            # If not scaling, assume actions are direct commands. This part might need adjustment.
            cmd_thrust = actions[0]
            roll_br, pitch_br, yaw_br = actions[1], actions[2], actions[3]


        control_input = {'cmd_thrust': cmd_thrust,
                         'cmd_w': np.array([roll_br, pitch_br, yaw_br]),
                         'cmd_q': np.array([1, 0, 0, 0]),
                         'cmd_motor_speeds': np.array([0, 0, 0, 0]),
                         'cmd_moment': np.array([0, 0, 0, 0])}
        
        # Compute loss components
        # Position loss: distance to current waypoint
        current_position_loss = float(dist_to_wp.detach().cpu().numpy())
        self.position_loss += current_position_loss
        
        # Control loss: norm of control inputs
        control_norm = np.sqrt(cmd_thrust**2 + roll_br**2 + pitch_br**2 + yaw_br**2)
        self.control_loss += control_norm
        
        # Update total loss dictionary with all components
        self.total_loss = {
            "total": self.position_loss + self.control_loss,
            "position": self.position_loss,
            "control": self.control_loss,
            "waypoints_achieved": self.waypoints_achieved
        }

        return control_input
    
    def get_waypoint_history(self):
        """
        Get the waypoint history for visualization.
        
        Returns:
            list: List of (time, waypoint_index) tuples
        """
        return self.waypoint_history
