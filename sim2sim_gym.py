import os, re
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.learning.quadrotor_reward_functions import hover_reward
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.controllers.policy_controller import RacingPolicy

# Load the model for the appropriate epoch
#model_path = '/home/lorenzo/Github/University/isaac_quad_sim2real/logs/rsl_rl/quadcopter_direct/2025-02-27_02-15-30/model_4550.pt'
model_path = '/home/lorenzo/Github/University/isaac_quad_sim2real/logs/rsl_rl/quadcopter_direct/2025-03-04_01-02-56/model_4999.pt'
#model_path = '/home/lorenzo/Github/University/isaac_quad_sim2real/logs/rsl_rl/quadcopter_direct/2025-03-08_16-29-22/model_800_15520.pt'
#model_path = '/home/lorenzo/Github/University/isaac_quad_sim2real/logs/rsl_rl/quadcopter_direct/2025-03-11_20-11-51/model_1650_13229.pt'

policy = RacingPolicy(quad_params, model_path, scale_output=False)
print(f"Loading model from the path {model_path}")

# Set up the figure for plotting the world
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Create the RL environment
sim_rate = 100  # Simulation runs at 100 Hz
dt = 1.0 / sim_rate  # Time step duration in seconds
window_size = 100  # Number of steps to keep
proximity_threshold = 1e-78
draw_live_plot = False
use_se3 = False

envs = [gym.make("Quadrotor-v0",
                 control_mode='cmd_ctbr',
                 quad_params=quad_params,
                 max_time=20,
                 world=None,
                 sim_rate=sim_rate,
                 render_mode='3D',
                 render_fps=60,
                 fig=fig,
                 ax=ax,
                 color='b')]

if use_se3:
    controller = SE3Control(quad_params)
    envs.append(gym.make("Quadrotor-v0",
                         control_mode='cmd_ctbr',
                         quad_params=quad_params,
                         max_time=20,
                         world=None,
                         sim_rate=sim_rate,
                         render_mode='3D',
                         render_fps=60,
                         fig=fig,
                         ax=ax,
                         color='r'))

obs_dim = envs[0].observation_space.shape[0]
action_dim = envs[0].action_space.shape[0]

# Initialize observations
options = {}
observations = [env.reset(options=options)[0] for env in envs]

# Termination conditions
terminated = [False] * len(observations)

# Desired position
pos_des = {'x': [0, 0, 0], 
           'x_dot': [0, 0, 0], 
           'x_ddot': [0, 0, 0], 
           'x_dddot': [0, 0, 0],
           'yaw': 0, 
           'yaw_dot': 0, 
           'yaw_ddot': 0}

# List to store action values and time
actions_history_window = []
actions_history = []
time_history_window = []
time_history = []

# Live plotting setup
if draw_live_plot:
    plt.ion()
    fig_live, ax_live = plt.subplots(figsize=(10, 5))
    lines = []
    labels = ["Thrust", "Roll rate", "Pitch rate", "Yaw rate"]
    colors = ['b', 'r', 'g', 'm']
    for i in range(4):
        line, = ax_live.plot([], [], label=labels[i], color=colors[i])
        lines.append(line)

    ax_live.set_xlabel("Time (s)")
    ax_live.set_ylabel("Action values")
    ax_live.legend()
    ax_live.grid(True)

timestep = 0
current_time = 0.0

# Simulation loop
while not all(terminated):
    for (i, env) in enumerate(envs):
        env.render()

        # Extract the state from the observation
        state = {'x': observations[i][0:3], 'v': observations[i][3:6], 'q': observations[i][6:10], 'w': observations[i][10:13]}

        if i == 0:
            # Check if the quadcopter is near the target position
            distance = np.linalg.norm(np.array(state['x']) - np.array(pos_des['x']))
            if distance < proximity_threshold:
                # Set a new random target position
                point_des = np.random.uniform(-3, 3, 3).tolist()
                point_des[2] = np.random.uniform(0.5, 2)
                point_des = [round(x, 2) for x in point_des]
                pos_des['x'] = point_des
                env.unwrapped.update_point(point_des)
        
                print(f"New target position: {pos_des['x']}")

            # Compute actions using the policy
            actions = policy.update(None, state, pos_des)
            # actions["cmd_w"] = -actions["cmd_w"]
            actions = np.array([actions["cmd_thrust"], *actions["cmd_w"]])

            # Store action values and time
            actions_history.append(actions)
            time_history.append(current_time)

            # Update live plot
            if draw_live_plot:
                actions_history_window.append(actions)
                time_history_window.append(current_time)

                # Keep only the last `window_size` steps
                if len(actions_history_window) > window_size:
                    actions_history_window.pop(0)
                    time_history_window.pop(0)

                actions_arr = np.array(actions_history_window)
                time_arr = np.array(time_history_window)

                for j in range(4):
                    lines[j].set_xdata(time_arr)
                    lines[j].set_ydata(actions_arr[:, j])

                ax_live.set_xlim(time_arr[0], time_arr[-1])  # Update time window
                ax_live.set_ylim(np.min(actions_arr) - 0.1, np.max(actions_arr) + 0.1)
                plt.pause(0.001)  # Update the plot
        else:
            control_dict = controller.update(0, state, pos_des)
            if env.unwrapped.control_mode == 'cmd_ctbr':
                cmd_thrust = control_dict['cmd_thrust']
                cmd_thrust = np.interp(cmd_thrust,
                                       [quad_params['num_rotors'] * env.unwrapped.min_thrust,
                                        quad_params['num_rotors'] * env.unwrapped.max_thrust],
                                       [-1,1])

                cmd_w = control_dict['cmd_w']
                cmd_w[0] = np.interp(cmd_w[0], [-env.unwrapped.max_roll_br, env.unwrapped.max_roll_br], [-1,1])
                cmd_w[1] = np.interp(cmd_w[1], [-env.unwrapped.max_pitch_br, env.unwrapped.max_pitch_br], [-1,1])
                cmd_w[2] = np.interp(cmd_w[2], [-env.unwrapped.max_yaw_br, env.unwrapped.max_yaw_br], [-1,1])

                actions = np.array([cmd_thrust, *cmd_w])

            elif env.unwrapped.control_mode == 'cmd_motor_speeds':
                # Extract the commanded motor speeds.
                cmd_motor_speeds = control_dict['cmd_motor_speeds']

                # The environment expects the control inputs to all be within the range [-1,1]
                actions = np.interp(cmd_motor_speeds, [env.unwrapped.rotor_speed_min, env.unwrapped.rotor_speed_max], [-1,1])

        # Step the environment forward
        observations[i], _, terminated[i], _, _ = env.step(actions)

        # Update time
        timestep += 1
        current_time += dt

# Disable interactive mode for final plot
plt.ioff()

# Convert to NumPy array for easier plotting
actions_history = np.array(actions_history)
time_history = np.array(time_history)

# Final static plot of action evolution
plt.figure(figsize=(10, 5))
plt.plot(time_history, actions_history[:, 0], label="Thrust", color='b')
plt.plot(time_history, actions_history[:, 1], label="Roll rate", color='r')
plt.plot(time_history, actions_history[:, 2], label="Pitch rate", color='g')
plt.plot(time_history, actions_history[:, 3], label="Yaw rate", color='m')

plt.xlabel("Time (s)")
plt.ylabel("Action values")
plt.legend()
plt.title("Evolution of actions over time")
plt.grid(True)

# Save the plot as pdf
folder_path = re.sub(r'model_\d+.*\.pt', 'figures', model_path)
os.makedirs(folder_path, exist_ok=True)
for ext in ['pdf', 'svg', 'png']:
    plt.savefig(os.path.join(folder_path, f'actions.{ext}'))

plt.show()
