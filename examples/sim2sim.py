import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from rotorpy.vehicles.crazyflie_params import quad_params

from rotorpy.learning.quadrotor_reward_functions import hover_reward

from rotorpy.controllers.policy_controller import PolicyControl

# Load the model for the appropriate epoch
model_path = '/home/lorenzo/Github/University/IsaacLab/logs/rsl_rl/quadcopter_direct/2025-02-27_02-15-30/model_4550.pt'
policy = PolicyControl(quad_params, model_path)
print(f"Loading model from the path {model_path}")

# Set up the figure for plotting all the agents.
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Make the environments for the RL agents
num_quads = 1
def make_env():
    return gym.make("Quadrotor-v0",
                    control_mode ='cmd_ctbr',
                    quad_params = quad_params,
                    max_time = 20,
                    world = None,
                    sim_rate = 100,
                    render_mode='3D',
                    render_fps = 60,
                    fig=fig,
                    ax=ax,
                    color='b')
envs = [make_env() for _ in range(num_quads)]

obs_dim = envs[0].observation_space.shape[0]
action_dim = envs[0].action_space.shape[0]

# Collect observations for each environment
options = {}
# options['initial_state'] = 'deterministic'
observations = [env.reset(options=options)[0] for env in envs]

# This is a list of env termination conditions so that the loop only ends when the final env is terminated
terminated = [False] * len(observations)

pos_des = {'x': [0, 0, 0], 
           'x_dot': [0, 0, 0], 
           'x_ddot': [0, 0, 0], 
           'x_dddot': [0, 0, 0],
           'yaw': 0, 
           'yaw_dot': 0, 
           'yaw_ddot': 0}

proximity_threshold = 0.1
first_time = True
while not all(terminated):
    for (i, env) in enumerate(envs):  # For each environment...
        env.render()
        if first_time:
            first_time = False
            input()

        # Unpack the observation from the environment
        state = {'x': observations[i][0:3], 'v': observations[i][3:6], 'q': observations[i][6:10], 'w': observations[i][10:13]}

        # compute distance from target
        distance = np.linalg.norm(np.array(state['x']) - np.array(pos_des['x']))
        if distance < proximity_threshold:
            point_des = np.random.uniform(-3, 3, 3).tolist()
            point_des[2] = np.random.uniform(0.5, 2)
            point_des = [round(x, 2) for x in point_des]
            pos_des['x'] = point_des
            env.unwrapped.update_point(point_des)
    
            print(f"New target position: {pos_des['x']}")

        actions = policy.update(None, state, pos_des)
        actions = np.array([actions["cmd_thrust"], *actions["cmd_w"]])

        # Step the environment forward
        observations[i], _, terminated[i], _, _ = env.step(actions)
