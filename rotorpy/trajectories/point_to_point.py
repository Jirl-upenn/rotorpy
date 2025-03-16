import numpy as np

class Point2Point(object):
    """
    This trajectory updates the desired point when the
    quadrotor is close.
    """
    def __init__(self, x0=np.array([0, 0, 0]), t_change_target=5):
        """
        This is the constructor for the Trajectory object. A fresh trajectory
        object will be constructed before each mission.
        """

        self.x0 = x0
        self.t_change_target = t_change_target
        self.t0 = 0.0

    def update(self, t, x=None):
        """
        Given the present time and drone state return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        if t - self.t0 > self.t_change_target:
            self.t0 = t
            self.x0 = np.random.uniform(-3, 3, 3)
            self.x0[2] = np.random.uniform(0.5, 2)
            self.x0 = [round(x, 2) for x in self.x0]
            print(f"New target position: {self.x0}")

        x = self.x0
        x_dot = np.zeros((3,))
        x_ddot = np.zeros((3,))
        x_dddot = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw    = 0
        yaw_dot = 0
        yaw_ddot = 0

        flat_output = {'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                       'yaw':yaw, 'yaw_dot':yaw_dot, 'yaw_ddot':yaw_ddot}
        return flat_output
