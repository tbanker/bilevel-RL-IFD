from pathlib import Path
import numpy as np
import do_mpc
from algorithm.models import quad_tank_model
from cfg import parse_cfg

class LQ():
    """linear system with batched transitions"""
    def __init__(self, cfg):
        self.cfg = cfg
        self.n = 4
        self.m = 1
        self.A = 0.5 * np.array([
            [1., 0., 2., 0.],
            [0., 1., 0., 1.],
            [0., 0., 1., 2.],
            [1., 0., 0., 1.],
        ])  # (n, n)
        self.B = np.array([
            [0.5],
            [0.],
            [0.],
            [0.]
        ]) # (n, m)
        self.Q = 1e-2 * np.eye(self.n,self.n) # (n, n)
        self.R = 1e-2 # (m, m)

        self.x = None  # (b, 4) current state
        self.t = None # (1)
        self.T = cfg.episode_length # (1)

    def reward(self, x, u):
        """quadratic reward function rewarding states and actions near origin"""
        return -(x @ self.Q * x).sum(axis=1, keepdims=True) - u * self.R * u

    def dynamics(self, x, u, w):
        """linear dynamics"""
        self.x = x @ self.A.T + u @ self.B.T + w
        return self.x
    
    def step(self, u, w):
        """provide control input (u) and disturbance (w) and take time step"""
        u = np.array(u)
        r = self.reward(self.x, u)
        x = self.dynamics(self.x, u, w)
        self.t += 1

        done = (self.t >= self.T)

        return x, r, done

    def reset(self, batch_size):
        """reset system to initial state"""
        # batch_size refers to number of simultaneously running environments; same as batch_size of policy grad.
        self.x = np.vstack(batch_size*[np.array([2., -1.5, -2., 1.])])
        self.t = 0
        return self.x

def qt_simulator(cfg, model):
    """true quadruple tank environment simulator"""
    t_step = 5.
    period = int((cfg.episode_length * t_step) // 4) # 750 # int(3000/t_step)

    simulator = do_mpc.simulator.Simulator(model)
    params_simulator = {
        'integration_tool': 'cvodes',
        'abstol': 1e-10,
        'reltol': 1e-10,
        't_step': t_step,
    }
    simulator.set_param(**params_simulator)

    # true environment parameters
    p_num = simulator.get_p_template()
    p_num['gamma_a'] = 0.3
    p_num['gamma_b'] = 0.4
    p_num['a_1'] = 1.31 # tank 1 leak area (cm^2)
    p_num['a_2'] = 1.51 # tank 2 leak area (cm^2)
    p_num['a_3'] = 0.927 # tank 3 leak area (cm^2)
    p_num['a_4'] = 0.882 # tank 4 leak area (cm^2)
    def p_fun(t_now):
        return p_num
    simulator.set_p_fun(p_fun)

    # time-varying parameters
    h_1r = np.concatenate([ 0.65*np.ones(period+1), 0.30*np.ones(period), 0.50*np.ones(period), 0.90*np.ones(period+2*cfg.horizon) ])
    h_2r = np.concatenate([ 0.65*np.ones(period+1), 0.30*np.ones(period), 0.75*np.ones(period), 0.75*np.ones(period+2*cfg.horizon) ])
    h_3r = np.concatenate([ 0.652*np.ones(period+1), 0.301*np.ones(period), 0.305*np.ones(period), 1.062*np.ones(period+2*cfg.horizon) ])
    h_4r = np.concatenate([ 0.664*np.ones(period+1), 0.305*np.ones(period), 1.200*np.ones(period), 0.579*np.ones(period+2*cfg.horizon) ])
    tvp_template = simulator.get_tvp_template()
    def tvp_fun(t_now):
        tvp_template['h_1r'] = h_1r[int(t_now)]
        tvp_template['h_2r'] = h_2r[int(t_now)]
        tvp_template['h_3r'] = h_3r[int(t_now)]
        tvp_template['h_4r'] = h_4r[int(t_now)]
        return tvp_template
    simulator.set_tvp_fun(tvp_fun)

    simulator.setup()

    return simulator

class QT():
    """quadruple tanke system with non-batched transitions"""
    def __init__(self, cfg, model):
        self.n = 4
        self.m = 2
        self.simulator = qt_simulator(cfg, model)
        self.estimator = do_mpc.estimator.StateFeedback(model)
        self.Q = np.eye(4,4) # (n,n)

        self.x = None # (1, n) current state
        self.x_p = None # (1, n) prev state
        self.y = None # (1, n) current obs
        self.y_p = None # (1, n) prev obs
        self.t = None # (1)
        self.T = cfg.episode_length # (1)
    
    def reward(self):
        """quadratic reward function rewarding states near reference"""
        return - np.array([self.simulator.data._aux[-1,-4:].T @ self.Q @ self.simulator.data._aux[-1,-4:]])
    
    def dynamics(self, u, w, v):
        """quadruple tank dynamics"""
        self.x_p = self.x.copy()
        self.y_p = self.y.copy()
        # with silence_output():
        y_next = self.simulator.make_step(u.T, v0=v.T, w0=w.T).T
            # x = # self.estimator.make_step(y_next).reshape(-1)
        self.x = y_next.copy() - v.T
        self.y = y_next.copy()

        return self.y
    
    def step(self, u, w=None, v=None):
        """provide control input (u), disturbance (w), and noise (v) and take time step"""
        y = self.dynamics(u, w, v)
        r = self.reward()
        self.t += 1

        done = (self.t >= self.T)

        return y, r, done
    
    def reset(self, batch_size=1):
        """reset system to initial state"""
        # batch_size refers to number of simultaneously running environments; not batch_size of policy grad.
        assert batch_size == 1
        h_1s_0 = 0.5 # (m)
        h_2s_0 = 0.5 # (m)
        h_3s_0 = 0.5 # (m)
        h_4s_0 = 0.5 # (m)
        x0 = np.array([[h_1s_0, h_2s_0, h_3s_0, h_4s_0]])
        self.simulator.x0 = x0.T
        self.simulator.t0 = np.array([0.])
        self.estimator.x0 = x0.T
        self.x = x0.copy() # x0.reshape(1,-1)
        self.x_p = x0.copy() # x0.reshape(1,-1)
        self.y = x0.copy() # x0.reshape(1,-1)
        self.y_p = x0.copy() # x0.reshape(1,-1)
        self.t = 0
        return self.x

def make_env(cfg, model=None):
    """create environment (env), optionally using provided quadruple tank model (model)"""
    if "LQ" in cfg.task:
        env = LQ(cfg)
    elif "QT" in cfg.task:
        env = QT(cfg, model)

    cfg.n = env.n
    cfg.m = env.m

    return env

if __name__ == "__main__":
    work_dir = Path().cwd()
    cfg_dir = work_dir / "cfgs"
    cfg_path = cfg_dir / "_base.yaml"
    cfg = parse_cfg(cfg_path)
    env = LQ(cfg)
    x0 = env.reset(cfg.epi_batch_size)
    u = np.random.randn(64,1)
    w = 0.01 * np.random.randn(64,1)
    x1, r, done = env.step(u, w)

    cfg_path = cfg_dir / "QT_base.yaml"
    cfg = parse_cfg(cfg_path)
    model = quad_tank_model()
    env = QT(cfg, model)
    x0 = env.reset(cfg.epi_batch_size)
    u = np.random.randn(1,2)
    x1, r, done = env.step(u)
    