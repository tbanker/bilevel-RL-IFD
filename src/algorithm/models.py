import numpy as np
import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import do_mpc
from casadi import *
from casadi.tools import *

import sys
import os
from contextlib import redirect_stdout, redirect_stderr, contextmanager
sys.path.append('../../')

import algorithm.helper as h

@contextmanager
def silence_output():
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull) as devnull, redirect_stderr(sys.stdout):
            yield

class QDyn(nn.Module):
	"""nn.Module containing all learnable models (dynamics & Q-function)"""
	def __init__(self, cfg):
		super().__init__()
		self._dynamics = h.DLSDM(cfg.n, cfg.m) #12/31; h.dlsdm(cfg.n, cfg.m)
		self._Q1, self._Q2 = h.q(cfg), h.q(cfg)
		self.apply(h.uniform_init)
		if cfg.init == "A":
			self._dynamics.fcA.apply(h.lq_init)
		elif cfg.init == "B":
			self._dynamics.fcB.apply(h.lq_init)
		elif cfg.init == "AB":
			self._dynamics.fcA.apply(h.lq_init)
			self._dynamics.fcB.apply(h.lq_init)

	def dynamics(self, s, a):
		"""Return next state for given state-action pair (s, a)"""
		return self._dynamics(s, a)

class CvxMPC(nn.Module):
	"""(Convex) Optimization-based policy"""
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device('cpu')
		self.dtype = torch.float64
		self.kwargs = {"device": self.device, "dtype": self.dtype}

		self.model = QDyn(cfg).to(self.dtype)
		self.A = list(self.model._dynamics.parameters())[0]
		self.B = list(self.model._dynamics.parameters())[1]
		self.Q_cost = 1e-2 # state cost (Q) scaling
		self.R_cost = 1e-2 # control cost (Q) scaling
		self.pi = self.construct_pi()

		if cfg.init == "A": # learn B
			self.pi_optim = torch.optim.Adam(self.model._dynamics.fcB.parameters(), lr=self.cfg.lr)
		elif cfg.init == "B": # learn A
			self.pi_optim = torch.optim.Adam(self.model._dynamics.fcA.parameters(), lr=self.cfg.lr)
		else: # learn A and B
			self.pi_optim = torch.optim.Adam(self.model._dynamics.parameters(), lr=self.cfg.lr)
		self.model.eval()

	def construct_pi(self):
		"""Create optimization problem to be solved at each time instance"""
		# set x, A, B, Q, R
		x = cp.Parameter(self.cfg.n)
		A = cp.Parameter((self.cfg.n,self.cfg.n))
		B = cp.Parameter((self.cfg.n,self.cfg.m))
		Q = self.Q_cost * np.eye(self.cfg.n,self.cfg.n)
		if self.cfg.m == 1:
			R = self.R_cost
		else:
			R = self.R_cost * np.eye(self.cfg.m,self.cfg.m)

		# initial states, controls, constraints, and objective
		states = [cp.Variable(self.cfg.n) for _ in range(self.cfg.horizon)]
		controls = [cp.Variable(self.cfg.m) for _ in range(self.cfg.horizon)]
		if self.cfg.u_lim != None:
			constraints = [states[0] == x, cp.norm(controls[0], 'inf') <= self.cfg.u_lim]
		else:
			constraints = [states[0] == x]
		if self.cfg.m == 1:
			objective = cp.quad_form(states[0],Q) + cp.multiply(cp.square(controls[0]),R)
		else:
			objective = cp.quad_form(states[0],Q) + cp.quad_form(controls[0],R)

		# predicted states, controls, constraints, and objective across horizon
		for t in range(1, self.cfg.horizon):
			if self.cfg.m == 1:
				objective += cp.quad_form(states[t],Q) + cp.multiply(cp.square(controls[t]),R)
			else:
				objective += cp.quad_form(states[t],Q) + cp.quad_form(controls[t],R)
			constraints += [states[t] == A @ states[t-1] + B @ controls[t-1]]
			if self.cfg.u_lim != None:
				constraints += [cp.norm(controls[t], 'inf') <= self.cfg.u_lim]
		problem = cp.Problem(cp.Minimize(objective), constraints)
		
		return CvxpyLayer(problem, variables=[controls[0]], parameters=[x, A, B])

	def plan(self, obs):
		"""Solve optimization for state (obs) to obtain control input (u)"""
		
		obs = torch.from_numpy(obs).to(torch.float64)
		action = self.pi(obs, self.A, self.B, solver_args={"max_iters": 100000})[0]

		return action # (b, m)
	
	def state_dict(self):
		"""Generate state dict of policy"""
		full_dict = self.model.state_dict()
		select_keys = ["_dynamics.fcA.weight", "_dynamics.fcB.weight"]
		selected_dict = {key: full_dict[key] for key in select_keys if key in full_dict}
		return {'model': selected_dict}

	def save(self, fp):
		"""Save state dict of policy to path (fp)"""
		torch.save(self.state_dict(), fp)

	def epi_update(self, episode_buffer):
		"""Estimate gradient, take gradient step, and keep within bounds"""
		self.model.train()
		log_probs, rewards = episode_buffer.sample()

		# Estimate gradient and take step
		self.pi_optim.zero_grad(set_to_none=True)
		pi_loss = torch.sum(-log_probs * rewards, axis=(0,1)) / self.cfg.epi_batch_size
		pi_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model._dynamics.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		self.pi_optim.step()

		# Clip to be within bounds
		self.model._dynamics.fcA.apply(h.param_clip)
		self.model._dynamics.fcB.apply(h.param_clip)

		self.model.eval()

		return {'pi_loss': pi_loss.item()}

def quad_tank_model(symvar_type='SX'):
	"""nonlinear quadruple tank model"""
	model_type = 'continuous'
	model = do_mpc.model.Model(model_type, symvar_type)

	# certain parameters
	S_c = 0.06 # cross section (m^2)
	g = 9.81 # gravity (m/s^2)

	# fixed parameters:
	gamma_a = model.set_variable('_p',  'gamma_a')
	gamma_b = model.set_variable('_p',  'gamma_b')
	a_1 = model.set_variable('_p',  'a_1')
	a_2 = model.set_variable('_p',  'a_2')
	a_3 = model.set_variable('_p',  'a_3')
	a_4 = model.set_variable('_p',  'a_4')

	# time-varying parameters:
	h_1r = model.set_variable('_tvp', 'h_1r')
	h_2r = model.set_variable('_tvp', 'h_2r')
	h_3r = model.set_variable('_tvp', 'h_3r')
	h_4r = model.set_variable('_tvp', 'h_4r')

	# states (optimization variables):
	h_1s = model.set_variable('_x',  'h_1s')  # tank 1 height (m)
	h_2s = model.set_variable('_x',  'h_2s')  # tank 2 height (m)
	h_3s = model.set_variable('_x',  'h_3s')  # tank 3 height (m)
	h_4s = model.set_variable('_x',  'h_4s')  # tank 4 height (m)

	# inputs (optimization variables):
	q_a = model.set_variable('_u',  'q_a') # valve a flow rate  (m^3/hr)
	q_b = model.set_variable('_u',  'q_b') # valve b flow rate  (m^3/hr)

	# model differential equations
	model.set_rhs('h_1s', -a_1*1e-4/S_c*(2*g*h_1s)**(1/2) + a_3*1e-4/S_c*(2*g*h_3s)**(1/2) + gamma_a*q_a/(S_c*3600), process_noise = True)
	model.set_rhs('h_2s', -a_2*1e-4/S_c*(2*g*h_2s)**(1/2) + a_4*1e-4/S_c*(2*g*h_4s)**(1/2) + gamma_b*q_b/(S_c*3600), process_noise = True)
	model.set_rhs('h_3s', -a_3*1e-4/S_c*(2*g*h_3s)**(1/2) + (1-gamma_b)*q_b/(S_c*3600), process_noise = True)
	model.set_rhs('h_4s', -a_4*1e-4/S_c*(2*g*h_4s)**(1/2) + (1-gamma_a)*q_a/(S_c*3600), process_noise = True)

	# reference and error
	x_r = vertcat(model.tvp['h_1r'], model.tvp['h_2r'], model.tvp['h_3r'], model.tvp['h_4r'])
	model.set_expression('x_r', x_r)
	model.set_expression('x_e', model.x - x_r)

	model.setup()

	return model

def linear_quad_tank_model(symvar_type='SX'):
	"""linearized quadruple tank model"""
	model_type = 'continuous'
	model = do_mpc.model.Model(model_type, symvar_type)

	# certain parameters
	S_c = 0.06 # cross section (m^2)
	g = 9.81 # gravity (m/s^2)
	h_1lp = 0.65
	h_2lp = 0.65
	h_3lp = 0.652
	h_4lp = 0.664

	# fixed parameters:
	gamma_a = model.set_variable('_p',  'gamma_a')
	gamma_b = model.set_variable('_p',  'gamma_b')
	a_1 = model.set_variable('_p',  'a_1')
	a_2 = model.set_variable('_p',  'a_2')
	a_3 = model.set_variable('_p',  'a_3')
	a_4 = model.set_variable('_p',  'a_4')

	# time-varying parameters:
	h_1r = model.set_variable('_tvp', 'h_1r')
	h_2r = model.set_variable('_tvp', 'h_2r')
	h_3r = model.set_variable('_tvp', 'h_3r')
	h_4r = model.set_variable('_tvp', 'h_4r')

	# states (optimization variables):
	h_1s = model.set_variable('_x',  'h_1s')  # tank 1 height (m)
	h_2s = model.set_variable('_x',  'h_2s')  # tank 2 height (m)
	h_3s = model.set_variable('_x',  'h_3s')  # tank 3 height (m)
	h_4s = model.set_variable('_x',  'h_4s')  # tank 4 height (m)

	# input (optimization variables):
	q_a = model.set_variable('_u',  'q_a') # valve a flow rate  (m^3/hr)
	q_b = model.set_variable('_u',  'q_b') # valve b flow rate  (m^3/hr)

	# model differential equations
	model.set_rhs('h_1s', -a_1*1e-4*h_1s/(S_c*(2*h_1lp/g)**(1/2)) + a_3*1e-4*h_3s/(S_c*(2*h_3lp/g)**(1/2)) + gamma_a*q_a/(S_c*3600))
	model.set_rhs('h_2s', -a_2*1e-4*h_2s/(S_c*(2*h_2lp/g)**(1/2)) + a_4*1e-4*h_4s/(S_c*(2*h_4lp/g)**(1/2)) + gamma_b*q_b/(S_c*3600))
	model.set_rhs('h_3s', -a_3*1e-4*h_3s/(S_c*(2*h_3lp/g)**(1/2)) + (1-gamma_b)*q_b/(S_c*3600))
	model.set_rhs('h_4s', -a_4*1e-4*h_4s/(S_c*(2*h_4lp/g)**(1/2)) + (1-gamma_a)*q_a/(S_c*3600))

	# reference and error
	x_r = vertcat(model.tvp['h_1r'], model.tvp['h_2r'], model.tvp['h_3r'], model.tvp['h_4r'])
	model.set_expression('x_r', x_r) #x_r)
	model.set_expression('x_e', model.x - x_r)

	model.setup()

	return model

class QTMPC(nn.Module):
	"""Optimization-based policy with nonlinear model and constraints"""
	def __init__(self, cfg, model):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device('cpu')
		self.dtype = torch.float64
		self.kwargs = {"device": self.device, "dtype": self.dtype}
		self.diffkwargs = {'check_LICQ': False, 'check_rank': False, 'lin_solver': 'casadi', 'lstsq_fallback': True}

		self.model = model
		self.gamma_a = nn.Parameter(1*torch.rand(1, requires_grad=True, **self.kwargs)) 
		self.gamma_b = nn.Parameter(1*torch.rand(1, requires_grad=True, **self.kwargs)) 
		self.a_1 = nn.Parameter(2*torch.rand(1, requires_grad=True, **self.kwargs)) 
		self.a_2 = nn.Parameter(2*torch.rand(1, requires_grad=True, **self.kwargs)) 
		self.a_3 = nn.Parameter(2*torch.rand(1, requires_grad=True, **self.kwargs)) 
		self.a_4 = nn.Parameter(2*torch.rand(1, requires_grad=True, **self.kwargs)) 
		self.t_step = 5.
		self.period = int((cfg.episode_length * self.t_step) // 4)
		self.h_1r = np.concatenate([ 0.65*np.ones(self.period+1),0.30*np.ones(self.period), 0.50*np.ones(self.period), 0.90*np.ones(self.period+2*self.cfg.horizon) ])
		self.h_2r = np.concatenate([ 0.65*np.ones(self.period+1), 0.30*np.ones(self.period), 0.75*np.ones(self.period), 0.75*np.ones(self.period+2*self.cfg.horizon) ])
		self.h_3r = np.concatenate([ 0.652*np.ones(self.period+1), 0.301*np.ones(self.period), 0.305*np.ones(self.period), 1.062*np.ones(self.period+2*self.cfg.horizon) ])
		self.h_4r = np.concatenate([ 0.664*np.ones(self.period+1), 0.305*np.ones(self.period), 1.200*np.ones(self.period), 0.579*np.ones(self.period+2*self.cfg.horizon) ])
		self.Q = np.eye(4,4)
		self.pi = self.construct_pi()
		
		self.nlp_diff = do_mpc.differentiator.DoMPCDifferentiator(self.pi, **self.diffkwargs)
		self.pi_optim = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)

	def construct_pi(self):
		"""Create optimization problem to be solved at each time instance"""
		mpc = do_mpc.controller.MPC(self.model)
		setup_mpc = {
			'n_horizon': self.cfg.horizon,
			'n_robust': 0,
			'open_loop': False,
			't_step': self.t_step,
			'state_discretization': 'collocation',
			'collocation_type': 'radau',
			'collocation_deg': 2,
			'collocation_ni': 2,
			'store_full_solution': True,
			'nlpsol_opts': {'ipopt.fixed_variable_treatment': 'make_constraint','ipopt.tol': 1e-16}
		}
		mpc.set_param(**setup_mpc)
		mpc.settings.supress_ipopt_output()

		# time-varying parameters
		tvp_template = mpc.get_tvp_template()
		def tvp_fun(t_now):
			for k in range(self.cfg.horizon+1):
					tvp_template['_tvp',k,'h_1r'] = self.h_1r[int(t_now)+k]
					tvp_template['_tvp',k,'h_2r'] = self.h_2r[int(t_now)+k]
					tvp_template['_tvp',k,'h_3r'] = self.h_3r[int(t_now)+k]
					tvp_template['_tvp',k,'h_4r'] = self.h_4r[int(t_now)+k]
			return tvp_template
		mpc.set_tvp_fun(tvp_fun)

		# objective
		lterm = transpose(self.model.aux['x_e']) @ self.Q @ self.model.aux['x_e']
		mterm = DM(np.zeros((1,1)))
		mpc.set_objective(lterm=lterm, mterm=mterm)
		mpc.set_rterm(q_a=0.1)
		mpc.set_rterm(q_b=0.1)

		# constraints
		mpc.bounds['lower', '_x', 'h_1s'] = 0.2
		mpc.bounds['lower', '_x', 'h_2s'] = 0.2
		mpc.bounds['lower', '_x', 'h_3s'] = 0.2
		mpc.bounds['lower', '_x', 'h_4s'] = 0.2
		mpc.bounds['upper', '_x', 'h_1s'] = 1.36
		mpc.bounds['upper', '_x', 'h_2s'] = 1.36
		mpc.bounds['upper', '_x', 'h_3s'] = 1.30
		mpc.bounds['upper', '_x', 'h_4s'] = 1.30
		mpc.bounds['lower','_u','q_a'] = 0.
		mpc.bounds['lower','_u','q_b'] = 0.
		mpc.bounds['upper','_u','q_a'] = 3.26
		mpc.bounds['upper','_u','q_b'] = 4.

		# learnable parameters
		mpc.set_uncertainty_values(
			gamma_a = self.gamma_a.detach().numpy(),
			gamma_b = self.gamma_b.detach().numpy(),
			a_1 = self.a_1.detach().numpy(),
			a_2 = self.a_2.detach().numpy(),
			a_3 = self.a_3.detach().numpy(),
			a_4 = self.a_4.detach().numpy(),
		)

		# with silence_output():
		mpc.setup()

		return mpc

	def plan(self, obs): # 1/01; , eval_mode=False, step=None, t0=True):
		"""Solve optimization for state (obs) to obtain control input (u)"""
		# with silence_output():
		action = self.pi.make_step(obs.T)

		return action.T # (1, m)

	def grad_log_prob(self, action, mu):
		"""Implicitly differentiate policy at solution"""
		with silence_output():
			self.nlp_diff.differentiate()
		dudp_num = np.array(self.nlp_diff.sens_num["dxdp", indexf["_u",0,0], indexf["_p"]])
		grad_log_prob = torch.autograd.functional.jacobian(h.calc_logprob, (action, torch.tensor(mu, requires_grad=True), torch.tensor([self.cfg.action_std])))[1][0]
		grad_log_prob = grad_log_prob @ torch.tensor(dudp_num, **self.kwargs)
		
		return torch.where(torch.isnan(grad_log_prob), 0., grad_log_prob) # (1, p)
	
	def state_dict(self):
		"""Generate state dict of policy"""
		return {
			'gamma_a': self.gamma_a,
			'gamma_b': self.gamma_b,
			'a_1': self.a_1,
			'a_2': self.a_2,
			'a_3': self.a_3,
			'a_4': self.a_4,
		}

	def save(self, fp):
		"""Save state dict of policy to path (fp)."""
		torch.save(self.state_dict(), fp)
	
	def epi_update(self, episode_buffer):
		"""Main update function for episode-based learning."""
		grad_log_probs, rewards = episode_buffer.sample_grads()

		# Estimate gradient and take step
		self.pi_optim.zero_grad(set_to_none=True)
		grad_J = torch.sum(-grad_log_probs * rewards, axis=(0,1)) / self.cfg.epi_batch_size
		with torch.no_grad():
			for i, (n, p) in enumerate(self.named_parameters()):
				p.grad = grad_J[i:i+1]
		torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		self.pi_optim.step()

		# Clip to be within bounds
		h.param_clip(self)

		self.pi = self.construct_pi() # remake policy w/ new parameters
		return {'grad_J': grad_J}