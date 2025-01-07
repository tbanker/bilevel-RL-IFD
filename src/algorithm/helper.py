import random
import numpy as np
import torch
import torch.nn as nn

from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize, Normalize

kwargs = {
	"device" : torch.device('cpu'),
	"dtype" : torch.float
}

def set_seed(seed):
	"""Set random elements with (seed)"""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def calc_logprob(action, mu, sigma):
	"""Compute log of probability of (a) for multivariate normal distribution N(mu, sigma)"""
	cov = torch.eye(action.shape[-1]) * sigma**2
	dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, cov)
	logprob = dist.log_prob(action)
	return logprob

def uniform_init(m):
	"""Initialize nn.Linear module (m) parameters from uniform distribution U(-1,1)"""
	if isinstance(m, nn.Linear):
		torch.nn.init.uniform_(m.weight.data,a=-1.,b=1.)
		if m.bias is not None:
			nn.init.zeros_(m.bias)

def lq_init(m):
	"""Depending on shape, set nn.Linear module (m) parameters to either A or B of LQ environment"""
	A = torch.tensor([
		[0.5, 0., 1., 0.],
		[0., 0.5, 0., 0.5],
		[0., 0., 0.5, 1.],
		[0.5, 0., 0., 0.5],
	])
	B = torch.tensor([
		[0.5],
		[0.],
		[0.],
		[0.],
	])

	with torch.no_grad():
		if isinstance(m, nn.Linear):
			if m.weight.shape == A.shape:
				m.weight.copy_(A)
			else:
				m.weight.copy_(B)		

def param_clip(m):
	"""Clip nn.Module parameters to remain within upper- (ul) and lower-limits (ll)"""
	if isinstance(m, nn.Linear): # LQ
		ul = torch.tensor([2.])
		ll = torch.tensor([-2.])
	else: # QT
		ul = torch.tensor([1., 1., 2., 2., 2., 2.])
		ll = torch.tensor([0., 0., 0., 0., 0., 0.])
	for i, (n, p) in enumerate(m.named_parameters()):
		p.data.clamp_(ll[i], ul[i])

class DLSDM(nn.Module):
	"""Discrete linear state-space dynamics model"""
	def __init__(self, x_dim, u_dim):
		super().__init__()
		self.fcA = nn.Linear(x_dim, x_dim, bias=False)
		self.fcB = nn.Linear(u_dim, x_dim, bias=False)

	def forward(self, x, u):
		return self.fcA(x) + self.fcB(u)

def q(cfg, act_fn=nn.ELU()):
	"""MLP Q-function"""
	return nn.Sequential(nn.Linear(cfg.n+cfg.m, cfg.mlp_dim), nn.LayerNorm(cfg.mlp_dim), nn.Tanh(),
						 nn.Linear(cfg.mlp_dim, cfg.mlp_dim), nn.ELU(),
						 nn.Linear(cfg.mlp_dim, 1))

class Episode(object):
	"""Object to store relevant information for singular episode by sequentially adding state transitions"""
	def __init__(self, cfg, init_obs):
		self.cfg = cfg
		self.device = torch.device('cpu')
		self.dtype = torch.float64
		self.kwargs = {"device": self.device, "dtype": self.dtype}
		# storage tensors
		self.obs = torch.empty((cfg.episode_length+1, cfg.n), **self.kwargs) # (T+1, n)
		self.dist = torch.empty((cfg.episode_length, cfg.n), **self.kwargs) # (T, n)
		self.noise = torch.empty((cfg.episode_length, cfg.n), **self.kwargs) # (T, n)
		self.obs[0] = torch.tensor(init_obs, **self.kwargs) # (b, n)
		self.action = torch.empty((cfg.episode_length, cfg.m), **self.kwargs) # (T, m)
		self.log_prob = torch.empty((cfg.episode_length, 1), **self.kwargs) # (T, 1)
		self.grad_log_prob = torch.empty((cfg.episode_length, cfg.n_params), **self.kwargs) # (T, p)
		self.action_v = torch.empty((cfg.episode_length, cfg.m), **self.kwargs) # (T, m)
		self.reward = torch.empty((cfg.episode_length, 1), **self.kwargs) # (T, 1)
		self.cumulative_reward = 0 # cumulative reward (1)
		self.d_cumulative_reward = 0 # discounted cumulative reward (1)

		self.done = False # episode termination
		self._idx = 0 # time index
	
	def __len__(self):
		return self._idx
	
	def __add__(self, transition):
		self.add(*transition)
		return self

	def add(self, obs, action, reward, done, action_v, dist, noise, log_prob = None, grad_log_prob = None):
		"""Add a singular state transition to be stored"""
		self.obs[self._idx+1] = torch.tensor(obs, **self.kwargs)
		self.dist[self._idx] = torch.tensor(dist, **self.kwargs)
		self.noise[self._idx] = torch.tensor(noise, **self.kwargs)
		self.action[self._idx] = action
		self.log_prob[self._idx] = log_prob if log_prob != None else torch.empty((1,1))
		self.grad_log_prob[self._idx] = grad_log_prob if grad_log_prob != None else torch.empty((self.cfg.m,self.cfg.n_params))
		self.action_v[self._idx] = action_v
		self.reward[self._idx] = torch.tensor(reward, **self.kwargs).item()
		self.cumulative_reward += float(reward)
		self.d_cumulative_reward += self.cfg.discount ** self._idx * float(reward)

		self.done = done
		self._idx += 1

class EpisodeBuffer():
	"""Object to store information needed to estimate gradients for multiple episode by sequentially adding episodes"""
	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device('cpu')
		self.dtype = torch.float64
		self.kwargs = {"device": self.device, "dtype": self.dtype}
		# storage tensors
		self._obs = torch.empty((self.cfg.epi_batch_size, cfg.episode_length+1, cfg.n), **self.kwargs) # (b, T+1, n)
		self._action = torch.empty((self.cfg.epi_batch_size, cfg.episode_length, cfg.m), **self.kwargs) # (b, T, m)
		self._log_prob = torch.empty((self.cfg.epi_batch_size, cfg.episode_length, cfg.m), **self.kwargs) # (b, T, 1)
		self._grad_log_prob = torch.empty((self.cfg.epi_batch_size, cfg.episode_length, cfg.n_params), **self.kwargs) # (b, T, p)
		self._reward = torch.empty((self.cfg.epi_batch_size, cfg.episode_length, 1), **self.kwargs) # (b, T, n)
		self._cumulative_reward = torch.empty((self.cfg.epi_batch_size, 1), **self.kwargs) # cumulative reward (b, 1)
		self._d_cumulative_reward = torch.empty((self.cfg.epi_batch_size, 1), **self.kwargs) # discounted cumulative reward (b, 1)

		self._idx = 0 # batch index

	def __add__(self, episode: Episode):
		self.add(episode)
		return self

	def add(self, episode: Episode):
		"""Add a singular episode to be stored"""
		self._obs[self._idx] = episode.obs
		self._action[self._idx] = episode.action
		self._log_prob[self._idx] = episode.log_prob
		self._grad_log_prob[self._idx] = episode.grad_log_prob
		self._reward[self._idx] = episode.reward
		self._cumulative_reward[self._idx] = episode.cumulative_reward
		self._d_cumulative_reward[self._idx] = episode.d_cumulative_reward

		self._idx = (self._idx + 1) % self.cfg.epi_batch_size # increment batch index (QT)
	
	def _calc_norm_d_cum_reward(self):
		"""Calculate normalized and discounted cumulative rewards"""
		# discount cumulative rewards
		discount = (self.cfg.discount ** torch.arange(0, self.cfg.episode_length)).unsqueeze(1)
		d_reward = discount * self._reward
		d_cum_reward = torch.sum(d_reward, dim=1, keepdim=True)
		# normalize cumulative rewards
		mean = torch.mean(d_cum_reward[:,0,0])
		std = torch.std(d_cum_reward[:,0,0])
		norm_d_cum_reward = (d_cum_reward - mean) / std

		return norm_d_cum_reward

	def sample(self):
		"""Sample log probs and normalized rewards"""
		norm_d_cum_reward = self._calc_norm_d_cum_reward()

		return self._log_prob, norm_d_cum_reward


	def sample_grads(self):
		"""Sample gradient of log probs and normalized rewards"""
		norm_d_cum_reward = self._calc_norm_d_cum_reward()

		return self._grad_log_prob, norm_d_cum_reward

def run_BO_iter(X, Y, cfg):
	"""Fit Gaussian process model to (X, Y) and select (next_x) by optimizing acquisition function"""
	gp, mll = create_model(X, Y)
	gp, mll = fit_model(gp, mll)
	next_x = get_next_x(gp, X, Y, cfg)
	return next_x

def create_model(X, Y):
	"""Create Gaussian process model for (X, Y)"""
	model = SingleTaskGP(train_X=X, train_Y=Y, input_transform=Normalize(d=X.shape[1]), outcome_transform=Standardize(m=1))
	mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)

	return model, mll

def fit_model(model, mll):
	"""Fit Gaussian process model"""
	fit_gpytorch_mll(mll)

	return model, mll

def get_next_x(model, X, Y, cfg):
	"""Optimize acquisition function to select (next_x)"""
	acq_fun = cfg.acq_fun
	dim = X.shape[1]

	if acq_fun == 'random':          
		next_x = cfg.bound * 2 * (torch.rand(dim) - 0.5)
	elif acq_fun == 'EI':
		EI = ExpectedImprovement(model, best_f=max(Y).item())
		next_x, _ = optimize_acqf(
			acq_function=EI,
			bounds=torch.tensor([[-cfg.bound] * dim, [cfg.bound] * dim]),
			q=1, num_restarts=20, raw_samples=100, options={},
		)
	elif acq_fun == 'UCB':
		UCB = UpperConfidenceBound(model, beta=cfg.bo_beta)
		next_x, _ = optimize_acqf(
			acq_function=UCB,
			bounds=torch.tensor([[-2.] * dim, [2.] * dim]),
			q=1, num_restarts=20, raw_samples=100, options={},
		)

	return next_x

def update_data(x, y, X, Y):
	"""Append newly observed point (x,y) to (X,Y)"""
	X = torch.cat((X, x))
	Y = torch.cat((Y, y))

	return X, Y

def agent_to_x(agent, cfg):
	"""Extract learnable parameters of (agent) policy into 1-D parameter vector (x)"""
	if cfg.policy_type == "DoMPC":
		params = torch.concat([p.data for n, p in agent.named_parameters()])
		x = params.unsqueeze(0)
	else:
		if cfg.init == "A":
			B_flat = agent.model._dynamics.fcB.weight.clone().detach().flatten()
			x = B_flat.unsqueeze(0)
		elif cfg.init == "B":
			A_flat = agent.model._dynamics.fcA.weight.clone().detach().flatten()
			x = A_flat.unsqueeze(0)
		else:
			A_flat = agent.model._dynamics.fcA.weight.clone().detach().flatten()
			B_flat = agent.model._dynamics.fcB.weight.clone().detach().flatten()
			x = torch.concat([A_flat,B_flat]).unsqueeze(0)
			
	return x

def x_to_agent(x, agent, cfg):
	"""Parameterize (agent) policy with 1-D parameter vector (x)"""
	if cfg.policy_type == "DoMPC":
		for i, (n, p) in enumerate(agent.named_parameters()):
			p.data = x[:,i]
		agent.pi = agent.construct_pi()

	else:
		A_size = sum(p.numel() for p in agent.model._dynamics.fcA.parameters())
		A_shape = agent.model._dynamics.fcA.weight.shape
		B_shape = agent.model._dynamics.fcB.weight.shape

		if cfg.init == "A": # learn B
			B = x.reshape(B_shape)
			with torch.no_grad():
				agent.model._dynamics.fcB.weight.copy_(B)
		elif cfg.init == "B": # learn A
			A = x.reshape(A_shape)
			with torch.no_grad():
				agent.model._dynamics.fcA.weight.copy_(A)
		else: # learn A and B
			A = x[:,:A_size].reshape(A_shape)
			B = x[:,A_size:].reshape(B_shape)
			with torch.no_grad():
				agent.model._dynamics.fcA.weight.copy_(A)
				agent.model._dynamics.fcB.weight.copy_(B)

	return agent
