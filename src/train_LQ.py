import time
from pathlib import Path
import warnings

import numpy as np
import torch
from torch.distributions.normal import Normal
from sklearn.model_selection import ParameterGrid

from cfg import parse_cfg, modify_cfg
from envs import make_env
from algorithm.models import CvxMPC
from algorithm.helper import *
import logger
__CONFIG__, __LOGS__ = 'cfgs', 'logs'

def train(cfg):
	"""main algorithm for control policy learning"""
	set_seed(cfg.seed)
	if cfg.exp_vars is None:
		work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.algo / str(cfg.seed)
	else:
		work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.algo / cfg.exp_vars / str(cfg.seed)
	train_metrics = {}
	episode_idx, start_time = 0, time.time()
	best_y, best = -np.inf, False
	L = logger.Logger(work_dir, cfg)

	env = make_env(cfg)
	agent = CvxMPC(cfg)

	if cfg.algo == "BO":
		X = torch.empty(0, agent_to_x(agent, cfg).shape[1])
		Y = torch.empty(0, 1)

	# policy update loop
	for batch in range(0, cfg.num_epi_batches):
		epi_buffer = EpisodeBuffer(cfg)

		obs = env.reset(cfg.epi_batch_size)
		episodes = [Episode(cfg, obs[i]) for i in range(cfg.epi_batch_size)]

		# select action, step environment, and store transition
		with torch.set_grad_enabled(not cfg.algo == "BO"):
			while not episodes[0].done:
				if np.any(np.abs(obs) > 150):
					warnings.warn(f"{obs[0]} may produce infeasible optimization")
				try:
					mu = agent.plan(obs)
				except:
					print(f"Infeasible optimization at {obs[0]}")
					return f"Early termination {best_y}"
				action_v = cfg.action_std * torch.randn(mu.shape)
				action = (mu + action_v).detach() 

				action_distr = Normal(mu, cfg.action_std)
				log_prob = action_distr.log_prob(action)

				dist = np.random.normal(0, cfg.dist_std, size=obs.shape)
				noise = np.random.normal(0, cfg.noise_std, size=obs.shape)
				obs, reward, done = env.step(action.numpy(), dist)

				for i in range(cfg.epi_batch_size):
					episodes[i] += (obs[i], action[i], reward[i], done, action_v[i], dist[i], noise[i], log_prob[i], None)
			
			# store episode
			for i in range(cfg.epi_batch_size):
				epi_buffer += episodes[i]
				L.finish_episode(episodes[i], episode_idx)

				episode_idx += 1
				common_metrics = {
					'episode': episode_idx,
					'total_time': time.time() - start_time,
					'episode_reward': episodes[i].cumulative_reward,
					'params': str(agent.model._dynamics.fcB.weight.flatten().detach())
				}
				train_metrics.update(common_metrics)
				L.log(train_metrics)

		# record batch information
		y = torch.mean(epi_buffer._d_cumulative_reward).reshape(1,1) # torch.tensor(np.mean(epi_buffer._d_cumulative_reward)).reshape(1,1)
		best = True if (y.item() >= best_y) else False
		best_y = y.item() if best else best_y
		L.finish_batch(agent, epi_buffer, batch, best)

		# update policy based with batch
		if cfg.algo == "REINFORCE":
			train_metrics.update(agent.epi_update(epi_buffer))
		elif cfg.algo == "BO":
			x = agent_to_x(agent, cfg)
			X, Y = update_data(x, y, X, Y)
			next_x = run_BO_iter(X, Y, cfg)
			agent = x_to_agent(next_x, agent, cfg)

	L.finish(agent)
	print('Training completed successfully')

	return best_y


if __name__ == '__main__':

	cfg_dir = Path().cwd() / __CONFIG__
	base_cfg = parse_cfg(Path().cwd() / __CONFIG__ / 'LQ_base.yaml')

	param_grid = {
		'task': ["LQ-reftrack"],
		'policy_type': ["CVXPY"],
		'init': ["A",None],
		'algo': ["REINFORCE","BO"],
		'num_epi_batches': [101],
		'seed': list(range(131,132))
	}
	param_list = list(ParameterGrid(param_grid))

	cfgs = [modify_cfg(base_cfg, params, True, cfg_dir) for params in param_list]

	for cfg in cfgs:
		train(cfg)
