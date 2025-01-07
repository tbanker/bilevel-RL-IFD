import time
from pathlib import Path

import numpy as np
import torch
from torch.distributions.normal import Normal

from cfg import parse_cfg
from envs import make_env
from algorithm.models import QTMPC, quad_tank_model
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

	model = quad_tank_model()
	env = make_env(cfg, model)
	agent = QTMPC(cfg, model)

	if cfg.algo == "BO":
		X = torch.empty(0, agent_to_x(agent, cfg).shape[1])
		Y = torch.empty(0, 1)

	# policy update loop
	for batch in range(0, cfg.num_epi_batches):
		epi_buffer = EpisodeBuffer(cfg)

		for i in range(cfg.epi_batch_size):

			obs = env.reset(batch_size=1)
			episode = Episode(cfg, obs[0])

			agent.pi.t0 = np.array([0.])
			agent.pi.x0 = obs.T
			agent.pi.set_initial_guess() 

			# select action, step environment, and store transition
			with torch.set_grad_enabled(not cfg.algo == "BO"):
				while not episode.done:
					# 1/01; try:
					mu = agent.plan(obs)
					# 1/01; except:
					# 1/01; 	return f"Early termination {best_y}: Policy could not be solved @ U{batch} E{episode_idx}"
					action_v = cfg.action_std * torch.randn(mu.shape)
					action = (torch.tensor(mu) + action_v).detach() 

					grad_log_prob = agent.grad_log_prob(action, mu)

					dist = np.random.normal(0, cfg.dist_std, size=obs.shape)
					noise = np.random.normal(0, cfg.noise_std, size=obs.shape)
					obs, reward, done = env.step(action.numpy(), dist, noise)

					episode += (obs, action, reward, done, action_v, dist, noise, None, grad_log_prob)

				# store episode
				epi_buffer += episode
				L.finish_episode(episode, episode_idx)

				episode_idx += 1
				common_metrics = {
					'episode': episode_idx,
					'total_time': time.time() - start_time,
					'episode_reward': episode.cumulative_reward,
					'params': str([np.round(p.item(),6) for n, p in agent.named_parameters()])
				}
				train_metrics.update(common_metrics)
				L.log(train_metrics)

		# record batch information
		y = torch.mean(epi_buffer._d_cumulative_reward).reshape(1,1)
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
	for i in range(100):
		cfg = parse_cfg(Path().cwd() / __CONFIG__ / 'QT_base.yaml')
		cfg.seed = i
		train(cfg)
