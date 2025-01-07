import os
import datetime
import torch

CONSOLE_FORMAT = [('episode', 'E', 'int'), ('episode_reward', 'R', 'float'), ('total_time', 'T', 'time'), ('params', 'p', 'string')]

def make_dir(dir_path):
	"""create directory if not already existing"""
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path

class Logger(object):
	"""logger for saving transitions, models, etc."""
	def __init__(self, log_dir, cfg):
		self.log_dir = make_dir(log_dir)
		self.save_model = cfg.save_model
		self.model_dir = make_dir(self.log_dir / 'models')
		self.save_episode = cfg.save_episode
		self.episode_dir = make_dir(self.log_dir / 'episodes')
		print('-'*20)
	
	def finish_episode(self, episode, idx):
		"""optionally save episode information"""
		if self.save_episode:
			fp = self.episode_dir / f'episode{idx}.pt'
			data = {
				'obs': episode.obs,
				'dist': episode.dist,
				'noise': episode.noise,
				'action': episode.action,
				'action_v': episode.action_v,
				'reward': episode.reward,
				'cumulative_reward': episode.cumulative_reward,
				'd_cumulative_reward': episode.d_cumulative_reward
			}
			torch.save(data, fp)

	def finish_batch(self, agent, episode_buffer, idx, best):
		"""optionally save model and reward information"""
		if self.save_model:
			fp = self.model_dir / f'batch{idx}.pt'
			data = {
				'cumulative_reward': episode_buffer._cumulative_reward.detach().numpy(),
				'avg_cumulative_reward': torch.mean(episode_buffer._cumulative_reward).detach().numpy(),
				'd_cumulative_reward': episode_buffer._d_cumulative_reward.detach().numpy(),
				'avg_d_cumulative_reward': torch.mean(episode_buffer._d_cumulative_reward).detach().numpy(),
			}
			model_data = {**agent.state_dict()}
			data.update(model_data)
			torch.save(data, fp)
			if best:
				fp = self.model_dir / f'best_model.pt'
				torch.save(data, fp)

	def finish(self, agent):
		"""optionally save final model"""
		if self.save_model:
			fp = self.model_dir / f'final_model.pt'
			torch.save(agent.state_dict(), fp)
		print('-'*20)

	def _format(self, key, value, var_type):
		"""format dictionary value"""
		if var_type == 'int':
			return f'{key+":"} {int(value):,}'
		elif var_type == 'float':
			return f'{key+":"} {value:.03f}'
		elif var_type == 'time':
			value = str(datetime.timedelta(seconds=int(value)))
			return f'{key+":"} {value}'
		elif var_type == 'string':
			return f'{key+":"} {value}'
		else:
			raise f'invalid log format type: {var_type}'

	def _print(self, dictionary):
		"""format and print dictionary"""
		pieces = []
		for k, disp_k, var_type in CONSOLE_FORMAT:
			pieces.append(f'{self._format(disp_k, dictionary.get(k, 0), var_type):<16}')
		print('   '.join(pieces))

	def log(self, dictionary):
		"""log dictionary"""
		self._print(dictionary)