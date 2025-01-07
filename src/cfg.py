import re
import copy
from pathlib import Path
from omegaconf import OmegaConf

def parse_cfg(cfg_path: str):
	"""read configuration file"""
	base = OmegaConf.load(cfg_path)
	cli = OmegaConf.from_cli()
	for k,v in cli.items():
		if v == None:
			cli[k] = True
	base.merge_with(cli)

	return base

def modify_cfg(cfg, params, save, save_dir):
	"""update configuration file with dictionary of (params)"""
	mod_cfg = copy.deepcopy(cfg)
	for key, value in params.items():
		OmegaConf.update(mod_cfg, key, value)
	if "seed" in params.keys(): # no need to save same cfg for every seed
		params.pop("seed")
	exp_vars = re.sub(r"[{}':, ]+", '_', str(params))[1:-1]
	OmegaConf.update(mod_cfg, "exp_vars", exp_vars)

	if save:
		fn = f"{cfg['exp_name']}_{exp_vars}{'.yaml'}"
		fp = save_dir / fn
		with open(fp, "w") as f:
			OmegaConf.save(mod_cfg, f)

	return mod_cfg

if __name__ == "__main__":
	work_dir = Path().cwd()
	cfg_dir = work_dir / "cfgs"
	cfg_path = cfg_dir / "LQ_base.yaml"
	cfg = parse_cfg(cfg_path)
	print(cfg)