import numpy as np
import scipy
import torch

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import casadi as ca
import do_mpc

plt.style.use('visualization/ieee.mplstyle') 
plt.rcParams['text.usetex'] = True
lw = 1
ms = 2.
mew = 0.1
alpha_sym = 0.5
J_colors = list(mcolors.TABLEAU_COLORS.values())[:2]
cm_colors = [(0.86, 0.86, 0.86), (0, 0, 0)]
cust_cm = mcolors.LinearSegmentedColormap.from_list("Custom", cm_colors, N=256)
mpl.colormaps.register(cmap=cust_cm, name="cust_cm")
cmap = "cust_cm" 
LQ_traj_colors = (np.interp(np.linspace(0,1,11), [0,1], [0.86,0]).reshape(-1,1) * np.ones((11,3))).tolist()
QT_traj_colors = (np.interp(np.linspace(0,1,5), [0,1], [0.86,0]).reshape(-1,1) * np.ones((5,3))).tolist()
t_colors = [list(mcolors.TABLEAU_COLORS.values())[8],
          list(mcolors.TABLEAU_COLORS.values())[9]]
markers = ['o','s','*','x','D'][:1]

def cummax(arr):
    """compute the cumulative maximum"""
    best = -np.inf
    arr_sort = np.empty(arr.shape)
    for i, r in enumerate(arr):
        arr_sort[i] = r if r >= best else best
        best = r if r >= best else best
    return arr_sort

def mean_confidence_interval(arr, confidence=0.95):
    """compute mean and confidence interval"""
    n = arr.shape[-1]
    m, se = np.mean(arr, axis=-1), scipy.stats.sem(arr, axis=-1)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def check_seed_validity(base_dir, methods, batch_sizes, n_seeds, n_batches):
    """remove any seeds where learning terminated early, i.e., policy became infeasible"""
    seeds = np.arange(n_seeds)
    seed_validity = np.zeros((len(methods), len(batch_sizes), n_seeds), dtype=bool)

    for i, method in enumerate(methods):
        for j, batch_size in enumerate(batch_sizes):
            dir = base_dir / method
            try:
                for subdir in dir.iterdir():
                    try:
                        if len(list((subdir / "models").glob("batch*.pt"))) >= n_batches:
                            seed_validity[i,j,int(subdir.name)] = True
                    except:
                        pass
            except:
                pass 

    valid_mask = np.all(seed_validity, axis=(0,1))
    valid_seeds = seeds[valid_mask]

    return valid_seeds

def get_results(base_dir, methods, batch_sizes, batches, valid_seeds, n_batches):
    """load and average cumulative reward observations for all batches and seeds"""
    res = np.empty((len(methods), len(batch_sizes), n_batches, len(valid_seeds)))

    for i, method in enumerate(methods):
        for j, batch_size in enumerate(batch_sizes):
            for batch in batches:
                d_avg_cumulative_rewards = np.empty(0)
                for seed in valid_seeds:
                    fp = base_dir / method / str(seed) / "models" / f"batch{batch}.pt"
                    d_cumulative_reward = torch.load(fp)["d_cumulative_reward"]
                    d_avg_cumulative_rewards = np.append(d_avg_cumulative_rewards, np.mean(d_cumulative_reward))
                res[i, j, batch, :] = d_avg_cumulative_rewards

    return res

def sort_results(res, methods, batch_sizes, valid_seeds):
    """sort results such that they represent the largest observed cumulative reward at each batch"""
    sorted_res = np.empty_like(res)
    
    for i, method in enumerate(methods):
        for j, batch_size in enumerate(batch_sizes):
            for l, seed in enumerate(valid_seeds):
                sorted_res[i,j,:,l] = cummax(res[i,j,:,l])

    return sorted_res

def get_trajs(base_dir, method, batch_size, batches, seed, T, n, m):
    """load trajectories for LQ case study"""
    xs = torch.empty(len(batches), batch_size, T+1, n)
    us = torch.empty(len(batches), batch_size, T, m)
    rs = torch.empty(len(batches), batch_size, T, 1)

    for i, batch in enumerate(batches):
        for j, episode in enumerate(range(batch_size)):
            fp = base_dir / method / str(seed) / "episodes" / f"episode{batch_size * batch + episode}.pt"
            xs[i, j] = torch.load(fp)["obs"]
            us[i, j] = torch.load(fp)["action"]
            rs[i, j] = torch.load(fp)["reward"]

    return xs, us, rs

def get_matrices(base_dir, methods, batch_sizes, batches, valid_seeds):
    """load learned A and B matrices for all batches and seeds"""
    res_A = np.empty((len(methods), len(batch_sizes), len(batches), len(valid_seeds), 4, 4))
    res_B = np.empty((len(methods), len(batch_sizes), len(batches), len(valid_seeds), 4, 1))

    for i, method in enumerate(methods):
        for j, batch_size in enumerate(batch_sizes):
            for k, batch in enumerate(batches):
                for l, seed in enumerate(valid_seeds):
                    fp = base_dir / method / str(seed) / "models" / f"batch{batch}.pt"
                    res = torch.load(fp)
                    res_A[i, j, k, l] = res["model"]["_dynamics.fcA.weight"].numpy()
                    res_B[i, j, k, l] = res["model"]["_dynamics.fcB.weight"].numpy()
                    
    return res_A, res_B

def calc_norm(res_A, res_B, A, B, norm="fro"):
    """calculate (frobenius) norm between learned and true A and B matrices"""
    diff_A = res_A - A
    diff_B = res_B - B
    
    frob_A = np.linalg.norm(diff_A, ord=norm, axis=(-1,-2))
    frob_B = np.linalg.norm(diff_B, ord=norm, axis=(-1,-2))
    
    mean_frob_A, err_frob_A = mean_confidence_interval(frob_A)
    mean_frob_B, err_frob_B = mean_confidence_interval(frob_B)
    
    return mean_frob_A, mean_frob_B, err_frob_A, err_frob_B

def plot_2xJ(batches, data1, plot_methods1, data2, plot_methods2, batch_sizes, fn):
    """plot regret for  case study with data1 and data2 corresponding to low- and high-dimension problem"""
    fig, axes = plt.subplots(1,2, layout='constrained', figsize=(7,3.33), sharey=True, dpi=1200)
    plt_tuples = zip([plot_methods1, plot_methods2], [data1, data2])

    for k, (plot_methods, data) in enumerate(plt_tuples):
        sorted_avgs = data["avgs"]
        sorted_errs = data["errs"]

        x = batches
        for i, method in enumerate(plot_methods):
            for j, batch_size in enumerate(batch_sizes):
                label = f"{method}"
                axes[k].errorbar(x, sorted_avgs[i, j], sorted_errs[i, j], label=label, color=J_colors[i], fmt=f'-{markers[j]}', lw=lw, markersize=ms, markeredgewidth=mew, markeredgecolor='k', alpha=alpha_sym, capsize=0, capthick=0.0, elinewidth=1.8*lw, zorder=1)

        axes[k].set_xlabel('Episode $\displaystyle E$')
        axes[k].set_xlim(x[0],x[-1])

    axes[0].set_ylabel(r'Regret$\displaystyle_{E}\bigl(\theta^{(0)}\bigr)$') #$\displaystyle-\hat{J}(\theta)$')
    axes[-1].set_yscale('log')
    axes[-1].set_ylim(1e-3, 1e1)
    axes[-1].legend(ncols=1)

    if fn != None:
        fig.savefig(fn)

def plot_LQ_2xtrajs(data1, data2, T, batches, seed, fn):
    """plot trajectories for LQ case study with data1 and data2 coresponding to REINFORCE and BO"""

    fig, axes = plt.subplots(5, 2, figsize=(7, 3.33), facecolor='white', sharex=True, layout='constrained', height_ratios=[1,1,1,1,2], dpi=1200)
    var_names = [r"$\displaystyle(x_t)_0$", r"$\displaystyle(x_t)_1$", r"$\displaystyle(x_t)_2$", r"$\displaystyle(x_t)_3$"] # [r'$x_{0,t}$', r'$x_{1,t}$', r'$x_{2,t}$', r'$x_{3,t}$']
    t = np.arange(T+1)
    
    for k, data in enumerate([data1, data2]):
        xs = data["xs"]
        us = data["us"]
        for i, batch in enumerate(batches):
            label = f"{batch}" if k == 0 else ""
            for j in range(len(var_names)):
                axes[j,k].plot(t, xs[i, :, j], '-', color=LQ_traj_colors[i], lw=lw)
                axes[j,k].set_xlim(t[0], t[-1])
                axes[j,k].set_ylim(-2, 2)
                axes[j,k].set_yticks([-2,0,2]) if k == 0 else axes[j,k].set_yticks([-2,0,2], labels=3*[""])
                axes[j,0].set_ylabel(var_names[j])
            axes[-1,k].plot(t[:-1], us[i, :], '-', color=LQ_traj_colors[i], label=label)
        axes[-1,k].set_xlim(t[0], t[-1])
        axes[-1,k].set_ylim(-4, 4)
        axes[-1,k].set_yticks([-4,-2,0,2,4]) if k == 0 else axes[-1,k].set_yticks([-4,-2,0,2,4], labels=5*[""])
        axes[-1,k].set_xlabel(r'$\displaystyle t$')
        axes[-1,0].set_ylabel(r'$\displaystyle u_{t}$')

    norm = mpl.colors.Normalize(vmin=batches[0], vmax=batches[-1], clip=False)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    clb = fig.colorbar(sm, ax=axes[0,:], fraction=1.5, location='top', ticks=batches, aspect=100)
    clb.ax.xaxis.set_ticks_position('top')
    clb.ax.set_title('Episode $\displaystyle E$')

    if fn != None:
        fig.savefig(fn)
    plt.show()

def plot_norms(batches, mean_frob_A, mean_frob_B, err_frob_A, err_frob_B, plot_methods, batch_sizes, fn):
    """plot model mismatch represented by frobenius-norm for LQ case study"""
    fig, axes = plt.subplots(2, 1, figsize=(3.33,3.33), sharex=True, layout='constrained', height_ratios=[2, 1], dpi=1200) #, sharey=True)
    y_labels = [r'$\displaystyle||\hat{A}-A||_{F}$', r'$\displaystyle||\hat{B}-B||_{F}$']

    x = batches
    y = np.array([mean_frob_A, mean_frob_B])
    err = np.array([err_frob_A, err_frob_B])

    for i in range(y.shape[0]):
        for j, method in enumerate(plot_methods):
            for k, batch_size in enumerate(batch_sizes):
                label = f"{method} "
                axes[i].errorbar(x, y[i, j, k], err[i, j, k], label=label, color=J_colors[j], fmt=f'-{markers[k]}', lw=lw, markersize=ms, markeredgewidth=mew, markeredgecolor='k', alpha=alpha_sym, capsize=0, capthick=0.0, elinewidth=1.8*lw, zorder=1)

        axes[i].set_ylabel(y_labels[i])
        axes[i].set_yticks(np.arange(0, 2+np.floor(np.max(y[i]+err[i]))))
        axes[i].set_ylim(0, 1+np.floor(np.max(y[i]+err[i]))) 

    axes[-1].set_xlabel('Episode $\displaystyle E$')
    axes[-1].set_xlim(x[0],x[-1])
    axes[0].legend(ncols=1, loc="lower right")

    if fn != None:
        fig.savefig(fn)

def plot_QT_trajs(xs, us, T, batches, seed, fn):
    """plot trajectories for quadruple-tank case study with data1 and data2 coresponding to REINFORCE and BO"""
    fig, axes = plt.subplots(6, 1, figsize=(3.33, 5), facecolor='white', sharex=True, layout='constrained', height_ratios=[1,1,1,1,2,2], dpi=1200)
    var_names = [r"$\displaystyle y_1(t)$", r"$\displaystyle y_2(t)$", r"$\displaystyle y_3(t)$", r"$\displaystyle y_4(t)$", r"$\displaystyle q_{a}(t)$", r"$\displaystyle q_{b}(t)$"] # [r'$x_{0,t}$', r'$x_{1,t}$', r'$x_{2,t}$', r'$x_{3,t}$']
    t = 5*np.arange(T+1)
    t_r = np.arange(5*(T)+2)

    for i, batch in enumerate(batches):
        for j in range(4):
            axes[j].plot(t, xs[i, :, j], '-', color=QT_traj_colors[i], lw=lw)
            axes[j].set_xlim(t[0], t[-1])
            axes[j].set_ylim(-0.1, 2.)
            axes[j].set_yticks(np.linspace(0,2,3))
            axes[j].set_ylabel(var_names[j])
        for j in range(2):
            axes[j+4].plot(t[:-1], us[i, :, j], '-', color=QT_traj_colors[i], lw=lw)
            axes[j+4].set_xlim(t[0], t[-1])
            axes[j+4].set_ylim(-0.1, 4.1)
            axes[j+4].set_yticks(np.linspace(0,4,5))
            axes[j+4].set_ylabel(var_names[j+4])
    axes[-1].set_xlabel(r'$\displaystyle t$ (s)')

    period = 750
    h_1r = np.concatenate([ 0.65*np.ones(period+1), 0.30*np.ones(period), 0.50*np.ones(period), 0.90*np.ones(period+1) ])
    h_2r = np.concatenate([ 0.65*np.ones(period+1), 0.30*np.ones(period), 0.75*np.ones(period), 0.75*np.ones(period+1) ])
    h_3r = np.concatenate([ 0.652*np.ones(period+1), 0.301*np.ones(period), 0.305*np.ones(period), 1.062*np.ones(period+1) ])
    h_4r = np.concatenate([ 0.664*np.ones(period+1), 0.305*np.ones(period), 1.200*np.ones(period), 0.579*np.ones(period+1) ])
    h_r = np.vstack([h_1r, h_2r, h_3r, h_4r])
    ylims = [1.36, 1.36, 1.30, 1.30]
    ulims = [3.26, 4.]

    for j in range(4):
        label = "Reference" if j==0 else None
        axes[j].plot(t_r[:-1], h_r[j][:-1], 'b--', lw=0.5*lw, label=label)
        label = "Constraint" if j==0 else None
        axes[j].hlines([0,ylims[j]], t[0], t[-2], colors='r', linestyles='dashed', lw=0.5*lw, label=label)
        axes[j].hlines([0,ylims[j]], t[0], t[-2], colors='r', linestyles='dashed', lw=0.5*lw, label=label)
    for j in range(2):
        axes[j+4].hlines([0,ulims[j]], t[0], t[-2], colors='r', linestyles='dashed', lw=0.5*lw)

    norm = mpl.colors.Normalize(vmin=batches[0], vmax=batches[-1], clip=False)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    clb = fig.colorbar(sm, ax=axes[0], fraction=1.5, location='top', ticks=batches, aspect=100)
    clb.ax.xaxis.set_ticks_position('top')
    clb.ax.set_title('Episode $\displaystyle E$')

    if fn != None:
        fig.savefig(fn)
    plt.show()

def plot_2xdouble_bar(methods1, data1, xlabel1, methods2, data2, xlabel2, w=0.2, logy=True, fn=None, lloc="best"):
    """plot computation time for evaluating and differentiating policy for LQ case-study with data1 and data2 corresponding to n- and m-timings"""
    fig, axes = plt.subplots(1,2, layout='constrained', figsize=(7,3.33), sharey=True, width_ratios=[5,7], dpi=1200)
    plt_tuples = zip([methods1, methods2], [data1, data2], [xlabel1, xlabel2])
    multiplier = 0.5
    
    for j, (methods, data, xlabel) in enumerate(plt_tuples):
        x = np.arange(len(methods))
        for i, data_i in enumerate(data.items()):
            attribute, measurement = data_i
            offset = w * multiplier
            mean, err = mean_confidence_interval(measurement)
            rects = axes[j].bar(x + offset, mean,
                        w, edgecolor='k', color=t_colors[i], alpha=1, label=attribute)
            axes[j].errorbar(x + offset, y=mean, yerr=err, fmt='none', lw=lw, markersize=ms, markeredgewidth=lw, markeredgecolor='k', capsize=3., capthick=lw, elinewidth=lw, zorder=1)
            multiplier += 1

        axes[j].set_xlabel(f"$\displaystyle {{{xlabel}}}$") # axes[j].set_xlabel(f"${xlabel}$")
        axes[j].set_xticks(x + (1+2*j)*w, methods)
        if logy:
            axes[j].set_yscale("log")
            axes[j].set_ylim(1e-3,1e-2)
            tick_labels = [1e-3] + 9*[""] + [1e-2]
            axes[i].set_yticks(np.linspace(1e-3,1e-2,11), labels=tick_labels, minor=True)

    axes[0].set_ylabel("Computation Time (s)")
    axes[-1].legend(loc=lloc, ncols=1)

    if fn != None:
        plt.savefig(fn, format="png")

    plt.show()

def plot_double_bar(methods, data, xlabel, w=0.2, logy=True, fn=None, lloc="best"):
    """plot computation time for evaluating and differentiating policy for quadruple-tank case-study"""
    fig, ax = plt.subplots(layout='constrained', figsize=(3.33,3.33), dpi=1200)
    multiplier = 0.5
    # w = 1. / len(methods)    # bar width
    x = np.arange(len(methods))

    for i, data_i in enumerate(data.items()):
        attribute, measurement = data_i
        offset = w * multiplier
        mean, err = mean_confidence_interval(measurement)
        rects = ax.bar(x + offset, mean, w, edgecolor='k', color=t_colors[i], alpha=1, label=attribute)
        ax.errorbar(x + offset, y=mean, yerr=err, fmt='none', lw=lw, markersize=ms, markeredgewidth=lw, markeredgecolor='k', capsize=3., capthick=lw, elinewidth=lw, zorder=1)
        multiplier += 1

    ax.set_xlabel(f"$\displaystyle {{{xlabel}}}$")
    ax.set_ylabel("Computation Time (s)")
    ax.set_xticks(x + w, methods)
    ax.legend(loc=lloc, ncols=1)
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 3e-2)
        tick_labels = [1e-3] + 9*[""] + [1e-2] + 2*[""]
        ax.set_yticks(np.concatenate([np.linspace(1e-3,1e-2,11), [2e-2], [3e-2]]), labels=tick_labels, minor=True)

        # ax.set_ylim(1e-3, 1e-2) # 3e-2)
        # tick_labels = [1e-3] + 9*[""] + [1e-2] # + 2*[""]
        # ax.set_yticks(np.concatenate([np.linspace(1e-3,1e-2,11), [2e-2], [3e-2]]), labels=tick_labels, minor=True)

    if fn != None:
        plt.savefig(fn, format="png")

    plt.show()

############################################################################################################################################################

def construct_pi(cfg):
    x = cp.Parameter(cfg.n)
    A = cp.Parameter((cfg.n,cfg.n))
    B = cp.Parameter((cfg.n,cfg.m)) 
    Q = 1e-2 * np.eye(cfg.n,cfg.n)
    if cfg.m == 1:
        R = 1e-2
    else:
        R = 1e-2 * np.eye(cfg.m,cfg.m)

    # initial states, controls, constraints, and objective
    states = [cp.Variable(cfg.n) for _ in range(cfg.horizon)]
    controls = [cp.Variable(cfg.m) for _ in range(cfg.horizon)]
    if cfg.u_lim != None:
        constraints = [states[0] == x, cp.norm(controls[0], 'inf') <= cfg.u_lim]
    else:
        constraints = [states[0] == x]
    if cfg.m == 1:
        objective = cp.quad_form(states[0],Q) + cp.multiply(cp.square(controls[0]),R)
    else:
        objective = cp.quad_form(states[0],Q) + cp.quad_form(controls[0],R)

    # predicted states, controls, constraints, and objective across horizon
    for t in range(1, cfg.horizon):
        if cfg.m == 1:
            #1/01; objective += cp.quad_form(states[t],Q) + cp.multiply(cp.square(controls[0]),R) # stage cost w/ state and action penalties
            objective += cp.quad_form(states[t],Q) + cp.multiply(cp.square(controls[t]),R)
        else:
            objective += cp.quad_form(states[t],Q) + cp.quad_form(controls[t],R)
        constraints += [states[t] == A @ states[t-1] + B @ controls[t-1]]
        if cfg.u_lim != None:
            constraints += [cp.norm(controls[t], 'inf') <= cfg.u_lim]
    problem = cp.Problem(cp.Minimize(objective), constraints)

    return CvxpyLayer(problem, variables=[controls[0]], parameters=[x, A, B])

def construct_con_mpc(cfg, model):
    """Create constrained optimization problem to be solved at each time instance"""
    period = 750
    h_1r = np.concatenate([ 0.65*np.ones(period+1),  0.30*np.ones(period),  0.50*np.ones(period),  0.90*np.ones(period+cfg.horizon) ])
    h_2r = np.concatenate([ 0.65*np.ones(period+1),  0.30*np.ones(period),  0.75*np.ones(period),  0.75*np.ones(period+cfg.horizon) ])
    h_3r = np.concatenate([ 0.652*np.ones(period+1), 0.301*np.ones(period), 0.305*np.ones(period), 1.062*np.ones(period+cfg.horizon) ])
    h_4r = np.concatenate([ 0.664*np.ones(period+1), 0.305*np.ones(period), 1.200*np.ones(period), 0.579*np.ones(period+cfg.horizon) ])
    Q = np.eye(4,4)

    mpc = do_mpc.controller.MPC(model)
    setup_mpc = {
        'n_horizon': cfg.horizon,
        'n_robust': 0,
        'open_loop': False,
        't_step': 5.0,
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
        for k in range(cfg.horizon+1):
                tvp_template['_tvp',k,'h_1r'] = h_1r[int(t_now)+k]
                tvp_template['_tvp',k,'h_2r'] = h_2r[int(t_now)+k]
                tvp_template['_tvp',k,'h_3r'] = h_3r[int(t_now)+k]
                tvp_template['_tvp',k,'h_4r'] = h_4r[int(t_now)+k]
        return tvp_template
    mpc.set_tvp_fun(tvp_fun)

    # objective
    mterm = ca.DM(np.zeros((1,1)))
    lterm = ca.transpose(model.aux['x_e']) @ Q @ model.aux['x_e']
    mpc.set_objective(mterm=mterm, lterm=lterm)
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
        gamma_a = np.array([0.3]),
        gamma_b = np.array([0.4]),
        a_1 = np.array([1.31]),
        a_2 = np.array([1.51]),
        a_3 = np.array([0.927]),
        a_4 = np.array([0.882]),
    )

    mpc.setup()

    return mpc

def construct_uncon_mpc(cfg, model):
    """Create unconstrained optimization problem to be solved at each time instance"""
    mpc = do_mpc.controller.MPC(model)
    period = 750
    h_1r = np.concatenate([ 0.65*np.ones(period+1),  0.30*np.ones(period),  0.50*np.ones(period),  0.90*np.ones(period+cfg.horizon) ])
    h_2r = np.concatenate([ 0.65*np.ones(period+1),  0.30*np.ones(period),  0.75*np.ones(period),  0.75*np.ones(period+cfg.horizon) ])
    h_3r = np.concatenate([ 0.652*np.ones(period+1), 0.301*np.ones(period), 0.305*np.ones(period), 1.062*np.ones(period+cfg.horizon) ])
    h_4r = np.concatenate([ 0.664*np.ones(period+1), 0.305*np.ones(period), 1.200*np.ones(period), 0.579*np.ones(period+cfg.horizon) ])
    Q = np.eye(4,4)

    setup_mpc = {
        'n_horizon': cfg.horizon,
        'n_robust': 0,
        'open_loop': False,
        't_step': 5.0,
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
        for k in range(cfg.horizon+1):
                tvp_template['_tvp',k,'h_1r'] = h_1r[int(t_now)+k]
                tvp_template['_tvp',k,'h_2r'] = h_2r[int(t_now)+k]
                tvp_template['_tvp',k,'h_3r'] = h_3r[int(t_now)+k]
                tvp_template['_tvp',k,'h_4r'] = h_4r[int(t_now)+k]
        return tvp_template
    mpc.set_tvp_fun(tvp_fun)

    # objective
    mterm = ca.DM(np.zeros((1,1)))
    lterm = ca.transpose(model.aux['x_e']) @ Q @ model.aux['x_e']
    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(q_a=0.1)
    mpc.set_rterm(q_b=0.1)

    # learnable parameters
    mpc.set_uncertainty_values(
        gamma_a = np.array([0.3]),
        gamma_b = np.array([0.4]),
        a_1 = np.array([1.31]),
        a_2 = np.array([1.51]),
        a_3 = np.array([0.927]),
        a_4 = np.array([0.882]),
    )

    mpc.setup()

    return mpc