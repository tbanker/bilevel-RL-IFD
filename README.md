# Bi-level Optimization with RL and IFD
**A Gradient-based Framework for Bi-level Optimization of Black-Box Functions: Synergizing Model-free Reinforcement Learning and Implicit Function Differentiation**

Please view our full paper in the Special Issue of *Industrial & Engineering Chemistry Research* on AI/ML in Chemical Engineering [[link coming soon]()]

## Abstract
Bi-level optimization problems are challenging to solve due to the complex interplay between upper-level and lower-level decision variables. Classical solution methods generally simplify the bi-level problem to a single level problem, whereas more recent methods such as evolutionary algorithms and Bayesian optimization take a black-box view that can suffer from scalability to larger problems. While advantageous for handling high-dimensional and non-convex optimization problems, the application of gradient-based solution methods to bi-level problems is impeded by the implicit relationship between the upper-level and lower-level decision variables. Additionally, lack of an equation-oriented relationship between decision variables and the upper-level objective can further impede differentiability. To this end, we present a gradient-based optimization framework that leverages implicit function theorem and model-free reinforcement learning (RL) to solve bi-level optimization problems wherein only zeroth-order observations of the upper-level objective are available. Implicit differentiation allows for differentiating the optimality conditions of the lower-level problem to enable calculation of gradients of the upper-level objective. Using policy gradient RL, gradient-based updates of the upper-level decisions can then be performed in a scalable manner for high-dimension problems. The proposed framework is applied to the bi-level problem of learning optimization-based control policies for uncertain systems. Simulation results on two benchmark problems illustrate the effectiveness of the framework for goal-oriented learning of model predictive control policies. Synergizing derivative-free optimization via model-free RL and gradient evaluation via implicit function differentiation can create new avenues for scalable and efficient solution of bi-level problems with black-box upper-level objective as compared to black-box optimization methods that discard the problem structure.

## Install
To run this code on your own device, you may create a virtual environment using the provided `environment.yaml` file, i.e., `conda env create --file=environment.yaml`.

## Case Studies
In the paper, we apply the framework learning optimization-based control policies for uncertain systems. Code to run the two case studies may be found at `src/train_LQ.py` and `src/train_QT.py`. The hyperparameters for these case studies are readily modified from their base configurations at `cfgs/LQ_base.yaml` and `cfgs/QT_base.yaml`. Upon completion, the corresponding results are readily viewed in `visualize_learning.ipynb` and `visualize_timings.ipynb`.

## Citation
If you find this repository or our work helpful, please cite as:
``` bibtex
\article{banker2025bilevel,
    author={Banker, Thomas and Mesbah, Ali},
    title={A Gradient-based Framework for Bi-level Optimization of Black-Box Functions: Synergizing Model-free Reinforcement Learning and Implicit Function Differentiation}
    journal={Industrial & Engineering Chemistry Research},
    year={In Press},
    volume = {},
    number = {},
    pages = {},
    note={Invited Contribution to the Special Issue on AI/ML in Chemical Engineering}
}
```

## Rights and Collaborators
(c) 2025 Mesbah Lab

in collaboration with Ali Mesbah.

Questions regarding this code may be directed to [thomas_banker (at) berkeley.edu](mailto:thomas_banker@berkeley.edu)