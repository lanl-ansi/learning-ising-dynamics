# Learning Graphical Models from Dynamics

``GML_Glauber_Dynamics`` is a julia package for learning graphical models from time correlated samples generated through Gibbs sampling (aka Glauber dynamics). It is built on top of the julia package ``GraphicalModelLearning.jl`` ([see package here](https://github.com/lanl-ansi/GraphicalModelLearning.jl)).

## Installation

Install with Pkg, just like any other registered Julia package (note the difference in name from repo):

```jl
pkg> add GML_Glauber_Dynamics  # Press ']' to enter the Pkg REPL mode.
```

## Getting started

Let's start with a simple example where we generate samples through Glauber dynamics from an Ising model defined on a three node graph. The goal is to then check if the learned graph is close to the true graph from which the samples were generated.

```
using GML_Glauber_Dynamics

model = FactorGraph([0.0 0.9 0.1; 0.9 0.0 0.1; 0.1 0.1 0.0])
n_samples = 100000
samples_T, samples_mixed_T = gibbs_sampling(model, n_samples, T_regime())
learned_gm = learn_glauber_dynamics(samples_T)

err = abs.(convert(Array{Float64,2}, model) - learned_gm)
```

## Reference
If you use this package, please cite this `paper <https://arxiv.org/abs/2104.00995>` (accepted at ICML 2021):
::

	@article{dutt2021exponential,
	  title={Exponential Reduction in Sample Complexity with Learning of Ising Model Dynamics},
	  author={Dutt, Arkopal and Lokhov, Andrey Y and Vuffray, Marc and Misra, Sidhant},
	  journal={arXiv preprint arXiv:2104.00995},
	  year={2021}
	}


## License

This code is provded under a BSD license.
