This repository contains both my master's thesis in computer science as well as a pytorch implemenation of Schulman et al.'s (2017) Proximal Policy Optimization (PPO) written for my thesis. Note that this implementation supports ATARI games only.

# Thesis
In my thesis I researched the effect of a few non-documented optimization choices in OpenAI's baselines implementation of PPO on a selection of ATARI 2600 games.
Overall these optimizations have a notable effect on the learning performance. Due to interplay of the optimizations with each other as well as dependence on specific hyperparameter choices, identifying the individual impact of each optimization is a non-trivial affair that merits further research.

# Install
See `requirements.txt` for all required packages. Copy `config.example` to `config` and adjust parameters as desired.

# Usage
Use `python -m source.main` to start a training run. The configuration can be adjusted by modifying either the file
`config` or the class `HyperParameters` in `source/utilities/config/hyperparameters.py`.

Use `python -m source.demo` to run a demo. Trained modes of BeamRider, Breakout and Pong are available. Consult 
`python -m source.demo -h` for a full list of options.

# Tests
This implementation was tested extensively through unit and integration tests as well as manual tests. The results of
many experiments can be found in the repository [ppo-results](https://github.com/aethiles/ppo-results). In terms of
learning performance, this implementation matches or exceeds the results reported by Schulman et al. (2017).

# Lessons Learned
- Sometimes *well-known* properties or algorithm design choices are the hardest to find sources on.
- Tensorboard logging is not well-suited for the learning process at hand.
- Unit tests and integration tests save a lot of time if done properly and extensively.
- Always challenge your assumptions.
