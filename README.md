This repository contains a pytorch implemenation of Schulman et al.'s (2017) Proximal Policy Optimization. Note that this
implementation supports ATARI games only.

# Install
See `requirements.txt` for all required packages. Copy `config.example` to `config` and adjust parameters as desired.

# Usage
Use `python -m source.main` to start a training run. The configuration can be adjusted by modifying either the file
`config` or the class `HyperParameters` in `source/utilities/config/hyperparameters.py`.

# Tests
This implementation was tested extensively through unit and integration tests as well as manual tests. The results of
many experiments can be found in the repository [ppo-results](https://github.com/aethiles/ppo-results). In terms of
learning performance, this implementation matches or exceeds the results reported by Schulman et al. (2017).

# Lessons Learned
- Tensorboard logging is not well-suited for the learning process at hand.
- Unit tests and integration tests save a lot of time if done properly and extensively.
- Always challenge your assumptions.
