## Nonparametric CLI Testing
Simulation studies of conditional local independence testing based on debiased machine learning.
Reproduces the results reported in *Arxiv link (todo)*

## Setup
The code utilizes the BoXHED estimator, specifically the version:
> https://github.com/BoXHED/BoXHED2.0/tree/8233c4f803878d28c93951b2c0653b0fde26bf96

1. Clone this repository and clone the above BoXHED repository into the main folder.
2. Follow instructions on BoXHED setup
3. Activate boxhed environment and run: `python run_simulations.py`

Options for simulation settings are `--file_save_path --sample_sizes --repetitions --same_kernels --kernels --betas --alternatives --store_sample_paths`

4. The data can be plotted using `python plot_results.py` *(to do)*
