from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from train import dqn
import json
import numpy as np

N_EPISODES = 200

# define the search space for bayesian optimization
space = [Integer(32,64, name="batch_size"),
          Integer(2, 10, name="update_every"),
          Integer(32, 128, name="fc1_units"),
          Integer(32, 128, name="fc2_units"),
          Real(1e-5, 1e0, "log-uniform", name='eps_decay'),
          Real(1e-5, 1e0, "log-uniform", name='lr'),
          Real(1e-5, 1e0, "log-uniform", name="gamma"),
          Real(1e-5, 1e0, "log-uniform", name="tau")]


def find_optimal_hyperparameters(env, brain_name):
    """
    Given an environment and unity brain_name, conduct a search for optimal hyperparameters, returning the optimal parameters
    and the raw output from the optimization process:
    
    Parameters
    ----------
    env: UnityEnvironment, the banana environment.
    brain_name: string, the name to associate with the model checkpoint

    Returns
    -------
    (params: Dict, the list of parameters used for the optimal score, res_gp: the raw output from the optimizer)
    
    """
    
    @use_named_args(space)
    def objective(**params):
        """The objective function to minimuze with Gaussian Process Regression"""
        scores = dqn(env=env, brain_name=brain_name, n_episodes=N_EPISODES, break_early=False, **params)
        return -np.mean(scores[-100:])
    
    """Find and return optimal hyperparameter values for the agent."""
    res_gp = gp_minimize(objective, space, n_calls=50, random_state=0)
    
    # show the best score achieved with optimal hyperparameters
    print(f"Best score on {N_EPISODES} episodes: {-res_gp.fun}")

    # extract the parameters used for the best score, and (parameters we didn't change)
    params = {
        'break_early': True,
        'eps_start': 1.0,
        'eps_end': 0.01,
        'max_t': 1000, 
        'n_episodes': 1800,
        'batch_size': int(res_gp.x[0]),
        'update_every': int(res_gp.x[1]),
        'fc2_units': int(res_gp.x[2]),
        'fc1_units': int(res_gp.x[3]),
        'eps_decay': int(res_gp.x[4]),
        'lr': float(res_gp.x[5]),
        'tau': float(res_gp.x[7]),
        'gamma': float(res_gp.x[6])    
    }
    
    print(json.dumps(params, indent=2))

    return params, res_gp