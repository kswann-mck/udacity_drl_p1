"""
This script is used to train a Deep Q Network model on the Unity Banana environment. The base model, agent
and training function were taken from the solution here:
https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn/solution.

It was modified to include parameters to make training with different hyperparamter options easier, and to work with
the Unity environment, which is slightly different than the OpenAI Gym API.
"""


from dqn_agent import Agent
import json
import torch
import numpy as np
from collections import deque


def dqn(
    env,
    brain_name,
    state_size=37,
    action_size=4,
    name="",
    break_early=False,
    solved_threshold=13.0,
    n_episodes=1800, 
    max_t=1000, 
    eps_start=1.0, 
    eps_end=0.01, 
    eps_decay=0.995,
    batch_size=64,
    update_every=4,
    fc1_units=64,
    fc2_units=64,
    lr=5e-4,
    gamma=0.99,
    tau=1e-3
    ):
    """
    This function uses the dqn_agent, and the neural network, to train an agent using a specific set of hyperparameters
    to solve the Unity Banana environment.
    
    Parameters
    ----------
    env: UnityEnvironment, the banana environment.
    name: string, the name to associate with the model checkpoint
    break_early: bool, if the operation should cease when the solved threshold is reached
    solved_threshold: float,  the point at which the task is considered solved
    n_episodes: int, the maximum number of episodes to train for
    max_t: int, the maximum number of timesteps per episode,
    eps_start: float, the starting value of epsilon
    eps_end: float, the minimum value of epsilon
    eps_decay: float, the rate at which epsilon decays with subsequent episodes
    batch_size: int, the batch sized used for gradient descent during the learning phase
    update_every: int, the interval of episodes at which the learning step occurs
    fc1_units: int, the number of neurons in the first fully connected layer of the neural network
    fc2_units: int, the number of neurons in the second fully connected layer of the neural network
    lr: float, the learning rate for gradient descent
    gamma: float, the reward discount factor used in updates
    tau: float, the interpolation parameter for the soft update

    Returns
    -------
    scores: List[float], the list of score values for each episode.
    """
    
    params = {key: val for key, val in locals().items() if key != "env"}
    # print the set of parameters used in this call
    print(json.dumps(params, indent=2, default=str), end="\r")
    
    # initialize agent
    agent = Agent(
        state_size=state_size, 
        action_size=action_size, 
        seed=0, 
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        lr=lr,
        update_every=update_every,
        fc1_units=fc1_units, 
        fc2_units=fc2_units
    )


    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        mean_score = np.mean(scores_window)
        print(f'\rEpisode {i_episode}\tAverage Score: {round(mean_score, 2)}', end="")
        if mean_score >= solved_threshold:
            print(f'\nEnvironment solved in {i_episode} episodes!\tAverage Score: {round(mean_score, 2)}')
            torch.save(agent.qnetwork_local.state_dict(), f'checkpoint{name}.pth')
            if break_early == True:
                break


    return scores