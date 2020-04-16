import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import joblib
import tensorflow as tf

import gym
from gym.spaces import Discrete, Box

def save_path(samples, filename):
    joblib.dump(samples, filename, compress=3)


def restore_latest_n_traj(dirname, n_path=10, max_steps=None):
    assert os.path.isdir(dirname)
    filenames = get_filenames(dirname, n_path=n_path)
    return load_trajectories(filenames, None)


def get_filenames(dirname, n_path=None):
    import re
    cra_reg = re.compile(
        r"step_(?P<step>[0-9]+)_epi_(?P<episodes>[0-9]+)_return_(-?)(?P<return_u>[0-9]+).(?P<return_l>[0-9]+).pkl"
    )
    cra_files = []
    for _, filename in enumerate(os.listdir(dirname)):
        result = cra_reg.match(filename)
        if result:
            cra_count = result.group('step')
            cra_files.append((cra_count, filename))

    n_path = n_path if n_path is not None else len(cra_files)
    cra_files = sorted(cra_files, key=lambda x:int(x[0]), reverse=True)[:n_path]
    filenames = []
    for cra_file in cra_files:
        filenames.append(os.path.join(dirname, cra_file[1]))

    return filenames


def load_trajectories(filenames, max_steps=None):
    assert len(filenames) > 0
    tra_params = []
    for filename in filenames:
        tra_params.append(joblib.load(filename))

    def get_obs_and_act(params):
        obses = params['obs'][:-1]
        next_obses = params['obs'][1:]
        actions = params['act'][:-1]
        if max_steps is not None:
            # TODO: actions[:max_steps-1] ?
            return obses[:max_steps], next_obses[:max_steps], actions[:max_steps]
        else:
            return obses, next_obses, actions

    for index, params in enumerate(tra_params):
        if index == 0:
            obses, next_obses, acts = get_obs_and_act(params)
        else:
            obs, next_obs, act = get_obs_and_act(params)
            obses = np.vstack((obs, obses))
            next_obses = np.vstack((next_obs, next_obses))
            acts = np.vstack((act, acts))

    return {'obses': obses, 'next_obses':next_obses, 'acts':acts}


def frames_to_gif(frames, prefix, save_dir, interval=50, fps=30):
    assert len(frames) > 0
    plt.figure(figsize=(frames[0].shape[1] / 72.,
                        frames[0].shape[0] / 72.), dpi=72)
    ax = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        ax.set_data(frames[i])

    # TODO: interval should be 1000 / fps ?
    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames=len(frames), interval=interval
    )
    outpath = "{}/{}.gif".format(save_dir, prefix)
    anim.save(outpath, writer='imagemagick', fps=fps)


def is_discrete(space):
    if isinstance(space, Discrete):
        return True
    elif isinstance(space, Box):
        return False
    else:
        raise NotImplementedError


def get_act_dim(action_space):
    if isinstance(action_space, Discrete):
        return action_space.n
    elif isinstance(action_space, Box):
        return action_space.low.size
    else:
        raise NotImplementedError