import random
from itertools import count

import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import gym
import ipdb
import numpy as np
from pyvirtualdisplay.smartdisplay import SmartDisplay

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim
from torch.distributions.categorical import Categorical
from torchvision import transforms as T
from torchvision import utils
from tqdm import tqdm

# from replay_memory import ReplayTrajectoryMemory, TrajectoryState
# from vqvae import VQVAE
from rrt_new import RRT

"""
MCTS imports
"""
import time
import numpy as np


# from trainer import Trainer
# from policy import HillClimbingPolicy
# from replay_memory import ReplayMemory
# from hill_climbing_env import HillClimbingEnv
# from mcts_mountain_car_env import MCTSMountainCarEnv

# from mcts import execute_episode

# from pyvirtualdisplay.smartdisplay import SmartDisplay


resize = T.Compose(
    [T.ToPILImage(), T.Resize((32, 32), interpolation=Image.CUBIC), T.ToTensor()]
)

normalize = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


def get_render(env):
    return env.render(mode="rgb_array")


def get_screen(env):
    render = get_render(env)
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = render.transpose((2, 0, 1))
    # _, screen_height, screen_width = screen.shape
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen)


def get_env_img(env):
    return get_screen(env).permute(1, 2, 0).cpu().numpy()


def show_example_screen(env, get_image_func, title, name):
    plt.figure()
    plt.imshow(
        get_image_func(env), interpolation="none",
    )
    plt.title(title)
    plt.savefig(f"{name}.png")


def get_normalized_img(screen):
    norm_img_tensor = normalize(screen)
    return norm_img_tensor.unsqueeze(0)


def select_action(n_actions):
    return random.randrange(n_actions)


# def train(env, vq_vae, vq_vae_optimizer, memory, device):

#     # Load vq_vae trained on single rollout of length 400
#     checkpoint_name = "checkpoint/vq_vae_mountaincar.pt"
#     vq_vae.load_state_dict(torch.load(checkpoint_name))

#     criterion = nn.MSELoss()
#     latent_loss_weight = 0.25
#     rollout_len = 4
#     env.reset()
#     for rollout_i in tqdm(range(rollout_len)):
#         action = select_action(n_actions=env.action_space.n)
#         _observation, reward, _done, _ = env.step(action)
#         img_tensor = get_screen(env)
#         memory.push(img_tensor, action, reward)
#         img = get_normalized_img(img_tensor)
#         img = img.to(device)

#         # Using a top encoding layer of 2x2 for easier mapping to discrete state
#         # n_embed ** 4 possible states
#         quant_t, quant_b, diff, id_t, id_b = vq_vae.encode(img)

#         # Train VQ VAE
#         vq_vae.zero_grad()
#         out, latent_loss = vq_vae(img)
#         recon_loss = criterion(out, img)
#         latent_loss = latent_loss.mean()
#         loss = recon_loss + latent_loss_weight * latent_loss
#         loss.backward()
#         vq_vae_optimizer.step()

# - idea: view hierarchical discrete encoding as the "state of the board"
# and use monte carlo tree search (for discrete actions)


# torch.save(vq_vae.state_dict(), checkpoint_name)

# utils.save_image(out, "test_normalize.png", normalize=True)


def set_state(env, state):
    _state = np.asarray(state, dtype=np.float32)
    env.state = _state
    return env


def gt_dynamics(env, state, action):
    env = set_state(env, state)
    result = env.step(action)
    return result[0]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    env = gym.make("MountainCar-v0").unwrapped
    env.reset()
    n_actions = env.action_space.n

    # action = select_action(n_actions)
    # print(action)
    # gt_dynamics(env, env.state, action)

    # Cost to move in state space should inform distance
    # Could predict this cost
    #
    start_state = env.state.tolist()
    goal_state = [0.5, 0] # hardcoded -> todo: explore the boundaries

    # How do we know the rand_area? (here assumed to be square)
    # print(env.high, env.low)
    rand_area = [-1.2, 0.6]  # hardcoded -> todo: improve the state space using learned or given dynamics

    rrt_args = {
        "start": start_state,
        "goal": goal_state,
        "rand_area": rand_area,
        "obstacle_list": [],
        "expand_dis": 0.01,
        "path_resolution": 0.005,
        "gt_dynamics": lambda state, action: gt_dynamics(env, state, action),
        "max_iter": 150,
    }

    with SmartDisplay(visible=False, size=(1400, 900)) as _display:
        rrt = RRT(**rrt_args)
        path = rrt.planning(animation=True)
        print(path)
        img = _display.waitgrab()
        img.save("display.png")
        # show_example_screen(env, get_render, "example screen", "example_screen")
        # show_example_screen(env, get_env_img, "example env", "example_env")

    # vq_vae = VQVAE(n_embed=8).to(device)
    # vq_vae_optimizer = optim.Adam(vq_vae.parameters(), lr=1e-3)
    # memory = ReplayTrajectoryMemory(10000)

    # train(env, vq_vae, vq_vae_optimizer, memory, device)

    print("Complete")


if __name__ == "__main__":
    main()
