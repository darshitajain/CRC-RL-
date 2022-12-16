import contextlib
import torch
import numpy as np
import gym
import os
from collections import deque
import random
from torch.utils.data import Dataset
import time
import torchvision
from torchvision.utils import save_image
import json
#import torchvision.transforms.functional as F
import torch.nn.functional as nnf
from augmentations import center_crop_image_batch



class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(dir_path):
    with contextlib.suppress(OSError):
        os.mkdir(dir_path)
    return dir_path

def load_config(key=None):
    path = os.path.join('../setup', 'config.cfg')
    with open(path) as f:
        data = json.load(f)
    return data[key] if key is not None else data


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2 ** bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2 ** (8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        capacity,
        batch_size,
        device,
        image_size=84,
        transform1=None,
        transform2=None
    ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.transform1 = transform1
        self.transform2 = transform2
        # pixels obs are stored as uint8
        obs_dtype = np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_img_obs(self):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        orig_obs = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        pos = obses.copy()

        obses = self.transform1(obses, self.image_size)
        obses = torch.as_tensor(obses, device=self.device).float()
        obses = self.transform2(obses) 

        #orig_obs = center_crop_image_batch(orig_obs, self.image_size)
        orig_obs = self.transform1(orig_obs, self.image_size)

        #print("after resize", orig_obs.shape, type(orig_obs))

        next_obses = self.transform1(next_obses, self.image_size)
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        next_obses = self.transform2(next_obses)

        pos = self.transform1(pos, self.image_size)
        pos = torch.as_tensor(pos, device=self.device).float()
        pos = self.transform2(pos)
        
        #print('inside replay buffer', obses.shape, next_obses.shape)

        # visualize augmented images
        # if counter() <= 24:
        #    visualise_aug_obs(obses, self.transform.__name__)

        #obses = torch.as_tensor(obses, device=self.device).float()
        orig_obs = torch.as_tensor(orig_obs, device=self.device).float()
        #next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        #pos = torch.as_tensor(pos, device=self.device).float()
        info_dict = dict(obs_anchor=obses, obs_pos=pos, time_anchor=None, time_pos=None)

        return orig_obs, obses, actions, rewards, next_obses, not_dones, info_dict

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, "%d_%d.pt" % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save : self.idx],
            self.next_obses[self.last_save : self.idx],
            self.actions[self.last_save : self.idx],
            self.rewards[self.last_save : self.idx],
            self.not_dones[self.last_save : self.idx],
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split("_")[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split(".")[0].split("_")]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

    def __getitem__(self, idx):
        idx = np.random.randint(0, self.capacity if self.full else self.idx, size=1)
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, action, reward, next_obs, not_done

    def __len__(self):
        return self.capacity


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype,
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


def count_parameters(net, as_int=False):
    """Returns total number of params in a network"""
    count = sum(p.numel() for p in net.parameters())
    if as_int:
        return count
    return f"{count:,}"


def visualise_aug_obs(obs, transform):
    # print("shape of obs inside visualise_aug_obs",obs.shape, obs[0].shape)=> (128, 9, 84, 84) (9, 84, 84)
    batch_tensor = obs[0].transpose(1, 2, 0)
    # print("shape after transpose", batch_tensor.shape), (84, 84, 9)
    batch_tensor = np.dsplit(batch_tensor, 3)
    # print('shape after dsplit', len(batch_tensor), batch_tensor[0].shape), 3 (84, 84, 3)
    out = [
        (batch_tensor[i].transpose(2, 0, 1)) / 255.0 for i in range(len(batch_tensor))
    ]
    # print('shape after transpose', len(out), out[0].shape), 3 (3, 84, 84)
    out = np.array(out)
    out = torch.from_numpy(out)
    grid_img = torchvision.utils.make_grid(out, nrow=3)
    # print("shape of grid", grid_img.shape), [3, 88, 260]
    save_image(grid_img, "aug_%s_%d.jpg" % (transform, counter()))


def counter():
    counter.counter = getattr(counter, "counter", 0) + 1
    return counter.counter


"""Config class"""


class Config:
    """Config class which contains data, train and model hyperparameters"""

    def __init__(
        self,
        env,
        replay_buffer,
        train,
        eval,
        critic,
        actor,
        encoder,
        decoder,
        sac,
        params,
    ):
        self.env = env
        self.replay_buffer = replay_buffer
        self.train = train
        self.eval = eval
        self.critic = critic
        self.actor = actor
        self.encoder = encoder
        self.decoder = decoder
        self.sac = sac
        self.params = params

    @classmethod
    def from_json(cls, cfg):
        """Creates config from json"""
        params = json.loads(json.dumps(cfg), object_hook=HelperObject)
        return cls(
            params.env,
            params.replay_buffer,
            params.train,
            params.eval,
            params.critic,
            params.actor,
            params.encoder,
            params.decoder,
            params.sac,
            params.params,
        )


class HelperObject(object):
    """Helper class to convert json into Python object"""

    def __init__(self, dict_):
        self.__dict__.update(dict_)
