import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from feature_extractor import make_encoder, make_decoder
import wandb
import augmentations
from rich.console import Console


LOG_FREQ = 10000
console = Console()


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        hidden_dim,
        encoder_type,
        encoder_feature_dim,
        log_std_min,
        log_std_max,
        num_layers,
        num_filters,
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type,
            obs_shape,
            encoder_feature_dim,
            num_layers,
            num_filters,
            output_logits=True,
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.layers = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0]),
        )

        self.outputs = {}
        self.apply(weight_init)

    def forward(self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)
        #print("inside actor forward", obs.shape)
        mu, log_std = self.layers(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )

        self.outputs["mu"] = mu
        self.outputs["std"] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None

        log_pi = gaussian_logprob(noise, log_std) if compute_log_pi else None
        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std, obs


class QFunction(nn.Module):
    """MLP for q-function."""

    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        obs_action = torch.cat([obs, action], dim=1)
        return self.layers(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        hidden_dim,
        encoder_type,
        encoder_feature_dim,
        num_layers,
        num_filters,
        WB_LOG,
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type,
            obs_shape,
            encoder_feature_dim,
            num_layers,
            num_filters,
            # WB_LOG,
            output_logits=True,
        )

        self.Q1 = QFunction(self.encoder.feature_dim, action_shape[0], hidden_dim)
        self.Q2 = QFunction(self.encoder.feature_dim, action_shape[0], hidden_dim)

        self.outputs = {}
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs["q1"] = q1
        self.outputs["q2"] = q2

        return q1, q2

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.apply(weight_init)

    def forward(self, x):
        return self.mlp(x)

class Predictor(nn.Module):
    def __init__(self, encoder, hidden_dim):
        super().__init__()
        self.encoder = encoder
        self.mlp = MLP(encoder.feature_dim, hidden_dim, encoder.feature_dim)
        self.apply(weight_init)
    
    def forward(self, x):
        return self.mlp(self.encoder(x))

class CURL(nn.Module):
    """
    CURL
    """

    def __init__(
        self,
        z_dim,
        batch_size,
        critic,
        critic_target,
        output_type="continuous",
    ):
        super(CURL, self).__init__()
        self.batch_size = batch_size
        self.encoder = critic.encoder
        self.encoder_target = critic_target.encoder
        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.output_type = output_type

    def encode(self, x, detach=False, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)

        if detach:
            z_out = z_out.detach()
        return z_out

    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits


class CurlSacAgent(object):
    """CURL representation learning with SAC."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        WB_LOG,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type="pixel",
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        decoder_type="pixel",
        decoder_lr=1e-3,
        decoder_update_freq=1,
        decoder_latent_lambda=1e-6,
        decoder_weight_lambda=1e-7,
        num_layers=4,
        num_filters=32,
        curl_encoder_update_freq=1,
        log_interval=100,
        detach_encoder=False,
        curl_latent_dim=128,
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.curl_encoder_update_freq = curl_encoder_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.curl_latent_dim = curl_latent_dim
        self.detach_encoder = detach_encoder
        self.encoder_type = encoder_type
        self.decoder_update_freq = decoder_update_freq
        self.decoder_latent_lambda = decoder_latent_lambda

        self.actor = Actor(
            obs_shape,
            action_shape,
            hidden_dim,
            encoder_type,
            encoder_feature_dim,
            actor_log_std_min,
            actor_log_std_max,
            num_layers,
            num_filters,
            #WB_LOG,
        ).to(device)

        self.critic = Critic(
            obs_shape,
            action_shape,
            hidden_dim,
            encoder_type,
            encoder_feature_dim,
            num_layers,
            num_filters,
            WB_LOG,
        ).to(device)

        self.critic_target = Critic(
            obs_shape,
            action_shape,
            hidden_dim,
            encoder_type,
            encoder_feature_dim,
            num_layers,
            num_filters,
            WB_LOG,
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic, and CURL and critic
        self.actor.encoder.copy_conv_weights(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True

        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)
        self.decoder = None
        if decoder_type != "identity":
            # create decoder
            self.decoder = make_decoder(
                decoder_type, obs_shape, encoder_feature_dim, num_layers, num_filters
            ).to(device)
            self.decoder.apply(weight_init)
            # optimizer for decoder
            self.decoder_optimizer = torch.optim.Adam(
                self.decoder.parameters(),
                lr=decoder_lr,
                weight_decay=decoder_weight_lambda,
            )
        
        # predictor (Encoder + MLP) for consistency loss
        self.predictor = Predictor(self.critic.encoder,hidden_dim).to(device)



        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        if self.encoder_type == "pixel":
            # create CURL encoder (the 128 batch size is probably unnecessary)
            self.CURL = CURL(
                encoder_feature_dim,
                self.curl_latent_dim,
                self.critic,
                self.critic_target,
                output_type="continuous",
            ).to(self.device)

            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=encoder_lr
            )

            self.curl_encoder_optimizer = torch.optim.Adam(
                self.CURL.parameters(), lr=encoder_lr
            )

            self.pred_optimizer = torch.optim.Adam(self.predictor.parameters(), lr=encoder_lr)

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.train()
        self.critic_target.train()

    """function to help toggle between train and eval mode"""

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.encoder_type == "pixel":
            self.CURL.train(training)
        if self.decoder is not None:
            self.decoder.train(training)
        if self.predictor is not None:
            self.predictor.train(training)
        
        

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = augmentations.center_crop_image(obs, self.image_size)
        #print("inside sample action", obs.shape)
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            #print("inside sample action torch.no grad", obs.shape)
            _, pi, _, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, step, WB_LOG):
        with torch.no_grad():
            _, policy_action, log_pi, _, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(
            obs,
            action,
            detach_encoder=self.detach_encoder,  # set detach_encoder to True to stop critic's gradient flow to the encoder.
        )
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        if step % self.log_interval == 0 and WB_LOG:
            wandb.log({"train_critic/loss": critic_loss, "step": step})

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, step, WB_LOG):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std, _ = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0 and WB_LOG:
            wandb.log({"train_actor/loss": actor_loss, "step": step})
            wandb.log({"train_actor/target_entropy": self.target_entropy, "step": step})
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(
            dim=-1
        )
        if step % self.log_interval == 0 and WB_LOG:
            wandb.log({"train_actor/entropy": entropy.mean(), "step": step})

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0 and WB_LOG:
            wandb.log({"train_alpha/loss": alpha_loss, "step": step})
            wandb.log({"train_alpha/value": self.alpha, "step": step})

        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_curl_encoder_and_decoder(
        self, orig_obs, obs_anchor, obs_pos, target_obs, step, WB_LOG, c1, c2, c3
    ):
        z_a = self.CURL.encode(obs_anchor)
        z_pos = self.CURL.encode(obs_pos, ema=True)

        logits = self.CURL.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        curl_loss = self.cross_entropy_loss(logits, labels)

        h = self.critic.encoder(obs_anchor)
        if target_obs.dim() == 4:
            # preprocess images to be in [-0.5, 0.5] range
            target_obs = utils.preprocess_obs(target_obs)
        rec_obs = self.decoder(h)
        rec_loss = F.mse_loss(target_obs, rec_obs)

        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()
      
        # Consistency loss
        h0 = self.predictor(obs_anchor)
        with torch.no_grad():
            h1 = self.critic_target.encoder(orig_obs)
        h0 = F.normalize(h0, p=2, dim=1)
        h1 = F.normalize(h1, p=2, dim=1)

        consis_loss = F.mse_loss(h0, h1)

        loss = (
            c1 * curl_loss + c2 * rec_loss + self.decoder_latent_lambda * latent_loss + c3 * consis_loss
        )

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        self.curl_encoder_optimizer.zero_grad()
        self.pred_optimizer.zero_grad()

        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.curl_encoder_optimizer.step()
        self.pred_optimizer.step()

        if step % self.log_interval == 0 and WB_LOG:
            wandb.log({"train/curl_loss": curl_loss, "step": step})
            wandb.log({"train/rec_loss": rec_loss, "step": step})
            wandb.log({"train/consistency_loss": consis_loss, "step": step})
            wandb.log({"train/total_loss": loss, "step": step})
            

    def update(self, replay_buffer, step, WB_LOG, c1, c2, c3):
        if self.encoder_type == "pixel":
            (
                orig_obs,
                obs,
                action,
                reward,
                next_obs,
                not_done,
                info_dict,
            ) = replay_buffer.sample_img_obs()

        if step % self.log_interval == 0 and WB_LOG:
            wandb.log({"train/batch_reward": reward.mean(), "step": step})

        self.update_critic(obs, action, reward, next_obs, not_done, step, WB_LOG)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, step, WB_LOG)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder, self.encoder_tau
            )

        if step % self.curl_encoder_update_freq == 0 and self.encoder_type == "pixel":
            obs_anchor, obs_pos = info_dict["obs_anchor"], info_dict["obs_pos"]
            self.update_curl_encoder_and_decoder(
                orig_obs, obs_anchor, obs_pos, obs_anchor, step, WB_LOG, c1, c2, c3
            )

    def save(self, model_dir, step):
        torch.save(self.actor.state_dict(), f"{model_dir}/actor_{step}.pt")
        torch.save(self.critic.state_dict(), f"{model_dir}/critic_{step}.pt")
        if self.decoder is not None:
            torch.save(self.decoder.state_dict(), f"{model_dir}/decoder_{step}.pt")

    def save_curl(self, model_dir, step):
        torch.save(self.CURL.state_dict(), f"{model_dir}/curl_{step}.pt")

    def load(self, model_dir, step):
        self.actor.load_state_dict(torch.load(f"{model_dir}/actor_{step}.pt"))
        self.critic.load_state_dict(torch.load(f"{model_dir}/critic_{step}.pt"))
        if self.decoder is not None:
            self.decoder.load_state_dict(torch.load(f"{model_dir}/decoder_{step}.pt"))
