import torch
import torch.nn as nn
import wandb

# for 84x84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}


def copy_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias
    #print("Are target and source pointing to the same reference? ")
    #print(id(trg.weight)==id(src.weight))


class Encoder(nn.Module):
    def __init__(
        self,
        obs_shape,
        feature_dim,
        num_layers,
        num_filters,
        output_logits=False,
    ):
        super().__init__()
        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.num_filters = num_filters

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], self.num_filters, 3, stride=2)]
        )

        for _ in range(self.num_layers - 1):
            self.convs.append(
                nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1)
            )

        out_dim = (
            OUT_DIM_64[self.num_layers]
            if obs_shape[-1] == 64
            else OUT_DIM[self.num_layers]
        )
        self.fc = nn.Linear(self.num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = {}
        self.output_logits = output_logits

    def reparameterize(self, mu, logstd):
        """Reparameterize takes in the input mu and logvar and sample the mu + std * eps

        Args:
            mu: mean from the encoder's latent space
            logstd: log std from the encoder's latent space
        """
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.0
        self.outputs["obs"] = obs
        conv = torch.relu(self.convs[0](obs))
        self.outputs["conv1"] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs[f"conv{i + 1}"] = conv

        return conv.view(conv.size(0), -1)

    def forward(self, obs, detach=False):
        #print("inside encoder forward", obs.shape, type(obs))
        flatten = self.forward_conv(obs)
        #print('inside encoder forward', flatten.shape)

        if detach:
            flatten = flatten.detach()

        out_fc = self.fc(flatten)
        self.outputs["fc"] = out_fc

        out_norm = self.ln(out_fc)
        self.outputs["ln"] = out_norm

        if self.output_logits:
            out = out_norm
        else:
            out = torch.tanh(out_norm)
            self.outputs["tanh"] = out
        #print("inside encoder shape of out", out.shape, type(out))
        return out

    def copy_conv_weights(self, source):
        for i in range(self.num_layers):
            copy_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, step, log_freq, WB_LOG):
        if step % log_freq != 0:
            return

        for i in range(self.num_layers):
            if WB_LOG:
                wandb.log({f"train_encoder/conv{i + 1}": self.convs[i], "step": step})

        if WB_LOG:
            wandb.log({"train_encoder/fc": self.fc, "step": step})
            wandb.log({"train_encoder/ln": self.ln, "step": step})


class Decoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.out_dim = OUT_DIM[num_layers]
        self.fc = nn.Linear(feature_dim, num_filters * self.out_dim * self.out_dim)

        self.deconvs = nn.ModuleList()

        for _ in range(self.num_layers - 1):
            self.deconvs.append(
                nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1)
            )
        self.deconvs.append(
            nn.ConvTranspose2d(num_filters, obs_shape[0], 3, stride=2, output_padding=1)
        )

        self.outputs = {}

    def forward(self, h):
        h = torch.relu(self.fc(h))
        self.outputs["fc"] = h

        deconv = h.view(-1, self.num_filters, self.out_dim, self.out_dim)
        self.outputs["deconv1"] = deconv

        for i in range(self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))
            self.outputs[f"deconv{i + 1}"] = deconv

        obs = self.deconvs[-1](deconv)
        self.outputs["obs"] = obs

        return obs

    def log(self, step, log_freq, WB_LOG):
        if step % log_freq != 0:
            return

        for i in range(self.num_layers):
            if WB_LOG:
                wandb.log(
                    {f"train_decoder/deconv{i + 1}": self.deconvs[i], "step": step}
                )

        if WB_LOG:
            wandb.log({"train_decoder/fc": self.fc, "step": step})


AVAILABLE_ENCODERS = {"pixel": Encoder}
AVAILABLE_DECODERS = {"pixel": Decoder}


def make_encoder(
    encoder_type,
    obs_shape,
    feature_dim,
    num_layers,
    num_filters,
    output_logits=False,
):
    assert encoder_type in AVAILABLE_ENCODERS
    return AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, output_logits
    )


def make_decoder(decoder_type, obs_shape, feature_dim, num_layers, num_filters):
    assert decoder_type in AVAILABLE_DECODERS
    return AVAILABLE_DECODERS[decoder_type](
        obs_shape, feature_dim, num_layers, num_filters
    )

