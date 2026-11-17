# From CausalAILab/Causally-Aligned-Curriculum-Learning @ f52fcea, deep_sprl/teachers/goal_gan/gan.py
# (MIT License, (c) 2026 Mingxuan Li). Only the import paths are changed.
import numpy as np
import torch
from functools import partial


def batch_feed_array(array, batch_size):
    data_size = array.shape[0]

    if data_size <= batch_size:
        while True:
            yield array
    else:
        start = 0
        while True:
            if start + batch_size < data_size:
                yield array[start:start + batch_size, ...]
            else:
                yield np.concatenate(
                    [array[start:data_size], array[0: start + batch_size - data_size]],
                    axis=0
                )
            start = (start + batch_size) % data_size


class FCGAN(object):
    """Fully connected least-squares GAN (LSGAN) used by GoalGAN."""

    def __init__(self, generator_output_size, discriminator_output_size, generator_layers, discriminator_layers,
                 noise_size, batch_size=64, device='cpu'):

        self.discriminator_output_size = discriminator_output_size
        self.noise_size = noise_size
        self.batch_size = batch_size
        self.device = device

        self.generator = Generator(generator_output_size, generator_layers, noise_size).to(self.device)
        self.discriminator = Discriminator(generator_output_size, discriminator_layers,
                                           discriminator_output_size).to(self.device)

        # This init function mimics the initialization of the original TensorFlow implementation
        def weights_init(weight_init_fn, bias_init_fn, m):
            if isinstance(m, torch.nn.Linear):
                weight_init_fn(m.weight)
                bias_init_fn(m.bias)

        self.generator.apply(partial(weights_init, torch.nn.init.xavier_uniform_, torch.nn.init.zeros_))
        self.discriminator.apply(partial(weights_init, torch.nn.init.xavier_uniform_, torch.nn.init.zeros_))

        self.generator_optimizer = torch.optim.RMSprop(self.generator.parameters(), lr=1e-3)
        self.discriminator_optimizer = torch.optim.RMSprop(self.discriminator.parameters(), lr=1e-3)

    def sample_random_noise(self, size):
        return np.random.randn(size, self.noise_size)

    def sample_generator(self, size):
        generator_samples = []
        generator_noise = []
        batch_size = self.batch_size
        for i in range(0, size, batch_size):
            sample_size = min(batch_size, size - i)
            noise = self.sample_random_noise(sample_size)
            generator_noise.append(noise)
            generator_samples.append(
                self.generator.forward(torch.from_numpy(noise).float()).detach().numpy()
            )
        return np.vstack(generator_samples), np.vstack(generator_noise)

    def train(self, X, Y, outer_iters):
        batch_size = self.batch_size
        generated_Y = np.zeros((batch_size, self.discriminator_output_size))
        batch_feed_X = batch_feed_array(X, batch_size)
        batch_feed_Y = batch_feed_array(Y, batch_size)

        for _ in range(outer_iters):
            # One discriminator step on a batch of labeled goals plus a batch of generated (negative) goals
            sample_X = next(batch_feed_X)
            sample_Y = next(batch_feed_Y)
            generated_X, random_noise = self.sample_generator(batch_size)

            train_X = np.vstack([sample_X, generated_X])
            train_Y = np.vstack([sample_Y, generated_Y])
            dis_log_loss = self.train_discriminator(train_X, train_Y)

            # One generator step using the same noise
            gen_log_loss = self.train_generator(random_noise)

        return dis_log_loss, gen_log_loss

    def train_discriminator(self, X, Y):
        """One LSGAN discriminator update on goals X with labels Y (targets 2 * Y - 1)."""
        train_X = torch.from_numpy(X).float()
        train_Y = torch.from_numpy(Y).float()

        preds = self.discriminator.forward(train_X)
        loss = torch.mean(torch.square(2 * train_Y - 1 - preds))

        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()

        return loss.detach().numpy()

    def train_generator(self, X):
        """One LSGAN generator update from the latent noise X."""
        train_X = torch.from_numpy(X).float()

        generated_samples = self.generator.forward(train_X)
        loss = torch.mean(torch.square(self.discriminator.forward(generated_samples) - 1))

        self.generator_optimizer.zero_grad()
        loss.backward()
        self.generator_optimizer.step()

        return loss.detach().numpy()


class Generator(torch.nn.Module):
    def __init__(self, output_size, hidden_layers, noise_size):
        super(Generator, self).__init__()
        self.dummy_param = torch.nn.Parameter(torch.empty(0))

        layers = []
        input_dim = noise_size
        for size in hidden_layers:
            layers.append(torch.nn.Linear(input_dim, size, bias=True))
            input_dim = size
            layers.append(torch.nn.ReLU())

        layers.append(torch.nn.Linear(input_dim, output_size, bias=True))
        layers.append(torch.nn.Tanh())

        self.network = torch.nn.Sequential(*layers)

    def _get_device(self):
        return self.dummy_param.device

    def forward(self, noise):
        input_device = noise.device
        return self.network.forward(noise.to(self._get_device())).to(input_device)


class Discriminator(torch.nn.Module):

    def __init__(self, input_dim, hidden_layers, output_size):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.empty(0))
        layers = []
        for size in hidden_layers:
            layers.append(torch.nn.Linear(input_dim, size, bias=True))
            input_dim = size
            layers.append(torch.nn.LeakyReLU(0.1))

        layers.append(torch.nn.Linear(input_dim, output_size, bias=True))

        self.network = torch.nn.Sequential(*layers)

    def _get_device(self):
        return self.dummy_param.device

    def forward(self, samples):
        input_device = samples.device
        return self.network.forward(samples.to(self._get_device())).to(input_device)
