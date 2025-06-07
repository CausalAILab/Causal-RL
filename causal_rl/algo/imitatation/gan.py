import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalGenerator(nn.Module):
    def __init__(self, cond_dim, num_actions, noise_dim = 3, hidden_dim = 64):
        super().__init__()

        self.cond_dim = cond_dim
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        self.fc1 = nn.Linear(cond_dim + noise_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_actions)

    def forward(self, cond, noise):
        x = torch.cat([cond, noise], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return logits
    

class Discriminator(nn.Module):
    def __init__(self, cond_dim, num_actions, hidden_dim = 64):
        super().__init__()

        self.cond_dim = cond_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(cond_dim + num_actions, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, cond, act):
        if act.dim() == 1 or act.size(-1) != self.num_actions:
            act = F.one_hot(act.long(), num_classes=self.num_actions).float()

        x = torch.cat([cond, act], dim=-1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        logits = self.out(x).squeeze(-1)
        return logits
    

class GANPolicy:
    def __init__(self, generator: ConditionalGenerator, noise_dim: int = 3, device = 'cpu'):
        self.generator = generator.to(device).eval()
        self.noise_dim = noise_dim
        self.device = device

    def __call__(self, cond_vec) -> int:
        cond = torch.tensor([cond_vec], dtype=torch.float32, device=self.device)
        noise = torch.randn((1, self.noise_dim), device=self.device)
        logits = self.generator(cond, noise)
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy().ravel()
        return int(np.random.choice(len(probs), p=probs))

    
def train_gan(
    generator: ConditionalGenerator,
    discriminator: Discriminator,
    data_loader: torch.utils.data.DataLoader,
    noise_dim: int = 3,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
    device: torch.device = None
) -> ConditionalGenerator:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator.to(device)
    discriminator.to(device)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        for cond, real_act in data_loader:
            cond = cond.to(device)
            real_act = real_act.to(device)
            batch_size = cond.size(0)

            # train discriminator
            real_labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)

            # real data
            real_logits = discriminator(cond, real_act)
            loss_real = criterion(real_logits, real_labels)

            # fake data
            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_logits_gen = generator(cond, noise)
            fake_act = F.gumbel_softmax(fake_logits_gen, dim=-1)
            fake_logits = discriminator(cond, fake_act.detach())
            loss_fake = criterion(fake_logits, fake_labels)

            loss_d = (loss_real + loss_fake) * 0.5
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            # train generator
            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_logits_gen = generator(cond, noise)
            fake_act = F.gumbel_softmax(fake_logits_gen, dim=-1)
            fake_logits = discriminator(cond, fake_act)
            loss_g = criterion(fake_logits, real_labels)
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

    return generator