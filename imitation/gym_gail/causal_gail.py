import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

def _split_token(tok: str):
    i = 0
    while i < len(tok) and not tok[i].isdigit():
        i += 1

    var = tok[:i]
    t = int(tok[i:])
    return var, t

def _width(v: str, obs: dict):
    if v not in obs or len(obs[v]) == 0:
        return 1

    sample = obs[v][0]
    arr = np.array(sample)
    return int(arr.size) if arr.ndim > 0 else 1

def calc_categorical_dims(env):
    categorical_dims = {'X': env.env.action_space.n}

    for v, space in env.env.observation_space.spaces.items():
        space = space.feature_space
        if isinstance(space, spaces.Discrete):
            categorical_dims[v] = space.n

    return categorical_dims

def _build_union_token_order(Z_sets):
    union_tokens = set()

    for _, toks in Z_sets.items():
        union_tokens.update(toks)

    return sorted(union_tokens, key=lambda p: (_split_token(p)[1], _split_token(p)[0]))

def _build_var_dims(union_tokens, obs, categorical_dims):
    var_dims = {}

    for tok in union_tokens:
        var, _ = _split_token(tok)

        if var not in var_dims:
            var_dims[var] = int(categorical_dims.get(var, _width(var, obs)))

    return var_dims

def _encode_with_union(obs, t, union_tokens, categorical_dims, var_dims):
    parts = []

    for tok in union_tokens:
        var, tt = _split_token(tok)
        width = var_dims[var]

        if (var not in obs) or (tt < 0) or (tt >= len(obs[var])):
            parts.append(np.zeros((width,), dtype=np.float32))
            continue

        v = obs[var][tt]

        if var in categorical_dims:
            dim = int(categorical_dims[var])
            vec = np.zeros((dim,), dtype=np.float32)
            idx = int(v)

            if 0 <= idx < dim:
                vec[idx] = 1.0

            parts.append(vec)

        else:
            arr = np.array(np.asarray(v, dtype=np.float32)).reshape(-1)

            if arr.shape[0] != width:
                out = np.zeros((width,), dtype=np.float32)
                out[:min(width, arr.shape[0])] = arr[:width]
                arr = out

            parts.append(arr.astype(np.float32, copy=False))

    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

def build_z_encoder(Z_sets, obs, categorical_dims, order='time_name'):
    union_tokens = _build_union_token_order(Z_sets)
    var_dims = _build_var_dims(union_tokens, obs, categorical_dims)
    z_dim = int(sum(var_dims[_split_token(tok)[0]] for tok in union_tokens))

    step_masks = {}
    for xk, toks in Z_sets.items():
        _, t = _split_token(xk)
        keep = set(toks)
        step_masks[t] = np.array([tok in keep for tok in union_tokens], dtype=bool)

    idxs = []
    off = 0
    for tok in union_tokens:
        w = var_dims[_split_token(tok)[0]]
        idxs.append(slice(off, off + w))
        off += w

    def encode(obs, t):
        z_full = _encode_with_union(obs, t, union_tokens, categorical_dims, var_dims)

        if t in step_masks:
            mask = step_masks[t]
            per_feat = np.concatenate([np.full(idx.stop - idx.start, m, dtype=bool) for m, idx in zip(mask, idxs)], axis=0)
            z_full[~per_feat] = 0.0

        return z_full

    return encode, z_dim, union_tokens, var_dims

def build_policy_input(z, a, num_actions):
    a_oh = F.one_hot(a.long(), num_classes=num_actions).float().to(z.device)
    x = torch.cat([z, a_oh], dim=1)
    return x

def make_expert_batch(records, encode, num_actions):
    Z_e = []
    A_e = []

    for r in records:
        t = r['step']
        obs = r['obs']
        action = int(r['action'])

        Z_e.append(encode(obs, t))
        A_e.append(action)

    Z_e = torch.from_numpy(np.stack(Z_e, axis=0)).float()
    A_e = torch.tensor(A_e, dtype=torch.long)
    X_e = build_policy_input(Z_e, A_e, num_actions)
    return Z_e, A_e, X_e

def rollout_policy(env, actor, critic, encode, num_actions, max_steps, num_episodes, seed=None):
    device = next(actor.parameters()).device

    all_Z = []
    all_action = []
    all_logp = []
    all_entropy = []
    all_values = []
    all_dones = []
    ep_lens = []
    last_values = []
    rewards = []

    for e in range(num_episodes):
        if seed is not None:
            obs, _ = env.reset(seed=seed + e)
        else:
            obs, _ = env.reset()

        t = 0
        done = False
        steps = 0
        reward = 0.0

        while not done and steps < max_steps:
            z_np = encode(obs, t)
            z = torch.from_numpy(z_np).float().unsqueeze(0).to(device)

            with torch.no_grad():
                action, logp, entropy = actor.act(z)
                value = critic(z).squeeze(1)

            a = int(action.item())
            next_obs, r, terminated, truncated, _ = env.do(lambda _: a, show_reward=True)
            done = terminated or truncated
            reward += r

            all_Z.append(z.squeeze(0))
            all_action.append(torch.tensor(a, dtype=torch.long, device=device))
            all_logp.append(logp.squeeze(0))
            all_entropy.append(entropy.squeeze(0))
            all_values.append(value.squeeze(0))
            all_dones.append(done)

            obs = next_obs
            t += 1
            steps += 1

        z_np_last = encode(obs, t)
        z_last = torch.from_numpy(z_np_last).float().unsqueeze(0).to(device)

        with torch.no_grad():
            v_last = critic(z_last).squeeze(0)

        last_values.append(v_last)
        ep_lens.append(steps)
        rewards.append(reward)

    Z_pi = torch.stack(all_Z, dim=0).to(device)
    A_pi = torch.stack(all_action, dim=0).to(device)
    X_pi = build_policy_input(Z_pi, A_pi, num_actions).to(device)
    logp = torch.stack(all_logp, dim=0).to(device)
    entropy = torch.stack(all_entropy, dim=0).to(device)
    values = torch.stack(all_values, dim=0).to(device)
    dones = torch.tensor(all_dones, dtype=torch.bool, device=device)

    return {
        'Z': Z_pi,
        'A': A_pi,
        'X': X_pi,
        'logp': logp,
        'entropy': entropy,
        'values': values,
        'dones': dones,
        'last_values': torch.stack(last_values, dim=0).to(device),
        'ep_lens': ep_lens,
        'rewards': rewards
    }

def discriminator_reward(scores, loss_type='linear'):
    if loss_type == 'linear':
        return scores

    elif loss_type in {'bce', 'gail'}:
        return F.softplus(scores)

    else:
        raise ValueError(f'Unsupported loss_type: {loss_type}')

def discriminator_loss_linear(D_real, D_fake, gp=None, gp_lambda=10.0):
    loss = -D_real.mean() + D_fake.mean()

    if gp is not None:
        loss += gp_lambda * gp

    return loss

def discriminator_loss_bce(D_real_logits, D_fake_logits, gp=None, gp_lambda=10.0):
    loss_real = F.binary_cross_entropy_with_logits(D_real_logits, torch.ones_like(D_real_logits))
    loss_fake = F.binary_cross_entropy_with_logits(D_fake_logits, torch.zeros_like(D_fake_logits))

    loss = loss_real + loss_fake

    if gp is not None:
        loss += gp_lambda * gp

    return loss

def gradient_penalty(D, real_samples, fake_samples):
    batch_size = real_samples.size(0)
    eps = torch.rand(batch_size, 1, device=real_samples.device)
    eps = eps.expand_as(real_samples)

    interpolates = eps * real_samples + (1 - eps) * fake_samples
    interpolates.requires_grad_(True)

    d_interpolates = D(interpolates)
    ones = torch.ones_like(d_interpolates, device=real_samples.device)

    grads = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    grads = grads.view(batch_size, -1)
    gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
    return gp

def compute_gae(rewards, values, dones, last_values, gamma=0.99, lam=0.95, ep_lens=None):
    rewards = rewards.view(-1)
    values = values.view(-1)
    dones = dones.view(-1)
    last_values = last_values.view(-1)

    T = rewards.numel()
    assert T == values.numel() == dones.numel() # must have same length

    device = rewards.device

    if ep_lens is None:
        done_idx = torch.nonzero(dones, as_tuple=False).view(-1).tolist()
        ep_lens = []
        prev = -1

        for di in done_idx:
            ep_lens.append(int(di - prev))
            prev = di

    advantages = torch.zeros(T, dtype=rewards.dtype, device=device)
    returns = torch.zeros(T, dtype=rewards.dtype, device=device)

    idx = 0
    for e, e_len in enumerate(ep_lens):
        r_e = rewards[idx:idx + e_len]
        v_e = values[idx:idx + e_len]
        d_e = dones[idx:idx + e_len].to(dtype=rewards.dtype, device=device)

        v_boot = last_values[e].reshape(1).to(dtype=rewards.dtype, device=device)
        v_ext = torch.cat([v_e, v_boot], dim=0)

        gae = torch.zeros(1, dtype=rewards.dtype, device=device)
        for t in range(e_len - 1, -1, -1):
            nonterm = 1.0 - d_e[t]
            delta = r_e[t] + gamma * v_ext[t + 1] * nonterm - v_ext[t]
            gae = delta + gamma * lam * nonterm * gae
            advantages[idx + t] = gae
            returns[idx + t] = gae + v_ext[t]

        idx += e_len

    return advantages, returns

def ppo_update(
    actor,
    critic,
    actor_optim,
    critic_optim,
    Z,
    A,
    logp_old,
    advantages,
    returns,
    epochs=10,
    minibatch_size=2048,
    clip_eps=0.2,
    entropy_coeff=1e-2,
    value_coeff=0.5,
    max_grad_norm=0.5,
    normalize_adv=False
):
    N = Z.size(0)
    logp_old = logp_old.view(-1)
    advantages = advantages.view(-1)
    returns = returns.view(-1)

    if normalize_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    total_actor_loss = 0.0
    total_critic_loss = 0.0
    total_entropy = 0.0
    total_approx_kl = 0.0
    total_clip_frac = 0.0
    total_batches = 0

    idx_all = torch.arange(N)

    for _ in range(epochs):
        perm = idx_all[torch.randperm(N)]

        for start in range(0, N, minibatch_size):
            mb = perm[start:start + minibatch_size]
            if mb.numel() == 0:
                continue

            z_mb = Z[mb]
            a_mb = A[mb]
            logp_old_mb = logp_old[mb].detach()
            adv_mb = advantages[mb].detach()
            ret_mb = returns[mb].detach()

            logp_new_mb, ent_mb = actor.evaluate_actions(z_mb, a_mb)
            value_mb = critic(z_mb).squeeze(-1)

            ratio = (logp_new_mb - logp_old_mb).exp()
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_mb

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.smooth_l1_loss(value_mb, ret_mb)
            entropy_loss = -ent_mb.mean()

            actor_loss = policy_loss + entropy_coeff * entropy_loss
            critic_loss = value_coeff * value_loss

            actor_optim.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()

            critic_optim.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optim.step()

            with torch.no_grad():
                approx_kl = (logp_old_mb - logp_new_mb).mean()
                clip_frac = (ratio - 1.0).abs().gt(clip_eps).float().mean()

            total_actor_loss += float(actor_loss.detach())
            total_critic_loss += float(critic_loss.detach())
            total_entropy += float(ent_mb.mean().detach())
            total_approx_kl += float(approx_kl.detach())
            total_clip_frac += float(clip_frac.detach())
            total_batches += 1

    if total_batches == 0:
        return {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'entropy': 0.0,
            'approx_kl': 0.0,
            'clip_frac': 0.0
        }

    return {
        'actor_loss': total_actor_loss / total_batches,
        'critic_loss': total_critic_loss / total_batches,
        'entropy': total_entropy / total_batches,
        'approx_kl': total_approx_kl / total_batches,
        'clip_frac': total_clip_frac / total_batches
    }

def train_discriminator(
    discriminator,
    discriminator_optim,
    X_e,
    X_pi,
    loss_type='bce',
    d_updates=1,
    minibatch_size=4096,
    use_gp=True,
    gp_lambda=10.0,
    instance_noise_std=0.0,
    label_smoothing=0.0
):
    device = next(discriminator.parameters()).device
    discriminator.train()

    X_e = X_e.to(device, non_blocking=True)
    X_pi = X_pi.to(device, non_blocking=True)

    N_real = X_e.size(0)
    N_fake = X_pi.size(0)
    N = int(min(N_real, N_fake))

    if N == 0:
        return {
            'loss': 0.0,
            'D_real': 0.0,
            'D_fake': 0.0,
            'gp': 0.0,
            'accuracy': 0.0
        }

    idx_real = torch.randperm(N_real, device=device)[:N]
    idx_fake = torch.randperm(N_fake, device=device)[:N]
    Xr_all = X_e[idx_real]
    Xf_all = X_pi[idx_fake]

    total_loss = 0.0
    total_D_real = 0.0
    total_D_fake = 0.0
    total_gp = 0.0
    total_acc = 0.0
    total_batches = 0

    for _ in range(d_updates):
        perm = torch.randperm(N, device=device)
        Xr = Xr_all[perm]
        Xf = Xf_all[perm]

        for start in range(0, N, minibatch_size):
            r = Xr[start:start + minibatch_size]
            f = Xf[start:start + minibatch_size]

            if r.numel() == 0 or f.numel() == 0:
                continue

            if instance_noise_std > 0.0:
                noise_r = torch.randn_like(r, device=device) * instance_noise_std
                noise_f = torch.randn_like(f, device=device) * instance_noise_std
                r = r + noise_r
                f = f + noise_f

            D_real = discriminator(r)
            D_fake = discriminator(f)

            gp = None
            if use_gp:
                gp = gradient_penalty(discriminator, r.detach(), f.detach())

            if loss_type == 'linear':
                d_loss = discriminator_loss_linear(D_real, D_fake, gp, gp_lambda)

            else:
                if label_smoothing > 0.0:
                    tgt_real = torch.full_like(D_real, 1.0 - label_smoothing, device=device)
                    tgt_fake = torch.full_like(D_fake, label_smoothing, device=device)

                else:
                    tgt_real = torch.ones_like(D_real, device=device)
                    tgt_fake = torch.zeros_like(D_fake, device=device)

                bce_real = F.binary_cross_entropy_with_logits(D_real, tgt_real)
                bce_fake = F.binary_cross_entropy_with_logits(D_fake, tgt_fake)
                d_loss = bce_real + bce_fake

                if gp is not None:
                    d_loss += gp_lambda * gp

            discriminator_optim.zero_grad(set_to_none=True)
            d_loss.backward()
            discriminator_optim.step()

            with torch.no_grad():
                if loss_type == 'linear':
                    p_real = (torch.sigmoid(D_real) >= 0.5).float().mean()
                    p_fake = (torch.sigmoid(D_fake) < 0.5).float().mean()
                    acc = 0.5 * (p_real + p_fake)

                else:
                    acc = 0.5 * ((D_real >= 0.0).float().mean() + (D_fake < 0.0).float().mean())

                total_loss += float(d_loss)
                total_D_real += float(D_real.mean())
                total_D_fake += float(D_fake.mean())
                total_gp += float(gp) if gp is not None else 0.0
                total_acc += float(acc)
                total_batches += 1

    if total_batches == 0:
        return {
            'loss': 0.0,
            'D_real': 0.0,
            'D_fake': 0.0,
            'gp': 0.0,
            'accuracy': 0.0
        }
    
    return {
        'loss': total_loss / total_batches,
        'D_real': total_D_real / total_batches,
        'D_fake': total_D_fake / total_batches,
        'gp': total_gp / total_batches if use_gp else 0.0,
        'accuracy': total_acc / total_batches
    }

def one_training_round(
    env,
    actor,
    critic,
    discriminator,
    actor_optim,
    critic_optim,
    discriminator_optim,
    encode,
    num_actions,
    X_e=None,
    expert_records=None,
    gamma=0.99,
    gae_lambda=0.95,
    ppo_clip=0.2,
    epochs=10,
    minibatch_size=2048,
    entropy_coeff=1e-2,
    value_coeff=0.5,
    max_grad_norm=0.5,
    normalize_adv=False,
    loss_type='bce',
    gp_lambda=10.0,
    d_updates=3,
    d_minibatch_size=4096,
    use_gp=True,
    instance_noise_std=0.0,
    label_smoothing=0.0,
    max_steps=10,
    num_episodes=10,
    seed=None
):
    actor.train()
    critic.train()
    discriminator.train()
    a_device = next(actor.parameters()).device
    c_device = next(critic.parameters()).device
    d_device = next(discriminator.parameters()).device

    assert a_device == c_device, 'Actor and critic must be on the same device.'

    if X_e is None:
        if expert_records is None or len(expert_records) == 0:
            raise ValueError('Either X_e or expert_records must be provided.')

        _Z_e, _A_e, X_e_cpu = make_expert_batch(expert_records, encode, num_actions)
        X_e = X_e_cpu.to(d_device, non_blocking=True)
    else:
        X_e = X_e.to(d_device, non_blocking=True)

    roll = rollout_policy(
        env,
        actor,
        critic,
        encode,
        num_actions,
        max_steps,
        num_episodes,
        seed
    )

    Z_pi = roll['Z'].to(a_device, non_blocking=True)
    A_pi = roll['A'].to(a_device, non_blocking=True)
    X_pi = roll['X']
    logp_old = roll['logp'].to(a_device, non_blocking=True).view(-1)
    values_old = roll['values'].to(a_device, non_blocking=True).view(-1)
    dones = roll['dones'].to(a_device, non_blocking=True).view(-1)
    last_values = roll['last_values'].to(a_device, non_blocking=True).view(-1)

    with torch.no_grad():
        X_pi_d = X_pi.to(d_device, non_blocking=True)
        D_scores_pi = discriminator(X_pi_d)
        r_D = discriminator_reward(D_scores_pi, loss_type=loss_type)
        r_D = r_D.to(a_device, non_blocking=True).view(-1)

    advantages, returns = compute_gae(
        r_D,
        values_old,
        dones,
        last_values,
        gamma=gamma,
        lam=gae_lambda,
        ep_lens=roll['ep_lens']
    )

    advantages = advantages.to(a_device, non_blocking=True)
    returns = returns.to(a_device, non_blocking=True)

    ppo_stats = ppo_update(
        actor,
        critic,
        actor_optim,
        critic_optim,
        Z_pi,
        A_pi,
        logp_old,
        advantages,
        returns,
        epochs=epochs,
        minibatch_size=minibatch_size,
        clip_eps=ppo_clip,
        entropy_coeff=entropy_coeff,
        value_coeff=value_coeff,
        max_grad_norm=max_grad_norm,
        normalize_adv=normalize_adv
    )

    d_stats = train_discriminator(
        discriminator,
        discriminator_optim,
        X_e,
        X_pi_d,
        loss_type=loss_type,
        d_updates=d_updates,
        minibatch_size=d_minibatch_size,
        use_gp=use_gp,
        gp_lambda=gp_lambda,
        instance_noise_std=instance_noise_std,
        label_smoothing=label_smoothing
    )

    with torch.no_grad():
        sl = slice(0, min(4096, X_e.size(0)))
        D_real_log = discriminator(X_e[sl]).mean().item()
        slf = slice(0, min(4096, X_pi_d.size(0)))
        D_fake_log = discriminator(X_pi_d[slf]).mean().item()

        avg_env_return = float(np.mean(roll['rewards']))
        avg_D_reward = float(torch.mean(r_D).item())

    return {
        'avg_env_return': avg_env_return,
        'avg_D_reward': avg_D_reward,

        'ppo_actor_loss': ppo_stats.get('actor_loss', 0.0),
        'ppo_critic_loss': ppo_stats.get('critic_loss', 0.0),
        'ppo_entropy': ppo_stats.get('entropy', 0.0),
        'ppo_approx_kl': ppo_stats.get('approx_kl', 0.0),
        'ppo_clip_frac': ppo_stats.get('clip_frac', 0.0),

        'D_loss': d_stats.get('loss', 0.0),
        'D_real_mean': d_stats.get('D_real', D_real_log),
        'D_fake_mean': d_stats.get('D_fake', D_fake_log),
        'D_gp': d_stats.get('gp', 0.0),
        'D_accuracy': d_stats.get('accuracy', 0.0),

        'ep_lens': roll['ep_lens'],
        'n_steps': int(Z_pi.size(0)),
        'n_episodes': int(len(roll['ep_lens']))
    }