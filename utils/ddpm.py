import torch
import torch.nn as nn

import sys
import time
from tqdm import tqdm



class MyDDPM(nn.Module):
    def __init__(
        self,
        network,
        n_steps=200,
        min_beta=10**-4,
        max_beta=0.02,
        device=None,
    ):
        super(MyDDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.network = network.to(self.device)
        
        # Linear beta schedule; buffers ensure proper device management with state_dict.
        betas = torch.linspace(min_beta, max_beta, n_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

        self.to(self.device)

    def forward(self, x0, t, eta=None):
        """
        Return the noisy sample x_t via the forward (diffusion) process.

        Args:
            x0: clean input, shape [N, d].
            t: 1-D integer tensor of timestep indices, length N.
            eta: optional pre-sampled noise; sampled from N(0,I) if None.
        """
        a_bar = self.alpha_bars[t]
        n = len(a_bar)

        if eta is None:
            eta = torch.randn_like(x0).to(self.device)

        x0_noisy = (
            a_bar.sqrt().reshape(n, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1) * eta
        )
        return x0_noisy

    def backward(self, x, t):
        return self.network(x, t)


def training_loop(yx, ddpm, n_epochs, optim, device=None, store_path="tabular_ddpm.pt"):
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = ddpm.n_steps

    if device is None:
        device = ddpm.device

    for epoch in tqdm(range(n_epochs), desc=f"Training progress", leave=False):

        x0 = yx.to(device)
        n = len(x0)

        eta = torch.randn_like(x0).to(device)
        t = torch.randint(0, n_steps, (n,)).to(device)

        noisy_imgs = ddpm(x0, t, eta)
        eta_theta = ddpm.backward(noisy_imgs, t)

        loss = mse(eta_theta, eta)
        optim.zero_grad()
        loss.backward()
        optim.step()

        epoch_loss = loss.item()
        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        if best_loss > epoch_loss:
            best_loss = epoch_loss
            if store_path is not None:
                torch.save(ddpm.state_dict(), store_path)
                log_string += f" --> Best model ever (stored to {store_path})"
            else:
                log_string += f" --> Best model ever"

        print(log_string)


def generate_samples(
    ddpm,
    n_samples=16,
    device=None,
    tabular_dim=8,
):
    """Generate unconditional samples from a trained DDPM."""

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        # Starting from random noise
        x = torch.randn(n_samples, tabular_dim).to(device)

        looper = tqdm(
            enumerate(list(range(ddpm.n_steps))[::-1]), total=ddpm.n_steps, leave=False
        )
        for idx, t in looper:
            # Estimating noise to be removed
            time_tensor = (torch.ones(n_samples) * t).to(device).long()
            eta_theta = ddpm.backward(x, time_tensor)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            x = (1 / alpha_t.sqrt()) * (
                x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta
            )

            if t > 0:
                z = torch.randn(n_samples, tabular_dim).to(device)

                # DDPM posterior variance: beta_tilda_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
                beta_t = ddpm.betas[t]
                prev_alpha_t_bar = ddpm.alpha_bars[t - 1]
                beta_tilda_t = ((1 - prev_alpha_t_bar) / (1 - alpha_t_bar)) * beta_t
                sigma_t = beta_tilda_t.sqrt()

                x = x + sigma_t * z

            looper.set_description(f"Imputation at step {t}")

    return x


def generate_imputation(
    ddpm,
    yx,
    mask_yx,
    resampling_steps=1,
    device=None,
):
    """
    Impute missing values using an unconditional DDPM with replacement sampling.

    Args:
        yx: full input tensor (placeholder values for missing entries are ignored).
        mask_yx: binary mask, 1 for observed features and 0 for features to impute.
        resampling_steps: forward-backward resampling iterations per denoising step;
            higher values improve sample quality at the cost of compute.
    """

    assert yx.shape == mask_yx.shape, "yx and mask_yx should have the same shape"

    n_samples = yx.shape[0]
    x0, m = yx.clone(), mask_yx.clone()

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        x0, m = x0.to(device), m.to(device)

        x = torch.randn_like(x0).to(device)

        looper = tqdm(
            enumerate(list(range(ddpm.n_steps))[::-1]), total=ddpm.n_steps, leave=False
        )
        for idx, t in looper:
            time_tensor = (torch.ones(n_samples) * t).to(device).long()
            for u in range(resampling_steps):
                x_known = ddpm(x0, time_tensor)

                alpha_t = ddpm.alphas[t]
                alpha_t_bar = ddpm.alpha_bars[t]
                beta_t = ddpm.betas[t]

                # DDPM posterior variance: beta_tilda_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
                if t > 0:
                    prev_alpha_t_bar = ddpm.alpha_bars[t - 1]
                    beta_tilda_t = ((1 - prev_alpha_t_bar) / (1 - alpha_t_bar)) * beta_t
                    sigma_t = beta_tilda_t.sqrt()
                else:
                    sigma_t = 0

                eta_theta = ddpm.backward(x, time_tensor)
                z = torch.randn_like(x0).to(device) if t > 0 else 0
                x_unknown = (1 / alpha_t.sqrt()) * (
                    x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta
                ) + sigma_t * z

                x = m * x_known + (1 - m) * x_unknown

                # resampling: forward noising uses forward variance beta_t (not posterior)
                if u < resampling_steps - 1 and t > 0:
                    x = (1 - beta_t).sqrt() * x + beta_t.sqrt() * torch.randn_like(
                        x0
                    ).to(device)

            looper.set_description(f"Imputation at step {t}")

    return x

