import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import math
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np # Needed for image permute
from torch_fidelity import calculate_metrics # <-- ADDED for FID
import torchvision.utils as vutils # <-- ADDED for saving individual images
from torch.utils.data import TensorDataset # <-- ADDED for wrapping generated samples

# --- Hyperparameters ---
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# <<< MODIFIED FOR CIFAR >>>
IMG_SIZE = 32
IMG_CHANNELS = 3
RESULTS_DIR = "imm_cifar10_results_v2" # New results dir
EPOCHS = 250
BATCH_SIZE = 128
# <<< END MODIFIED >>>

# <<< MODIFIED HYPERPARAMETERS based on Paper Analysis >>>
M_PARTICLES = 4 # Paper suggests M=4 might be better (Fig 5)
LEARNING_RATE = 1e-4
EMA_DECAY = 0.9999 # Often higher decay is used
TIME_EMB_DIM = 128 # Dimension for sinusoidal time embeddings
T_MAX = 0.994 # Recommended for OT-FM with eta-based r(s,t)
T_EPSILON = 1e-3 # Smallest time step
ETA_DECREMENT_K = 12 # Controls r(s,t) gap, e.g., (max-min)/2^k. 12 is from ImageNet Table 5
# R_DELTA = 0.1 # Replaced by ETA_DECREMENT_K
KERNEL_EPS = 1e-8 # Small epsilon for kernel stability
GRAD_CLIP_NORM = 1.0
SAVE_INTERVAL = 10
SIGMA_D = 0.5 # Data standard deviation assumption (Table 5)
WEIGHTING_B = 4.0 # VDM Weighting parameter 'b' (Table 5)
WEIGHTING_A = 2.0 # VDM Weighting parameter 'a' (a=2 recommended)
# <<< END MODIFIED >>>

# <<< ADDED: Set multiprocessing start method for CUDA compatibility >>>
try:
    mp.set_start_method('spawn', force=True)
    print("Set multiprocessing start method to 'spawn'")
except RuntimeError as e:
    # This might happen if it was already set, or in certain environments
    # like interactive notebooks where it can only be set once.
    print(f"Note: Could not set multiprocessing start method ('{e}'). Assuming it's already configured or not needed.")
# <<< END ADDED >>>

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# --- Helper Functions ---

# --- Time/Schedule Helpers (OT-FM: alpha_t=1-t, sigma_t=t) ---
def get_ot_fm_schedule(t):
    """OT-FM interpolation schedule (αt = 1-t, σt = t)."""
    # Clamp t to avoid issues at boundaries
    t = torch.clamp(t, T_EPSILON, T_MAX)
    alpha_t = 1.0 - t
    sigma_t = t
    return alpha_t, sigma_t

def eta_t_ot_fm(t):
    """eta_t = sigma_t / alpha_t for OT-FM"""
    t = torch.clamp(t, T_EPSILON, T_MAX) # Ensure t is within valid range
    # Avoid division by zero when t is close to 1 (alpha_t close to 0)
    alpha_t = torch.clamp(1.0 - t, min=1e-9)
    return t / alpha_t

def inverse_eta_t_ot_fm(eta):
    """Inverse of eta_t for OT-FM: t = eta / (1 + eta)"""
    # Clamp eta to avoid issues with large values or negatives
    eta = torch.clamp(eta, min=1e-9)
    return eta / (1.0 + eta)

def get_r_fn_constant_eta_decrement(eta_max_val=160.0, eta_min_val=eta_t_ot_fm(torch.tensor(T_EPSILON)).item(), k=ETA_DECREMENT_K):
    """Returns a function r(s, t) based on constant decrement in eta_t space."""
    eta_decrement = (eta_max_val - eta_min_val) / (2**k)
    # Precompute min_eta_for_r to avoid redundant calculations
    min_eta_for_r = eta_t_ot_fm(torch.tensor(T_EPSILON)) + eta_decrement

    def r_fn(s, t):
        # Ensure t is within valid range before calculating eta_t
        t_clamped = torch.clamp(t, T_EPSILON, T_MAX)
        eta_t_val = eta_t_ot_fm(t_clamped)
        # Ensure target eta is not below the minimum possible after decrement
        target_eta = torch.clamp(eta_t_val - eta_decrement, min=eta_t_ot_fm(torch.tensor(T_EPSILON, device=t.device)))
        # Calculate potential r based on inverse eta
        r_candidate = inverse_eta_t_ot_fm(target_eta)
        # Clamp r_candidate to be <= t
        r_candidate = torch.min(r_candidate, t_clamped)
        # Ensure r >= s
        r = torch.max(s, r_candidate)
        # Final clamp to ensure r is within valid time bounds
        return torch.clamp(r, T_EPSILON, T_MAX)

    return r_fn

r_function = get_r_fn_constant_eta_decrement()

def logSNR_t_ot_fm(t):
    """logSNR_t = log(alpha_t^2 / sigma_t^2) for OT-FM"""
    t = torch.clamp(t, T_EPSILON, T_MAX)
    alpha_t = 1.0 - t
    sigma_t = t
    # Add small epsilon to avoid log(0) or division by zero
    alpha_t_sq = torch.clamp(alpha_t**2, min=1e-9)
    sigma_t_sq = torch.clamp(sigma_t**2, min=1e-9)
    return torch.log(alpha_t_sq / sigma_t_sq)

def dlogSNR_dt_ot_fm(t):
    """Derivative of logSNR_t w.r.t t for OT-FM"""
    t = torch.clamp(t, T_EPSILON, T_MAX)
    alpha_t = 1.0 - t
    sigma_t = t
    # Avoid division by zero
    alpha_t = torch.clamp(alpha_t, min=1e-9)
    sigma_t = torch.clamp(sigma_t, min=1e-9)
    # Derivative: d/dt [2*log(1-t) - 2*log(t)] = 2*(-1/(1-t)) - 2*(1/t) = -2/(t(1-t))
    return -2.0 / (sigma_t * alpha_t)

def get_vdm_weighting(t, a=WEIGHTING_A, b=WEIGHTING_B):
    """Calculates VDM weighting w(t) based on logSNR (Eq 13 adaptation).
       Note: Original paper's w(s,t) depends only on t in the final form used.
    """
    t_clamped = torch.clamp(t, T_EPSILON, T_MAX)
    lambda_t = logSNR_t_ot_fm(t_clamped)
    dlambda_dt = dlogSNR_dt_ot_fm(t_clamped)
    alpha_t, sigma_t = get_ot_fm_schedule(t_clamped)

    # Add epsilon to denominator
    alpha_t_sq_plus_sigma_t_sq = torch.clamp(alpha_t**2 + sigma_t**2, min=1e-9)

    weight = 0.5 * torch.sigmoid(b - lambda_t) * (-dlambda_dt) * (alpha_t**a) / alpha_t_sq_plus_sigma_t_sq
    # Clamp weight to avoid extreme values
    return torch.clamp(weight, min=1e-6, max=1e6)


# --- Euler-FM Parameterization Helpers ---
def get_euler_fm_coefficients(s, t):
    """Calculate coefficients for f_st = c_skip * xt + c_out * G_theta"""
    # Ensure inputs are tensors
    s = torch.as_tensor(s, device=DEVICE) if not isinstance(s, torch.Tensor) else s.to(DEVICE)
    t = torch.as_tensor(t, device=DEVICE) if not isinstance(t, torch.Tensor) else t.to(DEVICE)

    # Clamp times to valid range
    s = torch.clamp(s, T_EPSILON, T_MAX)
    t = torch.clamp(t, T_EPSILON, T_MAX)

    c_skip = torch.ones_like(t) # c_skip(s,t) = 1
    c_out = -(t - s) * SIGMA_D   # c_out(s,t) = -(t-s) * sigma_d
    return c_skip, c_out

def get_input_conditioning_scale(t):
    """Calculate c_in(t) = 1 / (sigma_d * sqrt(alpha_t^2 + sigma_t^2))"""
    alpha_t, sigma_t = get_ot_fm_schedule(t)
    # Add epsilon to denominator
    denom = SIGMA_D * torch.sqrt(torch.clamp(alpha_t**2 + sigma_t**2, min=1e-9))
    return 1.0 / torch.clamp(denom, min=1e-9)


# --- Other Helpers ---
def sinusoidal_embedding(t, dim):
    """Sinusoidal time embeddings."""
    if dim <= 1:
        return t[:, None]

    half_dim = dim // 2
    denominator = half_dim - 1 if half_dim > 1 else 1
    # Ensure denominator is non-zero and calculation is stable
    if denominator == 0:
      emb_val = 0.0 # Avoid log(10000)/0
    else:
      emb_val = math.log(10000) / denominator

    emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb_val)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    if dim % 2 == 1: # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb

def ddim_interpolate(xt, x, t, target_t, schedule_fn):
    """DDIM formula to interpolate from xt at time t towards x at target_t < t."""
    alpha_s, sigma_s = schedule_fn(target_t)
    alpha_t, sigma_t = schedule_fn(t)
    sigma_t = torch.clamp(sigma_t, min=1e-9) # Avoid division by zero
    sigma_s_over_sigma_t = sigma_s / sigma_t
    term1_coeff = alpha_s - sigma_s_over_sigma_t * alpha_t
    term2_coeff = sigma_s_over_sigma_t
    target_x = term1_coeff[:, None, None, None] * x + term2_coeff[:, None, None, None] * xt
    return target_x

# <<< MODIFIED: Combined ddim_sample_step logic into apply_euler_fm_step >>>
# def ddim_sample_step(xt, x_pred, s, t, schedule_fn): ... (Removed)

def apply_euler_fm_step(xt, G_theta_output, s, t):
    """Applies the Euler-FM step: f_st = xt - (t-s)*sigma_d*G_theta"""
    c_skip, c_out = get_euler_fm_coefficients(s, t)
    # Ensure shapes match for broadcasting if needed
    while c_skip.dim() < xt.dim():
      c_skip = c_skip.unsqueeze(-1)
      c_out = c_out.unsqueeze(-1)
    xs = c_skip * xt + c_out * G_theta_output
    return xs


def laplace_kernel(x, y, cout_st):
    """Laplace kernel: exp(-||x - y||_1 / (|cout_st| * D))"""
    # Ensure cout_st is positive and non-zero
    abs_cout_st = torch.abs(cout_st) + 1e-9 # Add epsilon for stability

    if x.ndim < 2 or y.ndim < 2:
         return torch.tensor(1.0, device=x.device, dtype=x.dtype if torch.is_tensor(x) else torch.float32)

    if x.shape[1:] != y.shape[1:]:
        raise ValueError(f"Input shapes must match except for batch dim. Got {x.shape} and {y.shape}")

    # Calculate dimension D based on input shape (excluding batch)
    D = float(np.prod(x.shape[1:]))
    if D == 0:
        return torch.ones_like(x.reshape(x.shape[0], -1).sum(dim=1))

    # L1 norm
    diff_norm = torch.sum(torch.abs(x.reshape(x.shape[0], -1) - y.reshape(y.shape[0], -1)), dim=1)
    diff_norm = torch.clamp(diff_norm, min=1e-9) # Avoid issues if x=y

    # Denominator: |c_out(s,t)| * D
    denominator = abs_cout_st * D
    denominator = torch.clamp(denominator, min=1e-9) # Final safety clamp

    return torch.exp(-diff_norm / denominator)

def update_ema(ema_model, model, decay):
    """Update Exponential Moving Average model."""
    with torch.no_grad():
        model_params = dict(model.named_parameters())
        ema_params = dict(ema_model.named_parameters())
        for name, param in model_params.items():
            if param.requires_grad and name in ema_params:
                 ema_params[name].data.mul_(decay).add_(param.data, alpha=1 - decay)


# --- Model Architecture (Simple U-Net adjusted for CIFAR) ---
# <<< MODIFIED: Input now mainly uses t_emb_t for G_theta conditioning >>>
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, use_groupnorm=True):
        super().__init__()
        # Time embedding MLP applied to the projected time embedding t
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch) if use_groupnorm else nn.Identity()
        self.act1 = nn.SiLU()
        # Ensure the second convolution uses out_ch for its input channels
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch) if use_groupnorm else nn.Identity()
        self.act2 = nn.SiLU()

    def forward(self, x, t_emb_proj): # Takes projected time embedding
        h = self.conv1(x)
        h = self.norm1(h)
        # Add time embedding *after* first conv and norm
        # Reshape time embedding to [B, C, 1, 1] for broadcasting
        time_cond = self.time_mlp(self.act1(t_emb_proj)) # [B, C]
        h = h + time_cond[:, :, None, None] # Add broadcasting dimensions
        h = self.act1(h) # Apply activation

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)
        return h

class SimpleUNet(nn.Module):
    def __init__(self, img_channels=3, time_emb_dim=TIME_EMB_DIM, base_dim=128): # Adjusted base_dim
        super().__init__()
        # We only need embedding dim for t for G_theta input
        self.time_input_dim = time_emb_dim

        # Time embedding projection (conditioning for G_theta based on t)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_input_dim, self.time_input_dim * 4),
            nn.SiLU(),
            nn.Linear(self.time_input_dim * 4, self.time_input_dim)
        )

        # Contracting path (Increased width)
        self.down1 = Block(img_channels, base_dim, self.time_input_dim) # e.g., 3 -> 128
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = Block(base_dim, base_dim * 2, self.time_input_dim) # 128 -> 256
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bot1 = Block(base_dim * 2, base_dim * 4, self.time_input_dim) # 256 -> 512

        # Expansive path
        self.upconv1 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, 2, stride=2) # 512 -> 256
        self.up1 = Block(base_dim * 4, base_dim * 2, self.time_input_dim) # Input: 256 (skip) + 256 (up)
        self.upconv2 = nn.ConvTranspose2d(base_dim * 2, base_dim, 2, stride=2) # 256 -> 128
        self.up2 = Block(base_dim * 2, base_dim, self.time_input_dim) # Input: 128 (skip) + 128 (up)

        # Output layer predicts G_theta output (e.g., noise prediction target)
        self.out = nn.Conv2d(base_dim, img_channels, 1) # 128 -> 3

    def forward(self, x_t_conditioned, t_emb_t): # Takes conditioned input and t embedding
        # Project time embedding for t
        t_proj = self.time_mlp(t_emb_t)

        # Contracting path
        x1 = self.down1(x_t_conditioned, t_proj)
        p1 = self.pool1(x1)
        x2 = self.down2(p1, t_proj)
        p2 = self.pool2(x2)

        # Bottleneck
        b = self.bot1(p2, t_proj)

        # Expansive path
        u1 = self.upconv1(b)
        c1 = torch.cat([u1, x2], dim=1)
        x3 = self.up1(c1, t_proj)

        u2 = self.upconv2(x3)
        c2 = torch.cat([u2, x1], dim=1)
        x4 = self.up2(c2, t_proj)

        # Output: G_theta(c_in(t)*xt, c_noise(t))
        G_theta_output = self.out(x4)
        return G_theta_output # Predicts the noise term G_theta

# --- IMM Loss Function ---
# <<< MODIFIED: Uses Euler-FM, time-dep kernel, VDM weighting, eta-based r(s,t) >>>
def imm_loss(model, model_ema, x_batch, M, schedule_fn, r_func, kernel_fn):
    """Computes the Inductive Moment Matching loss with Euler-FM."""
    B, C, H, W = x_batch.shape
    if B == 0: return torch.tensor(0.0, device=DEVICE, requires_grad=True)
    if B % M != 0:
       print(f"Warning: Batch size {B} not divisible by M={M}. Adjusting batch size for loss.")
       # Simple truncation for now, could implement padding/masking
       B = (B // M) * M
       if B == 0: return torch.tensor(0.0, device=DEVICE, requires_grad=True)
       x_batch = x_batch[:B]
       # Consider raising an error or implementing more robust handling

    num_groups = B // M

    # Sample times t, s for the entire batch
    # Sample t uniformly from [epsilon, T_MAX]
    t = torch.rand(B, device=DEVICE) * (T_MAX - T_EPSILON) + T_EPSILON
    # Sample s uniformly from [epsilon, t]
    s = torch.rand(B, device=DEVICE) * (t - T_EPSILON) + T_EPSILON
    s = torch.clamp(s, min=T_EPSILON) # Ensure s >= epsilon

    # Calculate r = r_func(s, t) using the new function
    r = r_func(s, t)
    r = torch.clamp(r, min=T_EPSILON) # Ensure r >= epsilon

    # Get time embeddings for t (used for G_theta input conditioning)
    t_emb = sinusoidal_embedding(t, TIME_EMB_DIM)

    # Calculate VDM weighting based on t
    weights = get_vdm_weighting(t) # Shape (B,)

    # Calculate xt ~ N(alpha_t * x, sigma_t^2 * I)
    eps = torch.randn_like(x_batch) * SIGMA_D # Scale noise by assumed data std dev
    alpha_t, sigma_t = schedule_fn(t)
    xt = alpha_t[:, None, None, None] * x_batch + sigma_t[:, None, None, None] * eps

    # Calculate xr = DDIM(xt, x, r, t) by reusing ground truth x
    with torch.no_grad():
        xr = ddim_interpolate(xt, x_batch, t, r, schedule_fn)

    # Calculate input conditioning scales c_in(t) and c_in(r)
    c_in_t = get_input_conditioning_scale(t)
    c_in_r = get_input_conditioning_scale(r)

    # Prepare model inputs (apply c_in scaling)
    # Expand c_in to match spatial dimensions
    while c_in_t.dim() < xt.dim():
      c_in_t = c_in_t.unsqueeze(-1)
      c_in_r = c_in_r.unsqueeze(-1)

    model_input_t = c_in_t * xt
    model_input_r = c_in_r * xr

    total_loss = 0.0
    processed_samples = 0

    # Process in groups of M
    for i in range(num_groups):
        start_idx = i * M
        end_idx = (i + 1) * M

        # Get group data
        model_input_t_group = model_input_t[start_idx:end_idx]
        xt_group = xt[start_idx:end_idx] # Need original xt for Euler step
        model_input_r_group = model_input_r[start_idx:end_idx]
        xr_group = xr[start_idx:end_idx] # Need original xr for Euler step
        t_emb_group = t_emb[start_idx:end_idx] # Only need t embedding for G_theta
        s_group_val = s[start_idx:end_idx]
        t_group_val = t[start_idx:end_idx]
        r_group_val = r[start_idx:end_idx]
        weights_group = weights[start_idx:end_idx].mean() # Use mean weight for the group

        # Predict G_theta output using current model for t -> s path
        # Input: c_in(t)*xt, t_emb
        G_theta_pred_t = model(model_input_t_group, t_emb_group)

        # Predict G_theta output using EMA model for r -> s path
        # Input: c_in(r)*xr, r_emb (Note: Euler-FM G_theta depends on noise level time 't' or 'r')
        with torch.no_grad():
            # Need r_emb to condition G_theta for the r path start point
            r_emb_group = sinusoidal_embedding(r_group_val, TIME_EMB_DIM)
            G_theta_pred_r_ema = model_ema(model_input_r_group, r_emb_group)


        # Push forward to get samples at time s using Euler-FM step
        # Step from t to s using G_theta_pred_t
        ys_t = apply_euler_fm_step(xt_group, G_theta_pred_t, s_group_val, t_group_val)

        # Step from r to s using G_theta_pred_r_ema
        with torch.no_grad():
            ys_r = apply_euler_fm_step(xr_group, G_theta_pred_r_ema, s_group_val, r_group_val)

        # --- MMD Calculation ---
        term1 = 0.0
        term2 = 0.0
        term3 = 0.0

        # <<< ADDED: Log G_theta stats during training >>>
        with torch.no_grad():
            g_theta_t_norm_group = torch.mean(torch.linalg.norm(G_theta_pred_t.flatten(1), dim=1))
            g_theta_r_norm_group = torch.mean(torch.linalg.norm(G_theta_pred_r_ema.flatten(1), dim=1))
        # Store norms for averaging after the loop
        if 'g_theta_t_norms' not in locals():
             g_theta_t_norms = []
             g_theta_r_norms = []
        g_theta_t_norms.append(g_theta_t_norm_group.item())
        g_theta_r_norms.append(g_theta_r_norm_group.item())
        # <<< END ADDED >>>


        # Get coefficients for time-dependent kernel for this group (using s and t/r)
        _, cout_st = get_euler_fm_coefficients(s_group_val.mean(), t_group_val.mean()) # Approx for group
        _, cout_sr = get_euler_fm_coefficients(s_group_val.mean(), r_group_val.mean()) # Approx for group

        # Efficient MMD calculation
        for j in range(M):
            for k in range(M):
                # Kernel depends on the *step* taken (t->s or r->s)
                term1 += kernel_fn(ys_t[j:j+1], ys_t[k:k+1], cout_st)
                term2 += kernel_fn(ys_r[j:j+1], ys_r[k:k+1], cout_sr)
                # Cross terms - which cout to use? Average? Max? Paper isn't explicit. Using t->s version.
                term3 += kernel_fn(ys_t[j:j+1], ys_r[k:k+1], cout_st)


        mmd_sq = (term1 / (M * M) + term2 / (M * M) - 2 * term3 / (M * M))
        group_loss = torch.relu(mmd_sq) # Ensure non-negative

        # Apply VDM weighting for the group
        weighted_group_loss = weights_group * group_loss
        total_loss += weighted_group_loss
        processed_samples += M

    # Average loss over processed samples (or groups)
    final_loss = total_loss / num_groups if num_groups > 0 else torch.tensor(0.0, device=DEVICE)

    # <<< ADDED: Calculate average norms to return for logging >>>
    avg_g_theta_t_norm = np.mean(g_theta_t_norms) if 'g_theta_t_norms' in locals() and g_theta_t_norms else 0.0
    avg_g_theta_r_norm = np.mean(g_theta_r_norms) if 'g_theta_r_norms' in locals() and g_theta_r_norms else 0.0
    # <<< END ADDED >>>

    return final_loss, avg_g_theta_t_norm, avg_g_theta_r_norm # Return norms


# --- Sampling Function ---
# <<< MODIFIED: Uses Euler-FM steps, saves intermediates, logs G_theta >>>
@torch.no_grad()
def generate_samples(model, num_samples=64, num_steps=8, schedule_fn=get_ot_fm_schedule, save_intermediate_dir=None, save_interval=1):
    """Generate samples using the trained model with Euler-FM pushforward sampling.
       Optionally saves intermediate steps and logs G_theta statistics.
    """
    model.eval()
    # Start from prior N(0, sigma_d^2 * I) at T_MAX
    x_t = torch.randn(num_samples, IMG_CHANNELS, IMG_SIZE, IMG_SIZE, device=DEVICE) * SIGMA_D

    # Define time steps [T_MAX, ..., t1, T_EPSILON] - uniform for simplicity now
    time_steps = torch.linspace(T_MAX, T_EPSILON, num_steps + 1, device=DEVICE)

    # Create intermediate save directory if needed
    if save_intermediate_dir:
        os.makedirs(save_intermediate_dir, exist_ok=True)
        # Save initial noise
        initial_noise_vis = (torch.clamp(x_t, -3*SIGMA_D, 3*SIGMA_D) + 3*SIGMA_D) / (6*SIGMA_D) # Map approx N(0,sd^2) to [0,1] for vis more robustly
        initial_noise_vis = torch.clamp(initial_noise_vis, 0.0, 1.0)
        vutils.save_image(initial_noise_vis, os.path.join(save_intermediate_dir, f"step_000_xt_TMAX.png"), nrow=int(math.sqrt(num_samples)), normalize=False)

    g_theta_norms = []

    for i in range(num_steps):
        t_curr = time_steps[i]
        t_next = time_steps[i+1] # This is 's' in the f_st notation

        # Prepare batch time embeddings for current time t_curr
        t_curr_batch = torch.full((num_samples,), t_curr.item(), device=DEVICE)
        t_curr_emb = sinusoidal_embedding(t_curr_batch, TIME_EMB_DIM)

        # Calculate input conditioning scale c_in(t_curr)
        c_in_t_curr = get_input_conditioning_scale(t_curr_batch)
        while c_in_t_curr.dim() < x_t.dim():
             c_in_t_curr = c_in_t_curr.unsqueeze(-1)

        # Condition the input x_t
        model_input = c_in_t_curr * x_t

        # Model predicts G_theta given conditioned x_t and t_curr embedding
        G_theta_pred = model(model_input, t_curr_emb)

        # <<< ADDED: Log G_theta norm >>>
        current_g_theta_norm = torch.mean(torch.linalg.norm(G_theta_pred.flatten(1), dim=1)).item()
        g_theta_norms.append(current_g_theta_norm)
        # Only print if generating samples (not during FID calc which also uses this fn)
        if save_intermediate_dir and i % max(1, num_steps // 10) == 0: # Log periodically only when saving intermediates
             print(f"  Sampling Step {i+1}/{num_steps}, t={t_curr:.4f} -> {t_next:.4f}, Avg |G_theta|: {current_g_theta_norm:.4f}")
        # <<< END ADDED >>>

        # Push forward using Euler-FM step from t_curr to t_next
        x_t = apply_euler_fm_step(x_t, G_theta_pred, t_next, t_curr) # Pass non-conditioned x_t

        # <<< ADDED: Save intermediate step >>>
        if save_intermediate_dir and (i + 1) % save_interval == 0:
            x_t_vis = (torch.clamp(x_t, -1.0, 1.0) + 1.0) / 2.0 # Map to [0, 1] for visualization
            vutils.save_image(x_t_vis, os.path.join(save_intermediate_dir, f"step_{i+1:03d}_xt_{t_next:.3f}.png"), nrow=int(math.sqrt(num_samples)), normalize=False)
        # <<< END ADDED >>>

    model.train()
    # Final sample mapping remains the same
    samples = torch.clamp(x_t, -1.0, 1.0) # Clamp to expected data range [-1, 1]
    samples = (samples + 1.0) / 2.0      # Map [-1, 1] to [0, 1]
    samples = torch.clamp(samples, 0.0, 1.0) # Final clamp just in case
    return samples, g_theta_norms # Return norms as well

# --- FID Calculation Helper Classes (Moved to Top Level) --- # <-- ADDED SECTION

# <<< ADDED: Wrapper for generated samples >>>
class GeneratedDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, tensor_dataset):
        self.tensor_dataset = tensor_dataset

    def __len__(self):
        return len(self.tensor_dataset)

    def __getitem__(self, idx):
        # TensorDataset returns a tuple (tensor,), we need just the tensor
        return self.tensor_dataset[idx][0]

# <<< ADDED: Wrapper for real dataset for FID >>>
class FIDDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, device):
        self.dataset = dataset
        self.device = device # Store device for unnormalization
        self.needs_unnormalize = False # Flag to indicate if unnormalization is needed
        self.mean = None
        self.std = None

        # Determine if unnormalization is needed based on dataset transforms
        current_transform = self.dataset.transform
        norm_transform = None
        if isinstance(current_transform, transforms.Compose):
            for t in current_transform.transforms:
                if isinstance(t, transforms.Normalize):
                    norm_transform = t
                    break
        elif isinstance(current_transform, transforms.Normalize):
            norm_transform = current_transform

        if norm_transform is not None:
            self.needs_unnormalize = True
            # Ensure mean/std tensors are created on the correct device and store them
            self.mean = torch.tensor(norm_transform.mean, device=self.device).view(-1, 1, 1)
            self.std = torch.tensor(norm_transform.std, device=self.device).view(-1, 1, 1)

    def _unnormalize(self, x):
        """Performs unnormalization using stored mean and std."""
        if not self.needs_unnormalize or self.mean is None or self.std is None:
            return x # Should not happen if called correctly, but safe check
        # Ensure input is on the correct device for the operation
        x_on_device = x.to(self.device)
        # Unnormalize
        x_unnormalized = x_on_device * self.std + self.mean
        # Return result on CPU
        return x_unnormalized.cpu()

    def __len__(self):
        # Limit length for FID calculation if needed, or use full dataset
        # return min(len(self.dataset), num_samples) # Optional: Match num_samples
         return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx] # Get image data

        # Ensure img is a tensor before potentially unnormalizing
        if not isinstance(img, torch.Tensor):
             # Assuming the transform includes ToTensor if it's not already a tensor
             # If not, manual ToTensor might be needed here, but dataset transform should handle it.
             # Let's trust the dataset transform delivers a tensor.
             pass # Assume img is a tensor due to dataset's transform

        # Unnormalize if needed (expects input tensor, moves to device, returns CPU tensor [0, 1])
        if self.needs_unnormalize:
             img = self._unnormalize(img) # Call the instance method
        else:
            # If no normalization, assume data is already [0, 1] or needs clamping
            # Move to CPU just in case it was loaded to GPU by dataloader
            img = img.cpu()
            if img.min() < 0 or img.max() > 1:
                 # Attempt to bring to [0, 1] range if it looks like [-1, 1]
                 if img.min() >= -1.01 and img.max() <= 1.01: # Allow slight tolerance
                     img = (img + 1.0) / 2.0
                 else: # Fallback: clamp
                     img = torch.clamp(img, 0.0, 1.0)

        # Final conversion to uint8 [0, 255] on CPU
        img_uint8 = torch.clamp(img * 255, 0, 255).to(torch.uint8)
        return img_uint8 # Return image as uint8 [0, 255] on CPU

# --- END ADDED SECTION ---


# --- FID Calculation Function --- # <-- Renamed section
@torch.no_grad()
def calculate_fid(generator_model, dataset, device, num_samples=5000, batch_size=128, num_gen_steps=8):
    """Calculates FID between generated samples and the real dataset."""
    print(f"Calculating FID using {num_samples} samples...")
    generator_model.eval() # Ensure model is in eval mode

    # Generate samples
    generated_samples_list = []
    dataloader_batch_size = batch_size # Use the same batch size for generation
    for i in range(0, num_samples, dataloader_batch_size):
        n_batch = min(dataloader_batch_size, num_samples - i)
        if n_batch <= 0: break

        # generate_samples is now defined before this function
        # <<< MODIFIED: Unpack tuple returned by generate_samples >>>
        samples, _ = generate_samples(generator_model, num_samples=n_batch, num_steps=num_gen_steps)
        # <<< END MODIFIED >>>

        # torch-fidelity expects input in range [0, 255] as uint8 on CPU
        samples = (samples * 255).clamp(0, 255).to(torch.uint8)
        generated_samples_list.append(samples.cpu()) # Collect on CPU
        # Small print to show progress
        if (i // dataloader_batch_size) % 10 == 0:
            print(f"  Generated {i+n_batch}/{num_samples} samples...")

    # Consolidate generated samples into a single tensor
    if not generated_samples_list:
        print("Warning: No samples generated for FID calculation.")
        return float('inf') # Return infinity or handle error appropriately
    all_generated_samples = torch.cat(generated_samples_list, dim=0)

    # Wrap generated samples in a TensorDataset <-- ORIGINAL
    # generated_dataset = TensorDataset(all_generated_samples)

    # <<< MODIFIED: Use the new wrapper >>>
    generated_tensor_dataset = TensorDataset(all_generated_samples)
    generated_dataset_wrapper = GeneratedDatasetWrapper(generated_tensor_dataset)
    # <<< END MODIFIED >>>

    # Create the wrapper instance, passing the device
    fid_dataset_wrapper = FIDDatasetWrapper(dataset, device)

    # Use a DataLoader for the real dataset for efficient batching if needed, though
    # torch-fidelity can often handle the dataset object directly more efficiently.
    # We pass the dataset wrapper directly.

    metrics_dict = calculate_metrics(
        input1=generated_dataset_wrapper, # Pass the new wrapper instance <-- MODIFIED
        input2=fid_dataset_wrapper,    # Dataset wrapper outputting uint8 [0, 255] on CPU
        cuda=(device.type == 'cuda'),
        fid=True,
        batch_size=batch_size, # Pass batch size to calculate_metrics
        verbose=False, # Set to True for more detailed logs from torch-fidelity
        # Caching can speed up subsequent runs but uses disk space
        # input1_cache_name=f"fid_gen_epoch_{current_epoch}", # Consider epoch-specific cache name
        input2_cache_name="fid_real_cifar10_uint8",
    )

    generator_model.train() # Set model back to train mode
    print(f"FID calculation finished. FID: {metrics_dict['frechet_inception_distance']:.4f}")
    return metrics_dict['frechet_inception_distance']
# <<< REMOVED redundant section marker >>>
# --- END ADDED SECTION ---

# <<< Added guard for multiprocessing safety >>>
if __name__ == '__main__':
    # <<< END Added guard >>>
    # --- Training Setup ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalize to [-1, 1] - check if model assumes this or [0, 1] based on sigma_d
        # If using sigma_d=0.5, [-1, 1] is roughly correct for std dev scaling
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True, pin_memory=True if DEVICE.type == 'cuda' else False)

    # <<< MODIFIED: Adjusted base_dim >>>
    model = SimpleUNet(img_channels=IMG_CHANNELS, time_emb_dim=TIME_EMB_DIM, base_dim=128).to(DEVICE) # 128 might be more stable
    model_ema = deepcopy(model).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999)) # Use betas from paper? Default is ok.

    # Optional: Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(dataloader), eta_min=1e-6) # Step per iteration


    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Using OT-FM Schedule, Euler-FM Parameterization, Eta Decrement r(t), VDM Weighting, M={M_PARTICLES}")

    # --- Training Loop ---
    # <<< MODIFIED: Uses new loss inputs, steps scheduler per iteration >>>
    losses = []
    fid_scores = [] # <-- ADDED for FID tracking
    fid_epochs = [] # <-- ADDED for FID tracking

    for epoch in range(EPOCHS):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (x_real, _) in enumerate(progress_bar):
            x_real = x_real.to(DEVICE)
            # Map data from [-1, 1] to roughly N(0, sigma_d^2=0.25) if needed
            # Assuming ToTensor() -> [0,1], Normalize -> [-1,1].
            # If data is [-1, 1], std dev is roughly 1/sqrt(3) if uniform?
            # Paper uses sigma_d=0.5. If input is [-1,1], maybe scale by 0.5?
            # x_real = x_real * 0.5 # Optional: Scale input to match assumed sigma_d=0.5 more closely?

            if x_real.shape[0] == 0: continue

            optimizer.zero_grad()

            # <<< MODIFIED: Capture returned norms >>>
            loss, avg_gt_norm, avg_gr_norm = imm_loss(model, model_ema, x_real, M_PARTICLES, get_ot_fm_schedule, r_function, laplace_kernel)
            # <<< END MODIFIED >>>

            if torch.isnan(loss) or torch.isinf(loss):
               print(f"\nNaN/Inf loss detected at Epoch {epoch+1}, Batch {batch_idx}. Skipping update.")
               # Consider saving state or breaking
               optimizer.zero_grad() # Zero grads even if skipping step
               continue

            loss.backward()

            if GRAD_CLIP_NORM > 0:
                # total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                # if total_norm.item() > GRAD_CLIP_NORM * 5:
                #      print(f"\nHigh grad norm: {total_norm.item():.2f} before clipping (clipped to {GRAD_CLIP_NORM})")
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)


            optimizer.step()
            update_ema(model_ema, model, EMA_DECAY)
            scheduler.step() # Step scheduler each iteration

            epoch_loss += loss.item()
            num_batches += 1
            # <<< MODIFIED: Add G_theta norms to postfix >>>
            progress_bar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0], gt_norm=avg_gt_norm, gr_norm=avg_gr_norm)
            # <<< END MODIFIED >>>

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f} LR: {scheduler.get_last_lr()[0]:.6f}")


        # Generate and save samples periodically
        if (epoch + 1) % SAVE_INTERVAL == 0 or epoch == EPOCHS - 1:
            print("Generating samples...")
            num_samples_to_generate = 64 # Keep consistent with plot
            # <<< MODIFIED: Capture G_theta norms and specify intermediate save dir >>>
            intermediate_save_path = os.path.join(RESULTS_DIR, f"epoch_{epoch+1:03d}_intermediate_steps")
            generated_samples, gen_g_theta_norms = generate_samples(
                model_ema,
                num_samples=num_samples_to_generate,
                num_steps=8, # Keep default 8 for standard eval, maybe increase manually for debugging
                save_intermediate_dir=intermediate_save_path,
                save_interval=1 # Save every step
            )
            print(f"Intermediate generation steps saved to: {intermediate_save_path}")
            print(f"Generation G_theta norms: {gen_g_theta_norms}")
            # <<< END MODIFIED >>>

            # --- Save Sample Grid Plot --- <-- Renamed section for clarity
            fig, axes = plt.subplots(8, 8, figsize=(8, 8))
            for i, ax in enumerate(axes.flat):
                if i < generated_samples.shape[0]:
                    sample = generated_samples[i].cpu().permute(1, 2, 0).numpy() # HWC for imshow
                    ax.imshow(sample)
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f"epoch_{epoch+1:03d}_samples_grid.png")) # Added _grid to filename
            plt.close(fig)

            # --- Save Individual Samples --- <-- ADDED SECTION
            individual_samples_dir = os.path.join(RESULTS_DIR, f"epoch_{epoch+1:03d}_individual_samples")
            os.makedirs(individual_samples_dir, exist_ok=True)
            print(f"Saving {num_samples_to_generate} individual samples to {individual_samples_dir}...")
            for i in range(generated_samples.shape[0]):
                vutils.save_image(generated_samples[i], os.path.join(individual_samples_dir, f"sample_{i:03d}.png"), normalize=False) # Already in [0, 1]
            # --- End Individual Sample Saving ---

            # <<< MODIFIED: Uncomment FID Calculation >>>
            # --- Calculate FID ---
            print("Calculating FID...")
            current_fid = calculate_fid(model_ema, dataset, DEVICE, num_samples=5000, batch_size=128, num_gen_steps=8) # Use EMA model
            fid_scores.append(current_fid)
            fid_epochs.append(epoch + 1)
            print(f"FID at epoch {epoch+1}: {current_fid:.4f}")
            # --- End FID Calculation ---
            # <<< END MODIFIED >>>

            # Save model checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'ema_model_state_dict': model_ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_epoch_loss,
                'losses_list': losses,
                # 'fid_scores_list': fid_scores, # <-- ADDED
                # 'fid_epochs_list': fid_epochs, # <-- ADDED
                # <<< MODIFIED: Save FID scores in checkpoint >>>
                'fid_scores_list': fid_scores,
                'fid_epochs_list': fid_epochs,
                # <<< END MODIFIED >>>
            }, os.path.join(RESULTS_DIR, f"checkpoint_epoch_{epoch+1:03d}.pth"))


    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses) + 1), losses)
    plt.title(f"IMM Training Loss on CIFAR-10 (Euler-FM, M={M_PARTICLES})")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "training_loss_curve.png"))
    plt.show()

    # Plot FID curve <-- ADDED
    if fid_scores:
        plt.figure(figsize=(10, 5))
        plt.plot(fid_epochs, fid_scores, marker='o')
        plt.title(f"FID Score during Training on CIFAR-10 (Euler-FM, M={M_PARTICLES})")
        plt.xlabel("Epoch")
        plt.ylabel("FID Score (Lower is Better)")
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, "fid_score_curve.png"))
        plt.show()

    print("Training finished.")