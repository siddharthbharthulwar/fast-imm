import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np # Needed for image permute

# --- Hyperparameters ---
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# <<< MODIFIED FOR CIFAR >>>
IMG_SIZE = 32
IMG_CHANNELS = 3
RESULTS_DIR = "imm_cifar10_results"
EPOCHS = 250 # Increased epochs for harder dataset
# Adjust batch size based on GPU memory with larger model
BATCH_SIZE = 128 # Keep 128 if memory allows, else reduce (e.g., 64)
# <<< END MODIFIED >>>

LATENT_DIM = 128 # For time embeddings
M_PARTICLES = 8 # Number of particles for MMD estimation (must divide BATCH_SIZE)
LEARNING_RATE = 1e-4
EMA_DECAY = 0.999
TIME_EMB_DIM = 128 # Dimension for sinusoidal time embeddings
T_MAX = 1.0 # Max interpolation time
T_EPSILON = 1e-3 # Min interpolation time to avoid numerical issues near 0
R_DELTA = 0.1 # Fixed decrement for r(s, t) = max(s, t - R_DELTA)
KERNEL_BANDWIDTH = 1.0 # Bandwidth for Laplace kernel (can be tuned)
GRAD_CLIP_NORM = 1.0 # Max norm for gradient clipping
SAVE_INTERVAL = 10 # Save samples every N epochs (increased interval)


if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# --- Helper Functions --- (No changes needed here for CIFAR)

def sinusoidal_embedding(t, dim):
    """Sinusoidal time embeddings."""
    # Prevent log(0) error for dim=1
    if dim <= 1:
        return t[:, None] # Or handle differently if needed

    half_dim = dim // 2
    # Ensure denominator is not zero
    denominator = half_dim - 1 if half_dim > 1 else 1
    emb = math.log(10000) / denominator
    emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    if dim % 2 == 1: # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb

def get_linear_schedule(t):
    """Linear interpolation schedule (αt = 1-t, σt = t)."""
    alpha_t = 1.0 - t
    sigma_t = t
    return alpha_t, sigma_t

def ddim_interpolate(xt, x, t, target_t, schedule_fn):
    """DDIM formula to interpolate from xt at time t towards x at target_t < t.
       This is used to compute xr = DDIM(xt, x, r, t) when reusing x.
       Note: This uses the GROUND TRUTH x.
       Formula: DDIM(xt, x, s, t) = (αs - σs/σt * αt) * x + (σs/σt) * xt
    """
    alpha_s, sigma_s = schedule_fn(target_t)
    alpha_t, sigma_t = schedule_fn(t)

    # Avoid division by zero if sigma_t is close to 0
    sigma_t = torch.clamp(sigma_t, min=1e-6)

    term1_coeff = alpha_s - (sigma_s / sigma_t) * alpha_t
    term2_coeff = sigma_s / sigma_t

    target_x = term1_coeff[:, None, None, None] * x + term2_coeff[:, None, None, None] * xt
    return target_x


def ddim_sample_step(xt, x_pred, s, t, schedule_fn):
    """DDIM step: pushes xt at time t to xs at time s using the *predicted* x_pred.
       Formula: f_theta_s_t(xt) = DDIM(xt, x_pred, s, t)
                 = (αs - σs/σt * αt) * x_pred + (σs/σt) * xt
    """
    alpha_s, sigma_s = schedule_fn(s)
    alpha_t, sigma_t = schedule_fn(t)

    # Avoid division by zero
    sigma_t = torch.clamp(sigma_t, min=1e-6)

    term1_coeff = alpha_s - (sigma_s / sigma_t) * alpha_t
    term2_coeff = sigma_s / sigma_t

    xs = term1_coeff[:, None, None, None] * x_pred + term2_coeff[:, None, None, None] * xt
    return xs


def laplace_kernel(x, y, bandwidth):
    """Laplace kernel: exp(-||x - y||_1 / (max(bandwidth * D, 1e-9)))"""
    # Using L1 norm as suggested by some MMD implementations, L2 is also possible
    # Normalize by dimension D as in the paper
    if x.ndim < 2 or y.ndim < 2: # Handle potential scalar inputs if they occur
         # Return a tensor consistent with expected output type
         return torch.tensor(1.0, device=x.device, dtype=x.dtype if torch.is_tensor(x) else torch.float32)

    # Calculate dimension D based on input shape
    if x.ndim == 4: # Assuming (B, C, H, W)
        D = x.shape[1] * x.shape[2] * x.shape[3]
    elif x.ndim == 2: # Assuming (B, Features)
        D = x.shape[1]
    else:
        # Fallback or raise error for unexpected dimensions
        D = np.prod(x.shape[1:]) # Product of all dimensions except batch

    if D == 0: # Avoid division by zero if dimensions are somehow zero
        return torch.ones_like(x.reshape(x.shape[0], -1).sum(dim=1)) # Return tensor of 1s with correct batch size

    diff_norm = torch.sum(torch.abs(x.reshape(x.shape[0], -1) - y.reshape(y.shape[0], -1)), dim=1)
    # Add small epsilon to avoid issues if x=y
    diff_norm = torch.clamp(diff_norm, min=1e-9)

    # --- MODIFIED LINE ---
    # Calculate the denominator value and ensure it's not too small using Python's max
    bw_d_value = bandwidth * float(D) # Ensure D is float for multiplication
    safe_denominator = max(bw_d_value, 1e-9)
    # --- END MODIFIED LINE ---

    return torch.exp(-diff_norm / safe_denominator) # Divide tensor by the safe float value

def update_ema(ema_model, model, decay):
    """Update Exponential Moving Average model."""
    with torch.no_grad():
        model_params = dict(model.named_parameters())
        ema_params = dict(ema_model.named_parameters())

        for name, param in model_params.items():
            if name in ema_params:
                 ema_params[name].data.mul_(decay).add_(param.data, alpha=1 - decay)


# --- Model Architecture (Simple U-Net adjusted for CIFAR) ---

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, use_groupnorm=True): # Added GroupNorm option
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        # <<< MODIFIED: Added GroupNorm and changed Activation >>>
        self.norm1 = nn.GroupNorm(8, out_ch) if use_groupnorm else nn.Identity() # 8 groups, adjust if needed
        self.act1 = nn.SiLU() # Switched to SiLU/Swish
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch) if use_groupnorm else nn.Identity()
        self.act2 = nn.SiLU()
        # <<< END MODIFIED >>>

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h) # Apply norm before act and time emb
        time_emb_proj = self.time_mlp(self.act1(t_emb)) # Project time embedding
        # Extend time embedding to spatial dimensions
        h = h + time_emb_proj[:, :, None, None]
        h = self.act1(h) # Apply activation

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)
        return h

class SimpleUNet(nn.Module):
    # <<< MODIFIED FOR CIFAR >>>
    def __init__(self, img_channels=3, time_emb_dim=128, base_dim=128): # Base dim controls width
        super().__init__()
        self.time_emb_dim = time_emb_dim * 2 # We concat s and t embeddings

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_emb_dim, self.time_emb_dim * 4), # Increased complexity
            nn.SiLU(),
            nn.Linear(self.time_emb_dim * 4, self.time_emb_dim)
        )

        # Contracting path (Increased channel width)
        self.down1 = Block(img_channels, base_dim, self.time_emb_dim) # 3 -> 128
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = Block(base_dim, base_dim * 2, self.time_emb_dim) # 128 -> 256
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bot1 = Block(base_dim * 2, base_dim * 4, self.time_emb_dim) # 256 -> 512

        # Expansive path
        self.upconv1 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, 2, stride=2) # 512 -> 256
        # Input channels = skip channels + upconv channels
        self.up1 = Block(base_dim * 4, base_dim * 2, self.time_emb_dim) # 256 (skip) + 256 (upconv) -> 256
        self.upconv2 = nn.ConvTranspose2d(base_dim * 2, base_dim, 2, stride=2) # 256 -> 128
        self.up2 = Block(base_dim * 2, base_dim, self.time_emb_dim) # 128 (skip) + 128 (upconv) -> 128

        # Output layer
        self.out = nn.Conv2d(base_dim, img_channels, 1) # 128 -> 3
    # <<< END MODIFIED >>>

    def forward(self, x, t_emb_s, t_emb_t):
        # Combine and project time embeddings
        t_emb_combined = torch.cat([t_emb_s, t_emb_t], dim=-1)
        t = self.time_mlp(t_emb_combined)

        # Contracting path
        x1 = self.down1(x, t)
        p1 = self.pool1(x1)
        x2 = self.down2(p1, t)
        p2 = self.pool2(x2)

        # Bottleneck
        b = self.bot1(p2, t)

        # Expansive path
        u1 = self.upconv1(b)
        # Skip connection: concat u1 and x2
        c1 = torch.cat([u1, x2], dim=1) # Check dimensions: u1=(B,256,16,16), x2=(B,256,16,16) -> c1=(B,512,16,16)
        x3 = self.up1(c1, t) # up1 expects 512 in_ch

        u2 = self.upconv2(x3)
        # Skip connection: concat u2 and x1
        c2 = torch.cat([u2, x1], dim=1) # Check dimensions: u2=(B,128,32,32), x1=(B,128,32,32) -> c2=(B,256,32,32)
        x4 = self.up2(c2, t) # up2 expects 256 in_ch

        # Output
        out = self.out(x4)
        return out # Predicts clean data x0_pred

# --- IMM Loss Function --- (No changes needed here)

def imm_loss(model, model_ema, x_batch, M, schedule_fn, kernel_fn, bandwidth):
    """Computes the Inductive Moment Matching loss."""
    B, C, H, W = x_batch.shape
    if B == 0: return torch.tensor(0.0, device=DEVICE, requires_grad=True) # Handle empty batch
    assert B % M == 0, f"Batch size {B} must be divisible by particle count {M}"
    num_groups = B // M

    # Sample times t, s for the entire batch
    t = torch.rand(B, device=DEVICE) * (T_MAX - T_EPSILON) + T_EPSILON # Sample t in [epsilon, T]
    s = torch.rand(B, device=DEVICE) * (t - T_EPSILON) + T_EPSILON     # Sample s in [epsilon, t]

    # Calculate r = max(s, t - delta)
    r = torch.max(s, t - R_DELTA)

    # Get time embeddings
    t_emb = sinusoidal_embedding(t, TIME_EMB_DIM)
    s_emb = sinusoidal_embedding(s, TIME_EMB_DIM)
    r_emb = sinusoidal_embedding(r, TIME_EMB_DIM)

    # Calculate xt ~ N(alpha_t * x, sigma_t^2 * I)
    eps = torch.randn_like(x_batch)
    alpha_t, sigma_t = schedule_fn(t)
    xt = alpha_t[:, None, None, None] * x_batch + sigma_t[:, None, None, None] * eps

    # Calculate xr = DDIM(xt, x, r, t) by reusing ground truth x
    with torch.no_grad(): # xr calculation shouldn't require gradients wrt model
        xr = ddim_interpolate(xt, x_batch, t, r, schedule_fn)

    total_loss = 0.0

    # Process in groups of M
    for i in range(num_groups):
        start_idx = i * M
        end_idx = (i + 1) * M

        # Get group data
        xt_group = xt[start_idx:end_idx]
        xr_group = xr[start_idx:end_idx]
        t_emb_group = t_emb[start_idx:end_idx]
        s_emb_group = s_emb[start_idx:end_idx]
        r_emb_group = r_emb[start_idx:end_idx]
        s_group_val = s[start_idx:end_idx] # Keep shape M for broadcast in ddim_sample_step
        t_group_val = t[start_idx:end_idx]
        r_group_val = r[start_idx:end_idx]


        # Predict clean data using current model for t -> s path
        x0_pred_t = model(xt_group, s_emb_group, t_emb_group)

        # Predict clean data using EMA model for r -> s path
        with torch.no_grad():
            x0_pred_r_ema = model_ema(xr_group, s_emb_group, r_emb_group)

        # Push forward to get samples at time s
        ys_t = ddim_sample_step(xt_group, x0_pred_t, s_group_val, t_group_val, schedule_fn)

        # Push forward using EMA model prediction for target path
        with torch.no_grad():
            ys_r = ddim_sample_step(xr_group, x0_pred_r_ema, s_group_val, r_group_val, schedule_fn)

        # --- MMD Calculation ---
        term1 = 0.0
        term2 = 0.0
        term3 = 0.0

        # Efficient MMD calculation (Could be vectorized further if needed)
        for j in range(M):
            for k in range(M):
                term1 += kernel_fn(ys_t[j:j+1], ys_t[k:k+1], bandwidth)
                term2 += kernel_fn(ys_r[j:j+1], ys_r[k:k+1], bandwidth)
                term3 += kernel_fn(ys_t[j:j+1], ys_r[k:k+1], bandwidth)

        # Paper uses 1/M^2 in Eq 67.
        mmd_sq = (term1 / (M * M) + term2 / (M * M) - 2 * term3 / (M * M))

        # Ensure loss is non-negative
        group_loss = torch.relu(mmd_sq)
        total_loss += group_loss

    final_loss = total_loss / num_groups if num_groups > 0 else torch.tensor(0.0, device=DEVICE)
    return final_loss


# --- Sampling Function ---
@torch.no_grad()
def generate_samples(model, num_samples=64, num_steps=8, schedule_fn=get_linear_schedule):
    """Generate samples using the trained model with pushforward sampling."""
    model.eval()
    x_t = torch.randn(num_samples, IMG_CHANNELS, IMG_SIZE, IMG_SIZE, device=DEVICE) # Start from prior N(0,I)

    # Define time steps [T, ..., t1, eps]
    time_steps = torch.linspace(T_MAX, T_EPSILON, num_steps + 1, device=DEVICE)

    for i in range(num_steps):
        t_curr = time_steps[i]
        t_next = time_steps[i+1]

        # Prepare batch time embeddings
        t_curr_batch = torch.full((num_samples,), t_curr, device=DEVICE)
        t_next_batch = torch.full((num_samples,), t_next, device=DEVICE)
        t_curr_emb = sinusoidal_embedding(t_curr_batch, TIME_EMB_DIM)
        t_next_emb = sinusoidal_embedding(t_next_batch, TIME_EMB_DIM)

        # Model predicts x0 given xt, s, t. Here s=t_next, t=t_curr
        x0_pred = model(x_t, t_next_emb, t_curr_emb)

        # Push forward using DDIM step
        x_t = ddim_sample_step(x_t, x0_pred, t_next_batch, t_curr_batch, schedule_fn)

    model.train()
    # Clamp and denormalize samples (assuming input was normalized to [-1, 1])
    samples = torch.clamp(x_t, -1.0, 1.0)
    samples = (samples + 1.0) / 2.0  # Map from [-1, 1] to [0, 1]
    return samples


# --- Training Setup ---
# <<< MODIFIED FOR CIFAR >>>
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to [-1, 1] for 3 channels
])

dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# <<< END MODIFIED >>>
# Consider adding more workers if I/O is a bottleneck
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True, pin_memory=True if DEVICE.type == 'cuda' else False)


# <<< MODIFIED FOR CIFAR >>>
model = SimpleUNet(img_channels=IMG_CHANNELS, time_emb_dim=TIME_EMB_DIM, base_dim=256).to(DEVICE)
# <<< END MODIFIED >>>
model_ema = deepcopy(model).to(DEVICE) # Exponential Moving Average model
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Optional: Learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)


print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# --- Training Loop ---
losses = []
for epoch in range(EPOCHS):
    model.train()
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    epoch_loss = 0.0
    num_batches = 0

    for batch_idx, (x_real, _) in enumerate(progress_bar):
        x_real = x_real.to(DEVICE)
        if x_real.shape[0] == 0: continue # Skip empty batches if drop_last=False

        optimizer.zero_grad()

        loss = imm_loss(model, model_ema, x_real, M_PARTICLES, get_linear_schedule, laplace_kernel, KERNEL_BANDWIDTH)

        if torch.isnan(loss) or torch.isinf(loss):
           print(f"\nNaN/Inf loss detected at Epoch {epoch+1}, Batch {batch_idx}. Skipping update.")
           # Optional: Add debugging info here, e.g., print norms of weights/gradients
           continue # Skip update if loss is NaN/Inf

        loss.backward()

        # Gradient Clipping
        if GRAD_CLIP_NORM > 0:
            # Optional: Log grad norm before clipping
            # total_norm = 0
            # for p in model.parameters():
            #     if p.grad is not None:
            #         param_norm = p.grad.data.norm(2)
            #         total_norm += param_norm.item() ** 2
            # total_norm = total_norm ** 0.5
            # if total_norm > GRAD_CLIP_NORM * 5: # Log if norm is excessively high
            #      print(f"\nHigh grad norm: {total_norm:.2f} before clipping")

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

        optimizer.step()
        update_ema(model_ema, model, EMA_DECAY)

        epoch_loss += loss.item()
        num_batches += 1
        progress_bar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0]) # Show LR

    avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
    losses.append(avg_epoch_loss)
    print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f} LR: {scheduler.get_last_lr()[0]:.6f}")

    scheduler.step() # Step the scheduler each epoch

    # Generate and save samples periodically
    if (epoch + 1) % SAVE_INTERVAL == 0 or epoch == EPOCHS - 1:
        print("Generating samples...")
        # Use EMA model for generation
        generated_samples = generate_samples(model_ema, num_samples=64, num_steps=8) # Using 8 steps for sampling

        fig, axes = plt.subplots(8, 8, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            if i < generated_samples.shape[0]:
                # <<< MODIFIED FOR CIFAR >>>
                sample = generated_samples[i].cpu().permute(1, 2, 0).numpy() # Change dimension order for imshow
                # <<< END MODIFIED >>>
                ax.imshow(sample)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"epoch_{epoch+1:03d}_samples.png"))
        plt.close(fig)

        # Save model checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'ema_model_state_dict': model_ema.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_epoch_loss,
        }, os.path.join(RESULTS_DIR, f"checkpoint_epoch_{epoch+1:03d}.pth"))


# Plot loss curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS + 1), losses)
plt.title("IMM Training Loss on CIFAR-10")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "training_loss_curve.png"))
plt.show()

print("Training finished.")