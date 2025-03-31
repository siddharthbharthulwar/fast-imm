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

# --- Hyperparameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

IMG_SIZE = 28
IMG_CHANNELS = 1
LATENT_DIM = 128 # For time embeddings
BATCH_SIZE = 128
M_PARTICLES = 8 # Number of particles for MMD estimation (must divide BATCH_SIZE)
EPOCHS = 250
LEARNING_RATE = 1e-4
EMA_DECAY = 0.999
TIME_EMB_DIM = 128 # Dimension for sinusoidal time embeddings
T_MAX = 1.0 # Max interpolation time
T_EPSILON = 1e-3 # Min interpolation time to avoid numerical issues near 0
R_DELTA = 0.1 # Fixed decrement for r(s, t) = max(s, t - R_DELTA)
KERNEL_BANDWIDTH = 1.0 # Bandwidth for Laplace kernel (can be tuned)
GRAD_CLIP_NORM = 1.0 # Max norm for gradient clipping
SAVE_INTERVAL = 5 # Save samples every N epochs
RESULTS_DIR = "imm_mnist_results"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# --- Helper Functions ---

def sinusoidal_embedding(t, dim):
    """Sinusoidal time embeddings."""
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
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
    """Laplace kernel: exp(-||x - y||_1 / (bandwidth * D))"""
    # Using L1 norm as suggested by some MMD implementations, L2 is also possible
    # Normalize by dimension D as in the paper
    D = x.shape[1] * x.shape[2] * x.shape[3]
    diff_norm = torch.sum(torch.abs(x.view(x.shape[0], -1) - y.view(y.shape[0], -1)), dim=1)
    # Add small epsilon to avoid log(0) or division by zero if x=y
    diff_norm = torch.clamp(diff_norm, min=1e-9)
    return torch.exp(-diff_norm / (bandwidth * D))

def update_ema(ema_model, model, decay):
    """Update Exponential Moving Average model."""
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

# --- Model Architecture (Simple U-Net for MNIST) ---

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x, t_emb):
        h = self.relu(self.conv1(x))
        time_emb = self.relu(self.time_mlp(t_emb))
        # Extend time embedding to spatial dimensions
        h = h + time_emb[:, :, None, None]
        h = self.relu(self.conv2(h))
        return h

class SimpleUNet(nn.Module):
    def __init__(self, img_channels=1, time_emb_dim=128):
        super().__init__()
        self.time_emb_dim = time_emb_dim * 2 # We concat s and t embeddings

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_emb_dim, self.time_emb_dim * 2),
            nn.ReLU(),
            nn.Linear(self.time_emb_dim * 2, self.time_emb_dim)
        )

        # Contracting path
        self.down1 = Block(img_channels, 128, self.time_emb_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = Block(128, 256, self.time_emb_dim)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bot1 = Block(256, 512, self.time_emb_dim)

        # Expansive path
        self.upconv1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up1 = Block(512, 256, self.time_emb_dim)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up2 = Block(256, 128, self.time_emb_dim)

        # Output layer
        self.out = nn.Conv2d(128, img_channels, 1)

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
        # Pad if necessary (due to integer division in pooling)
        diffY = x2.size()[2] - u1.size()[2]
        diffX = x2.size()[3] - u1.size()[3]
        u1 = nn.functional.pad(u1, [diffX // 2, diffX - diffX // 2,
                                     diffY // 2, diffY - diffY // 2])
        c1 = torch.cat([u1, x2], dim=1)
        x3 = self.up1(c1, t)

        u2 = self.upconv2(x3)
        diffY = x1.size()[2] - u2.size()[2]
        diffX = x1.size()[3] - u2.size()[3]
        u2 = nn.functional.pad(u2, [diffX // 2, diffX - diffX // 2,
                                     diffY // 2, diffY - diffY // 2])
        c2 = torch.cat([u2, x1], dim=1)
        x4 = self.up2(c2, t)

        # Output
        out = self.out(x4)
        return out # Predicts clean data x0_pred

# --- IMM Loss Function ---

def imm_loss(model, model_ema, x_batch, M, schedule_fn, kernel_fn, bandwidth):
    """Computes the Inductive Moment Matching loss."""
    B, C, H, W = x_batch.shape
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
    # Note: The paper reuses xt AND x. DDIM(xt, x, r, t) computes the point on the
    # deterministic trajectory from x to xt at time r.
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
        # Select the single s, t, r value for the group (use the first one)
        # In practice, paper samples one (s,t) per group. Here we average embeddings for simplicity.
        # A better way: sample one s,t per group *before* embedding calculation.
        s_group_val = s[start_idx:end_idx] # Keep shape M for broadcast in ddim_sample_step
        t_group_val = t[start_idx:end_idx]
        r_group_val = r[start_idx:end_idx]


        # Predict clean data using current model for t -> s path
        # The model g_theta(xt, s, t) predicts x0 given xt, s, t
        x0_pred_t = model(xt_group, s_emb_group, t_emb_group)

        # Predict clean data using EMA model for r -> s path
        with torch.no_grad():
            x0_pred_r_ema = model_ema(xr_group, s_emb_group, r_emb_group)

        # Push forward to get samples at time s
        # ys,t = f_theta_s,t(xt) = DDIM(xt, x0_pred_t, s, t)
        ys_t = ddim_sample_step(xt_group, x0_pred_t, s_group_val, t_group_val, schedule_fn)

        # ys,r = f_theta-_s,r(xr) = DDIM(xr, x0_pred_r_ema, s, r)
        with torch.no_grad():
            ys_r = ddim_sample_step(xr_group, x0_pred_r_ema, s_group_val, r_group_val, schedule_fn)

        # --- MMD Calculation (V-statistic, unbiased estimate) ---
        # k(ys_t, ys_t) + k(ys_r, ys_r) - 2 * k(ys_t, ys_r)
        term1 = 0.0
        term2 = 0.0
        term3 = 0.0

        # Efficient MMD calculation
        for j in range(M):
            for k in range(M):
                # Diagonal terms are handled implicitly if kernel(x,x)=1 or const
                term1 += kernel_fn(ys_t[j:j+1], ys_t[k:k+1], bandwidth)
                term2 += kernel_fn(ys_r[j:j+1], ys_r[k:k+1], bandwidth)
                term3 += kernel_fn(ys_t[j:j+1], ys_r[k:k+1], bandwidth)

        # Use unbiased MMD estimator (remove diagonal terms j==k, scale)
        # Scaling factor: 1 / (M * (M-1)) ? Paper uses 1/M^2 in Eq 67. Let's use 1/M^2.
        mmd_sq = (term1 / (M * M) + term2 / (M * M) - 2 * term3 / (M * M))

        # Ensure loss is non-negative (numerical precision might cause small negatives)
        group_loss = torch.relu(mmd_sq)
        total_loss += group_loss

    final_loss = total_loss / num_groups
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
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1]
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

model = SimpleUNet(img_channels=IMG_CHANNELS, time_emb_dim=TIME_EMB_DIM).to(DEVICE)
model_ema = deepcopy(model).to(DEVICE) # Exponential Moving Average model
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# --- Training Loop ---
losses = []
for epoch in range(EPOCHS):
    model.train()
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    epoch_loss = 0.0
    num_batches = 0

    for batch_idx, (x_real, _) in enumerate(progress_bar):
        x_real = x_real.to(DEVICE)
        optimizer.zero_grad()

        loss = imm_loss(model, model_ema, x_real, M_PARTICLES, get_linear_schedule, laplace_kernel, KERNEL_BANDWIDTH)

        if torch.isnan(loss):
           print(f"NaN loss detected at Epoch {epoch+1}, Batch {batch_idx}. Skipping update.")
           continue # Skip update if loss is NaN

        loss.backward()

        # Gradient Clipping
        if GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

        optimizer.step()
        update_ema(model_ema, model, EMA_DECAY)

        epoch_loss += loss.item()
        num_batches += 1
        progress_bar.set_postfix(loss=loss.item())

    avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
    losses.append(avg_epoch_loss)
    print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")

    # Generate and save samples periodically
    if (epoch + 1) % SAVE_INTERVAL == 0 or epoch == EPOCHS - 1:
        print("Generating samples...")
        # Use EMA model for generation
        generated_samples = generate_samples(model_ema, num_samples=64, num_steps=8)

        fig, axes = plt.subplots(8, 8, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            if i < generated_samples.shape[0]:
                sample = generated_samples[i].cpu().squeeze()
                ax.imshow(sample, cmap='gray')
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
            'loss': avg_epoch_loss,
        }, os.path.join(RESULTS_DIR, f"checkpoint_epoch_{epoch+1:03d}.pth"))


# Plot loss curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS + 1), losses)
plt.title("IMM Training Loss on MNIST")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "training_loss_curve.png"))
plt.show()

print("Training finished.")