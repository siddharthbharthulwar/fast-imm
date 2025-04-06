# q_imm_mnist.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy

# --- Hyperparameters ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

IMG_SIZE = 32
IMG_CHANNELS = 3
TIME_EMB_DIM = 128
BATCH_SIZE = 128
M_PARTICLES = 4
EPOCHS = 250  # Increased for harder dataset
LEARNING_RATE = 1e-4
EMA_DECAY = 0.999
T_MAX = 1.0
T_EPSILON = 1e-3
R_DELTA = 0.1
KERNEL_BANDWIDTH = 1.0
GRAD_CLIP_NORM = 1.0
SAVE_INTERVAL = 10
RESULTS_DIR = "q_imm_cifar_results"

# --- Quantization Hyperparameters ---
W_BITS = 8
A_BITS = 8

# --- NeM Hyperparameter ---
LAMBDA_NEM = 0.1

# --- Directory Setup ---
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
    print(f"Created results directory: {RESULTS_DIR}")

# ==============================================================================
# 1. Quantization Components (TaQ, QuantW, QConv2d, QLinear)
# ==============================================================================

# --- Helper: Sinusoidal Embedding ---
def sinusoidal_embedding(t, dim):
    """Sinusoidal time embeddings."""
    if not isinstance(t, torch.Tensor):
        t = torch.tensor([t], dtype=torch.float32, device=DEVICE)
    if t.ndim == 0:
        t = t.unsqueeze(0).to(DEVICE)
    if t.device != DEVICE:
        t = t.to(DEVICE)

    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=DEVICE) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    if dim % 2 == 1: # zero pad
        emb = F.pad(emb, (0,1,0,0))
    return emb

# --- Quantization Core Logic ---
class RoundSTE(Function):
    """Straight-Through Estimator for Rounding."""
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

round_ste = RoundSTE.apply

def quantize_uniform(x, scale, zero_point, num_bits, symmetric=True):
    """Uniform quantization function."""
    qmin = -(2**(num_bits - 1)) if symmetric else 0
    qmax = 2**(num_bits - 1) - 1 if symmetric else 2**num_bits - 1

    scale = torch.abs(scale) + 1e-8 # Ensure positive and stable

    x_q = round_ste(x / scale + zero_point)
    x_q_clipped = torch.clamp(x_q, qmin, qmax)
    x_dq = (x_q_clipped - zero_point) * scale
    return x_dq

# --- Timestep-Aware Activation Quantization (TaQ) ---
class TimestepScale(nn.Module):
    """Small MLP to compute scale based on time embedding."""
    def __init__(self, time_emb_dim, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, t_emb):
        return F.softplus(self.mlp(t_emb)) + 1e-6 # Ensure positive scale

class QuantAct(nn.Module):
    """Quantized Activation Layer with Timestep-Aware Scaling (TaQ)."""
    def __init__(self, num_bits, time_emb_dim, scale_mlp_hidden_dim=32):
        super().__init__()
        if num_bits <= 0: # Allow disabling quantization for debugging
             print(f"Activation quantization disabled (bits={num_bits})")
        self.num_bits = num_bits
        self.symmetric = True
        self.zero_point = 0.0
        # Only create scale function if quantizing
        self.time_scale_fn = TimestepScale(time_emb_dim, scale_mlp_hidden_dim) if self.num_bits > 0 else None

    def forward(self, x, t_emb):
        if self.num_bits <= 0 or self.time_scale_fn is None:
            return x

        # Ensure t_emb is on the correct device
        if t_emb.device != x.device:
             t_emb = t_emb.to(x.device)

        scale = self.time_scale_fn(t_emb)

        if x.dim() == 4:
            scale = scale.view(-1, 1, 1, 1)
        elif x.dim() == 2:
             scale = scale.view(-1, 1)
        elif x.dim() == 3: # Handle potential cases like [Batch, Seq, Features]
             scale = scale.view(-1, 1, 1)
        # Add other dimensions if needed

        return quantize_uniform(x, scale, self.zero_point, self.num_bits, self.symmetric)

# --- Weight Quantization ---
class QuantW(nn.Module):
    """Weight Quantization Wrapper."""
    def __init__(self, num_bits):
        super().__init__()
        self.num_bits = num_bits
        self.symmetric = True
        self.zero_point = 0.0

    def forward(self, weight, scale):
        if self.num_bits <= 0:
            return weight
        return quantize_uniform(weight, scale, self.zero_point, self.num_bits, self.symmetric)

# --- Quantized Layers ---
class QConv2d(nn.Conv2d):
    """Quantized Conv2d Layer."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 w_bits=4, a_bits=4, time_emb_dim=128):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias)
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.time_emb_dim = time_emb_dim # Store for activation quantizer

        # Activation quantizer (needs time embedding dimension)
        self.act_quant = QuantAct(a_bits, time_emb_dim)

        # Weight quantizer setup
        self.weight_quant = QuantW(w_bits)
        self.weight_scale = nn.Parameter(torch.ones(out_channels, 1, 1, 1))
        # Initialize scale reasonably - performed later in model init potentially

    def forward(self, x, t_emb): # Requires time embedding now
        q_x = self.act_quant(x, t_emb)
        q_weight = self.weight_quant(self.weight, self.weight_scale)
        output = F.conv2d(q_x, q_weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
        return output

class QLinear(nn.Linear):
    """Quantized Linear Layer."""
    def __init__(self, in_features, out_features, bias=True,
                 w_bits=4, a_bits=4, time_emb_dim=128):
        super().__init__(in_features, out_features, bias)
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.time_emb_dim = time_emb_dim

        self.act_quant = QuantAct(a_bits, time_emb_dim)
        self.weight_quant = QuantW(w_bits)
        self.weight_scale = nn.Parameter(torch.tensor(1.0)) # Per-tensor scale for Linear
        # Initialize scale reasonably - performed later in model init potentially

    def forward(self, x, t_emb): # Requires time embedding now
        q_x = self.act_quant(x, t_emb)
        q_weight = self.weight_quant(self.weight, self.weight_scale)
        output = F.linear(q_x, q_weight, self.bias)
        return output

# ==============================================================================
# 2. Quantized Model Architecture (QSimpleUNet)
# ==============================================================================

class QBlock(nn.Module):
    """Quantized Block with GroupNorm."""
    def __init__(self, in_ch, out_ch, time_emb_dim_model, time_emb_dim_quant, w_bits=4, a_bits=4):
        super().__init__()
        self.time_mlp = nn.Sequential(
            QLinear(time_emb_dim_model, out_ch, w_bits=w_bits, a_bits=a_bits, time_emb_dim=time_emb_dim_quant),
            nn.SiLU(),  # Changed to SiLU activation
        )

        self.conv1 = QConv2d(in_ch, out_ch, 3, padding=1, w_bits=w_bits, a_bits=a_bits, time_emb_dim=time_emb_dim_quant)
        self.norm1 = nn.GroupNorm(8, out_ch)  # Added GroupNorm
        self.act1 = nn.SiLU()  # Changed to SiLU
        self.act_quant1 = QuantAct(a_bits, time_emb_dim_quant)

        self.conv2 = QConv2d(out_ch, out_ch, 3, padding=1, w_bits=w_bits, a_bits=a_bits, time_emb_dim=time_emb_dim_quant)
        self.norm2 = nn.GroupNorm(8, out_ch)  # Added GroupNorm
        self.act2 = nn.SiLU()  # Changed to SiLU
        self.act_quant2 = QuantAct(a_bits, time_emb_dim_quant)

    def forward(self, x, t_emb_proj, t_emb_quant):
        # Time projection path
        time_emb = self.time_mlp[0](t_emb_proj, t_emb_quant)
        time_emb = self.time_mlp[1](time_emb)

        # Main path
        h = self.conv1(x, t_emb_quant)
        h = self.norm1(h)
        h = self.act1(h)
        h = self.act_quant1(h, t_emb_quant)

        # Add time embedding
        h = h + time_emb[:, :, None, None]

        h = self.conv2(h, t_emb_quant)
        h = self.norm2(h)
        h = self.act2(h)
        h = self.act_quant2(h, t_emb_quant)

        return h

class QSimpleUNet(nn.Module):
    """Quantized UNet with increased capacity for CIFAR-10."""
    def __init__(self, img_channels=3, time_emb_dim=128, base_dim=256, w_bits=4, a_bits=4):
        super().__init__()
        self.time_emb_dim_quant = time_emb_dim
        self.time_emb_dim_model = time_emb_dim * 2

        self.w_bits = w_bits
        self.a_bits = a_bits

        # Time embedding projection MLP (Quantized)
        self.time_mlp = nn.Sequential(
            QLinear(self.time_emb_dim_model, self.time_emb_dim_model * 4, w_bits=w_bits, a_bits=a_bits, time_emb_dim=self.time_emb_dim_quant),
            nn.SiLU(),
            QLinear(self.time_emb_dim_model * 4, self.time_emb_dim_model, w_bits=w_bits, a_bits=a_bits, time_emb_dim=self.time_emb_dim_quant),
            nn.SiLU()
        )

        # Contracting path (increased capacity)
        self.down1 = QBlock(img_channels, base_dim, self.time_emb_dim_model, self.time_emb_dim_quant, w_bits, a_bits)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = QBlock(base_dim, base_dim * 2, self.time_emb_dim_model, self.time_emb_dim_quant, w_bits, a_bits)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bot1 = QBlock(base_dim * 2, base_dim * 4, self.time_emb_dim_model, self.time_emb_dim_quant, w_bits, a_bits)

        # Expansive path
        self.upconv1 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, 2, stride=2)
        self.act_quant_up1_conv = QuantAct(a_bits, self.time_emb_dim_quant)
        self.act_quant_up1_skip = QuantAct(a_bits, self.time_emb_dim_quant)
        self.up1 = QBlock(base_dim * 4, base_dim * 2, self.time_emb_dim_model, self.time_emb_dim_quant, w_bits, a_bits)

        self.upconv2 = nn.ConvTranspose2d(base_dim * 2, base_dim, 2, stride=2)
        self.act_quant_up2_conv = QuantAct(a_bits, self.time_emb_dim_quant)
        self.act_quant_up2_skip = QuantAct(a_bits, self.time_emb_dim_quant)
        self.up2 = QBlock(base_dim * 2, base_dim, self.time_emb_dim_model, self.time_emb_dim_quant, w_bits, a_bits)

        # Output layer
        self.out = QConv2d(base_dim, img_channels, 1, w_bits=w_bits, a_bits=a_bits, time_emb_dim=self.time_emb_dim_quant)

        # Initialize scales
        self._initialize_scales()

    def _initialize_scales(self):
        """Initialize quantization scales based on weight statistics."""
        for m in self.modules():
            if isinstance(m, (QConv2d, QLinear)):
                if hasattr(m, 'weight_scale'):
                    if isinstance(m, QConv2d):
                        # Per-channel initialization
                        std_dev = torch.std(m.weight.data.view(m.out_channels, -1), dim=1) + 1e-4
                        m.weight_scale.data = std_dev.view(-1, 1, 1, 1)
                    elif isinstance(m, QLinear):
                        # Per-tensor initialization
                        m.weight_scale.data.fill_(m.weight.data.std().item() + 1e-4)

    def forward(self, x, t_emb_s, t_emb_t):
        # Use t_emb_t as reference for quantization scales
        t_emb_quant_ref = t_emb_t

        # Combine and project time embeddings
        t_emb_combined = torch.cat([t_emb_s, t_emb_t], dim=-1)

        t_proj = self.time_mlp[0](t_emb_combined, t_emb_quant_ref)
        t_proj = self.time_mlp[1](t_proj)
        t_proj = self.time_mlp[2](t_proj, t_emb_quant_ref)
        t_proj = self.time_mlp[3](t_proj)

        # Contracting path
        x1 = self.down1(x, t_proj, t_emb_quant_ref)
        p1 = self.pool1(x1)
        x2 = self.down2(p1, t_proj, t_emb_quant_ref)
        p2 = self.pool2(x2)

        # Bottleneck
        b = self.bot1(p2, t_proj, t_emb_quant_ref)

        # Expansive path
        u1_fp = self.upconv1(b)
        u1 = self.act_quant_up1_conv(u1_fp, t_emb_quant_ref)
        x2_q = self.act_quant_up1_skip(x2, t_emb_quant_ref)

        diffY = x2_q.size()[2] - u1.size()[2]
        diffX = x2_q.size()[3] - u1.size()[3]
        u1 = F.pad(u1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        c1 = torch.cat([u1, x2_q], dim=1)
        x3 = self.up1(c1, t_proj, t_emb_quant_ref)

        u2_fp = self.upconv2(x3)
        u2 = self.act_quant_up2_conv(u2_fp, t_emb_quant_ref)
        x1_q = self.act_quant_up2_skip(x1, t_emb_quant_ref)

        diffY = x1_q.size()[2] - u2.size()[2]
        diffX = x1_q.size()[3] - u2.size()[3]
        u2 = F.pad(u2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        c2 = torch.cat([u2, x1_q], dim=1)
        x4 = self.up2(c2, t_proj, t_emb_quant_ref)

        # Output
        out = self.out(x4, t_emb_quant_ref)
        return out

# ==============================================================================
# 3. Helper Functions (DDIM, Schedule, Kernel, EMA)
# ==============================================================================

def get_linear_schedule(t):
    """Linear interpolation schedule (αt = 1-t, σt = t)."""
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, device=DEVICE, dtype=torch.float32)
    elif t.device != DEVICE:
        t = t.to(DEVICE)
    t = t.float() # Ensure float type
    alpha_t = 1.0 - t
    sigma_t = t
    return alpha_t, sigma_t

def ddim_interpolate(xt, x, t, target_t, schedule_fn):
    """DDIM formula to interpolate from xt at time t towards x at target_t < t."""
    alpha_s, sigma_s = schedule_fn(target_t)
    alpha_t, sigma_t = schedule_fn(t)
    sigma_t = torch.clamp(sigma_t, min=1e-6) # Avoid division by zero

    # Ensure correct device and shape for broadcasting
    alpha_s, sigma_s = alpha_s.to(xt.device), sigma_s.to(xt.device)
    alpha_t, sigma_t = alpha_t.to(xt.device), sigma_t.to(xt.device)

    term1_coeff = (alpha_s - (sigma_s / sigma_t) * alpha_t).view(-1, 1, 1, 1)
    term2_coeff = (sigma_s / sigma_t).view(-1, 1, 1, 1)

    target_x = term1_coeff * x + term2_coeff * xt
    return target_x

def ddim_sample_step(xt, x_pred, s, t, schedule_fn):
    """DDIM step: pushes xt at time t to xs at time s using the *predicted* x_pred."""
    alpha_s, sigma_s = schedule_fn(s)
    alpha_t, sigma_t = schedule_fn(t)
    sigma_t = torch.clamp(sigma_t, min=1e-6)

    # Ensure correct device and shape for broadcasting
    alpha_s, sigma_s = alpha_s.to(xt.device), sigma_s.to(xt.device)
    alpha_t, sigma_t = alpha_t.to(xt.device), sigma_t.to(xt.device)

    term1_coeff = (alpha_s - (sigma_s / sigma_t) * alpha_t).view(-1, 1, 1, 1)
    term2_coeff = (sigma_s / sigma_t).view(-1, 1, 1, 1)

    xs = term1_coeff * x_pred + term2_coeff * xt
    return xs

def laplace_kernel(x, y, bandwidth):
    """Laplace kernel: exp(-||x - y||_1 / (bandwidth * D))"""
    if x.shape[0] == 0 or y.shape[0] == 0: return torch.tensor(0.0, device=x.device)
    D = float(x.shape[1] * x.shape[2] * x.shape[3]) # Ensure D is float
    diff_norm = torch.sum(torch.abs(x.view(x.shape[0], -1) - y.view(y.shape[0], -1)), dim=1)
    # Clamp norm and ensure denominator is positive and non-zero
    diff_norm = torch.clamp(diff_norm, min=1e-9)
    denominator = (bandwidth * D) + 1e-9
    return torch.exp(-diff_norm / denominator)

def update_ema(ema_model, model, decay):
    """Update Exponential Moving Average model state dict."""
    with torch.no_grad():
        msd = model.state_dict()
        emsd = ema_model.state_dict()
        for k in msd:
            if emsd.get(k) is not None: # Ensure key exists in EMA model
                 emsd[k].mul_(decay).add_(msd[k].data, alpha=1 - decay)
            else:
                 print(f"Warning: Key {k} not found in EMA model state_dict.")


# ==============================================================================
# 4. Loss Function (IMM + NeM)
# ==============================================================================

def imm_loss_nem(model, model_ema, x_batch, M, schedule_fn, kernel_fn, bandwidth, lambda_nem, time_emb_dim_quant):
    """Computes the Inductive Moment Matching loss with NeM for quantized models."""
    B_orig, C, H, W = x_batch.shape
    # Ensure batch size is divisible by M
    if B_orig % M != 0:
        # Simplest fix: drop last samples to make it divisible
        new_B = (B_orig // M) * M
        if new_B == 0:
             print("Warning: Batch size too small for M particles, skipping batch.")
             return torch.tensor(0.0, device=DEVICE, requires_grad=True), \
                    torch.tensor(0.0, device=DEVICE), \
                    torch.tensor(0.0, device=DEVICE)
        x_batch = x_batch[:new_B]
        print(f"Warning: Batch size {B_orig} not divisible by M={M}. Using B={new_B}.")
    B = x_batch.shape[0]
    num_groups = B // M

    # Sample times t, s
    t = torch.rand(B, device=DEVICE) * (T_MAX - T_EPSILON) + T_EPSILON
    s = torch.rand(B, device=DEVICE) * (t - T_EPSILON) + T_EPSILON
    r = torch.max(s, t - R_DELTA)

    # Get time embeddings (original dimension for quantizers)
    t_emb = sinusoidal_embedding(t, time_emb_dim_quant)
    s_emb = sinusoidal_embedding(s, time_emb_dim_quant)
    r_emb = sinusoidal_embedding(r, time_emb_dim_quant)

    # Calculate xt
    eps_gt = torch.randn_like(x_batch)
    alpha_t, sigma_t = schedule_fn(t)
    sigma_t_clamped = torch.clamp(sigma_t, min=1e-6)
    xt = alpha_t.view(-1,1,1,1) * x_batch + sigma_t_clamped.view(-1,1,1,1) * eps_gt

    # Calculate xr using ground truth x
    with torch.no_grad():
        xr = ddim_interpolate(xt, x_batch, t, r, schedule_fn)

    total_mmd_loss = 0.0
    total_nem_loss = 0.0

    for i in range(num_groups):
        start_idx = i * M
        end_idx = (i + 1) * M

        # Group data slices
        xt_group = xt[start_idx:end_idx]
        xr_group = xr[start_idx:end_idx]
        t_emb_group = t_emb[start_idx:end_idx]
        s_emb_group = s_emb[start_idx:end_idx]
        r_emb_group = r_emb[start_idx:end_idx]
        s_group_val = s[start_idx:end_idx]
        t_group_val = t[start_idx:end_idx]
        r_group_val = r[start_idx:end_idx]
        # Get schedule values for the group needed for NeM
        alpha_t_group = alpha_t[start_idx:end_idx].view(-1, 1, 1, 1)
        sigma_t_group = sigma_t_clamped[start_idx:end_idx].view(-1, 1, 1, 1) # Use clamped

        # --- Predictions ---
        # Quantized model prediction (student)
        x0_pred_t = model(xt_group, s_emb_group, t_emb_group)

        # EMA model prediction (teacher for NeM and target for MMD r->s)
        with torch.no_grad():
            x0_pred_r_ema = model_ema(xr_group, s_emb_group, r_emb_group)
            x0_pred_t_ema = model_ema(xt_group, s_emb_group, t_emb_group) # Teacher for NeM

        # --- Pushforward for MMD ---
        ys_t = ddim_sample_step(xt_group, x0_pred_t, s_group_val, t_group_val, schedule_fn)
        with torch.no_grad():
            ys_r = ddim_sample_step(xr_group, x0_pred_r_ema, s_group_val, r_group_val, schedule_fn)

        # --- MMD Calculation ---
        term1 = 0.0
        term2 = 0.0
        term3 = 0.0
        for j in range(M):
            for k in range(M):
                 # Ensure inputs to kernel are float
                 term1 += kernel_fn(ys_t[j:j+1].float(), ys_t[k:k+1].float(), bandwidth)
                 term2 += kernel_fn(ys_r[j:j+1].float(), ys_r[k:k+1].float(), bandwidth)
                 term3 += kernel_fn(ys_t[j:j+1].float(), ys_r[k:k+1].float(), bandwidth)

        if M > 0:
             mmd_sq = (term1 / (M * M) + term2 / (M * M) - 2 * term3 / (M * M))
             group_mmd_loss = torch.relu(mmd_sq) # Ensure non-negative
        else:
             group_mmd_loss = torch.tensor(0.0, device=DEVICE)
        total_mmd_loss += group_mmd_loss

        # --- NeM Calculation ---
        # Predicted noise from quantized model (student)
        eps_pred_q = (xt_group - alpha_t_group * x0_pred_t) / sigma_t_group

        # Predicted noise from EMA model (teacher) - NO GRAD here
        with torch.no_grad():
             eps_pred_ema = (xt_group - alpha_t_group * x0_pred_t_ema) / sigma_t_group

        # NeM Loss: MSE between predicted noises (detach teacher)
        group_nem_loss = F.mse_loss(eps_pred_q, eps_pred_ema.detach())
        total_nem_loss += group_nem_loss

    # Average losses over groups
    final_mmd_loss = total_mmd_loss / num_groups if num_groups > 0 else torch.tensor(0.0, device=DEVICE)
    final_nem_loss = total_nem_loss / num_groups if num_groups > 0 else torch.tensor(0.0, device=DEVICE)

    # Combine losses
    final_loss = final_mmd_loss + lambda_nem * final_nem_loss

    return final_loss, final_mmd_loss, final_nem_loss

# ==============================================================================
# 5. Sampling Function
# ==============================================================================

@torch.no_grad()
def generate_samples(model, num_samples=64, num_steps=20, schedule_fn=get_linear_schedule, time_emb_dim_quant=128):
    """Generate samples using the trained quantized model."""
    model.eval()
    x_t = torch.randn(num_samples, IMG_CHANNELS, IMG_SIZE, IMG_SIZE, device=DEVICE)

    time_steps = torch.linspace(T_MAX, T_EPSILON, num_steps + 1, device=DEVICE)

    for i in tqdm(range(num_steps), desc="Sampling", leave=False):
        t_curr = time_steps[i]
        t_next = time_steps[i+1]

        # Prepare batch time embeddings (original dimension for quant scale MLPs)
        t_curr_batch = torch.full((num_samples,), t_curr, device=DEVICE)
        t_next_batch = torch.full((num_samples,), t_next, device=DEVICE)
        t_curr_emb = sinusoidal_embedding(t_curr_batch, time_emb_dim_quant)
        t_next_emb = sinusoidal_embedding(t_next_batch, time_emb_dim_quant)

        # Model predicts x0 given xt, s_emb, t_emb
        x0_pred = model(x_t, t_next_emb, t_curr_emb) # Pass s=next, t=curr embeddings

        # Push forward using DDIM step
        x_t = ddim_sample_step(x_t, x0_pred, t_next_batch, t_curr_batch, schedule_fn)

    model.train()
    samples = torch.clamp(x_t, -1.0, 1.0)
    samples = (samples + 1.0) / 2.0 # Map from [-1, 1] to [0, 1]
    return samples

# ==============================================================================
# 6. Training Setup
# ==============================================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1] for RGB
])

try:
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
except Exception as e:
    print(f"Error downloading/loading CIFAR-10 dataset: {e}")
    print("Please check your internet connection or dataset path.")
    exit()

# Ensure batch size is compatible with M_PARTICLES
if BATCH_SIZE % M_PARTICLES != 0:
     print(f"Adjusting BATCH_SIZE from {BATCH_SIZE} to {(BATCH_SIZE // M_PARTICLES) * M_PARTICLES} to be divisible by M_PARTICLES={M_PARTICLES}")
     BATCH_SIZE = (BATCH_SIZE // M_PARTICLES) * M_PARTICLES
if BATCH_SIZE == 0:
    print("Error: Batch size cannot be zero after adjustment for M_PARTICLES.")
    exit()

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=min(4, os.cpu_count()), drop_last=True, pin_memory=True if DEVICE.type == 'cuda' else False)

# Instantiate Quantized Model with increased capacity
model = QSimpleUNet(
    img_channels=IMG_CHANNELS,
    time_emb_dim=TIME_EMB_DIM,
    base_dim=256,  # Increased base dimension for CIFAR
    w_bits=W_BITS,
    a_bits=A_BITS
).to(DEVICE)

# Create EMA model
model_ema = deepcopy(model).to(DEVICE)
for param in model_ema.parameters():
    param.requires_grad = False

# Optimizer with learning rate scheduler
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

print(f"Quantized Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Using W_BITS={W_BITS}, A_BITS={A_BITS}, LAMBDA_NEM={LAMBDA_NEM}")
print(f"Batch Size: {BATCH_SIZE}, M Particles: {M_PARTICLES}")
print(f"Training for {EPOCHS} epochs.")

# ==============================================================================
# 7. Training Loop
# ==============================================================================

losses = []
mmd_losses = []
nem_losses = []

print("\nStarting training...")
for epoch in range(EPOCHS):
    model.train()
    model_ema.eval()

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    epoch_loss = 0.0
    epoch_mmd_loss = 0.0
    epoch_nem_loss = 0.0
    num_batches = 0

    for batch_idx, (x_real, _) in enumerate(progress_bar):
        x_real = x_real.to(DEVICE)

        if x_real.shape[0] < M_PARTICLES or x_real.shape[0] % M_PARTICLES != 0:
            continue

        optimizer.zero_grad()

        loss, mmd_loss, nem_loss = imm_loss_nem(
            model, model_ema, x_real, M_PARTICLES,
            get_linear_schedule, laplace_kernel, KERNEL_BANDWIDTH,
            lambda_nem=LAMBDA_NEM,
            time_emb_dim_quant=TIME_EMB_DIM
        )

        if torch.isnan(loss) or torch.isinf(loss):
           print(f"\nNaN/Inf loss detected at Epoch {epoch+1}, Batch {batch_idx}. Loss: {loss.item()}. Skipping update.")
           continue

        loss.backward()

        if GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

        optimizer.step()
        update_ema(model_ema, model, EMA_DECAY)

        epoch_loss += loss.item()
        epoch_mmd_loss += mmd_loss.item()
        epoch_nem_loss += nem_loss.item()
        num_batches += 1
        progress_bar.set_postfix(loss=loss.item(), mmd=mmd_loss.item(), nem=nem_loss.item(), lr=scheduler.get_last_lr()[0])

    # Step the scheduler
    scheduler.step()

    if num_batches > 0:
        avg_epoch_loss = epoch_loss / num_batches
        avg_epoch_mmd_loss = epoch_mmd_loss / num_batches
        avg_epoch_nem_loss = epoch_nem_loss / num_batches
        losses.append(avg_epoch_loss)
        mmd_losses.append(avg_epoch_mmd_loss)
        nem_losses.append(avg_epoch_nem_loss)

        print(f"Epoch {epoch+1} Avg Loss: {avg_epoch_loss:.4f} (MMD: {avg_epoch_mmd_loss:.4f}, NeM: {avg_epoch_nem_loss:.4f}) LR: {scheduler.get_last_lr()[0]:.6f}")
    else:
        print(f"Epoch {epoch+1} had no valid batches.")
        losses.append(float('nan'))
        mmd_losses.append(float('nan'))
        nem_losses.append(float('nan'))

    # Generate and save samples periodically
    if (epoch + 1) % SAVE_INTERVAL == 0 or epoch == EPOCHS - 1:
        if num_batches > 0:
            print("Generating samples...")
            generated_samples = generate_samples(
                model_ema,
                num_samples=64,
                num_steps=50,
                time_emb_dim_quant=TIME_EMB_DIM
            )

            fig, axes = plt.subplots(8, 8, figsize=(8, 8))
            for i, ax in enumerate(axes.flat):
                if i < generated_samples.shape[0]:
                    # Handle RGB images correctly
                    sample = generated_samples[i].cpu().permute(1, 2, 0).numpy()
                    ax.imshow(sample)
                ax.axis('off')
            plt.tight_layout()
            save_path = os.path.join(RESULTS_DIR, f"epoch_{epoch+1:03d}_samples_w{W_BITS}a{A_BITS}.png")
            plt.savefig(save_path)
            plt.close(fig)
            print(f"Saved samples to {save_path}")

            # Save model checkpoint
            checkpoint_path = os.path.join(RESULTS_DIR, f"checkpoint_epoch_{epoch+1:03d}_w{W_BITS}a{A_BITS}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'ema_model_state_dict': model_ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_epoch_loss,
                'w_bits': W_BITS,
                'a_bits': A_BITS,
                'lambda_nem': LAMBDA_NEM,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        else:
            print(f"Skipping sample generation and checkpoint saving for epoch {epoch+1} due to no valid batches.")


# ==============================================================================
# 8. Final Plotting
# ==============================================================================
print("\nPlotting loss curves...")
plt.figure(figsize=(12, 6))
valid_epochs = [(i+1) for i, l in enumerate(losses) if not math.isnan(l)]
valid_losses = [l for l in losses if not math.isnan(l)]
valid_mmd_losses = [l for i, l in enumerate(mmd_losses) if not math.isnan(losses[i])]
valid_nem_losses = [l for i, l in enumerate(nem_losses) if not math.isnan(losses[i])]

if valid_epochs: # Check if there's anything to plot
    plt.plot(valid_epochs, valid_losses, label='Total Loss', marker='.')
    plt.plot(valid_epochs, valid_mmd_losses, label='MMD Loss', linestyle='--', marker='.')
    plt.plot(valid_epochs, [l * LAMBDA_NEM for l in valid_nem_losses], label=f'NeM Loss (scaled by $\\lambda$={LAMBDA_NEM:.2f})', linestyle=':', marker='.')
    # Plot raw NeM loss too?
    # plt.plot(valid_epochs, valid_nem_losses, label=f'NeM Loss (raw)', linestyle='-.', marker='.')

    plt.title(f"Quantized IMM (W={W_BITS}, A={A_BITS}) Training Loss on CIFAR-10")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, f"training_loss_curve_w{W_BITS}a{A_BITS}.png")
    plt.savefig(plot_path)
    print(f"Saved loss plot to {plot_path}")
    # plt.show() # Comment out if running non-interactively
else:
    print("No valid loss data to plot.")

print("\nQuantized training finished.")