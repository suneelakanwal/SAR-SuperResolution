# ============================================================
# SRCNN Training 
# ============================================================
%matplotlib inline
from IPython.display import display, Image as IPImage, clear_output
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import rcParams
from PIL import Image
import os, gc

rcParams['font.family'] = 'serif'
rcParams['font.serif']  = ['DejaVu Serif', 'Times New Roman', 'Times']
rcParams['font.size']   = 10

# ════════════════════════════════════════════════════════════
#  SRCNN Model (inline — no repo import needed)
# ════════════════════════════════════════════════════════════
class SRCNN(nn.Module):
    """
    SRCNN — Dong et al., ECCV 2014
    Adapted for single-channel (grayscale) SAR images.
    """
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# ════════════════════════════════════════════════════════════
#  Dataset
# ════════════════════════════════════════════════════════════
class SARDataset(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, 'r') as hf:
            self.length = len(hf['data'])
        self.h5_path = h5_path
        self.hf      = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.hf is None:
            self.hf = h5py.File(self.h5_path, 'r')
        lr = torch.from_numpy(self.hf['data'][idx])
        hr = torch.from_numpy(self.hf['label'][idx])
        return lr, hr

# ── Device ───────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Device : {device} | GPU: {torch.cuda.get_device_name(0)}")
else:
    print(f"Device : {device} (no GPU — training will be slower)")

# ── Paths (Kaggle) ───────────────────────────────────────────
SAVE_DIR = '/kaggle/working/srcnn_outputs'
os.makedirs(SAVE_DIR, exist_ok=True)
train_h5 = '/kaggle/working/sar_train_x2.h5'
eval_h5  = '/kaggle/working/sar_eval_x2.h5'

# ── Hyperparameters ───────────────────────────────────────────
NUM_EPOCHS    = 50
BATCH_SIZE    = 128
MAX_TRAIN     = 50000
MAX_EVAL      = 5000
PREVIEW_EVERY = 10

# ── Model ─────────────────────────────────────────────────────
model     = SRCNN(num_channels=1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam([
    {'params': model.conv1.parameters(), 'lr': 1e-4},
    {'params': model.conv2.parameters(), 'lr': 1e-4},
    {'params': model.conv3.parameters(), 'lr': 1e-5},
])
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── DataLoaders ───────────────────────────────────────────────
full_train = SARDataset(train_h5)
full_eval  = SARDataset(eval_h5)

train_idx = np.random.choice(len(full_train),
                              min(MAX_TRAIN, len(full_train)),
                              replace=False)
eval_idx  = np.random.choice(len(full_eval),
                              min(MAX_EVAL, len(full_eval)),
                              replace=False)

train_loader = DataLoader(Subset(full_train, train_idx),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
eval_loader  = DataLoader(Subset(full_eval, eval_idx),
                          batch_size=64, shuffle=False,
                          num_workers=2, pin_memory=True)

print(f"Train : {len(train_idx):,} samples | {len(train_loader)} batches")
print(f"Eval  : {len(eval_idx):,}  samples | {len(eval_loader)} batches")

# ── 4 fixed patches for visual preview ───────────────────────
preview_patches = []
with h5py.File(eval_h5, 'r') as hf:
    for i in range(4):
        lr = hf['data'][i]
        hr = hf['label'][i]
        preview_patches.append((
            torch.from_numpy(lr).unsqueeze(0),
            torch.from_numpy(hr).unsqueeze(0)
        ))

# ════════════════════════════════════════════════════════════
#  Preview function — loss curves + LR/SR/HR patches
# ════════════════════════════════════════════════════════════
def show_preview(epoch, model, patches, train_losses, eval_psnrs):
    model.eval()
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle(
        f"SRCNN Training — Epoch {epoch}/{NUM_EPOCHS}  |  "
        f"Best PSNR: {max(eval_psnrs):.2f} dB",
        fontsize=13, fontweight='bold'
    )

    # Row 0 — Training Loss curve (span all 4 cols)
    ax_loss = fig.add_subplot(3, 1, 1)
    ax_loss.plot(train_losses, color='royalblue', linewidth=2, label='Train Loss')
    ax_loss.set_title("Training Loss (MSE)", fontweight='bold')
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("MSE Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(fontsize=9)

    # Row 1 — PSNR curve (span all 4 cols)
    ax_psnr = fig.add_subplot(3, 1, 2)
    ax_psnr.plot(eval_psnrs, color='darkorange', linewidth=2, label='Eval PSNR')
    best_ep = int(np.argmax(eval_psnrs))
    ax_psnr.axvline(best_ep, color='crimson', linestyle='--', linewidth=1.5,
                    label=f'Best: {max(eval_psnrs):.2f} dB (ep {best_ep+1})')
    ax_psnr.set_title("Validation PSNR", fontweight='bold')
    ax_psnr.set_xlabel("Epoch")
    ax_psnr.set_ylabel("PSNR (dB)")
    ax_psnr.grid(True, alpha=0.3)
    ax_psnr.legend(fontsize=9)

    # Row 2 — 4 patch comparisons: LR | SR | HR | Residual
    gs_bottom = fig.add_gridspec(1, 4, top=0.3, bottom=0.02,
                                  hspace=0.1, wspace=0.05)

    with torch.no_grad():
        for p_idx, (lr_t, hr_t) in enumerate(patches):
            sr_t   = model(lr_t.to(device)).clamp(0, 1).cpu()
            lr_np  = lr_t[0, 0].numpy()
            sr_np  = sr_t[0, 0].numpy()
            hr_np  = hr_t[0, 0].numpy()
            res_np = np.abs(hr_np - sr_np)

            inner = gs_bottom[p_idx].subgridspec(4, 1, hspace=0.05)
            for r, (img, title, cmap) in enumerate([
                (lr_np,  "LR Input",  'gray'),
                (sr_np,  "SR Output", 'gray'),
                (hr_np,  "HR Target", 'gray'),
                (res_np, "Residual",  'hot'),
            ]):
                ax = fig.add_subplot(inner[r])
                ax.imshow(img, cmap=cmap, vmin=0,
                          vmax=1 if r < 3 else res_np.max() + 1e-6,
                          interpolation='nearest', aspect='auto')
                if p_idx == 0:
                    ax.set_ylabel(title, fontsize=7, rotation=0,
                                  labelpad=38, va='center')
                ax.axis('off')

    save_path = f'{SAVE_DIR}/training_preview_ep{epoch}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    display(IPImage(save_path))
    plt.close(fig)
    model.train()

# ════════════════════════════════════════════════════════════
#  Training Loop
# ════════════════════════════════════════════════════════════
train_losses, eval_psnrs  = [], []
best_psnr, best_epoch     = 0.0, 0

print(f"\nTraining {NUM_EPOCHS} epochs | Preview every {PREVIEW_EVERY} epochs\n")
print(f"{'Epoch':>6} | {'Loss':>10} | {'PSNR':>8} | {'Best PSNR':>10}")
print("-" * 48)

for epoch in range(1, NUM_EPOCHS + 1):

    # ── Train ─────────────────────────────────────────────────
    model.train()
    epoch_loss = 0.0
    for lr_b, hr_b in train_loader:
        lr_b = lr_b.to(device, non_blocking=True)
        hr_b = hr_b.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = criterion(model(lr_b), hr_b)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    # ── Eval ──────────────────────────────────────────────────
    model.eval()
    epoch_psnr = 0.0
    n_samples  = 0
    with torch.no_grad():
        for lr_b, hr_b in eval_loader:
            lr_b  = lr_b.to(device, non_blocking=True)
            hr_b  = hr_b.to(device, non_blocking=True)
            preds = model(lr_b).clamp(0, 1)
            mse   = ((preds - hr_b) ** 2).mean(dim=[1, 2, 3])
            epoch_psnr += (10 * torch.log10(
                1.0 / (mse + 1e-10))).sum().item()
            n_samples += lr_b.size(0)

    avg_psnr = epoch_psnr / n_samples
    eval_psnrs.append(avg_psnr)

    # ── Save best ─────────────────────────────────────────────
    if avg_psnr > best_psnr:
        best_psnr, best_epoch = avg_psnr, epoch
        torch.save(model.state_dict(),
                   f'{SAVE_DIR}/srcnn_sar_best.pth')

    # ── Print every 5 epochs ──────────────────────────────────
    if epoch % 5 == 0 or epoch == 1:
        best_marker = '  ← best' if epoch == best_epoch else ''
        print(f"{epoch:>6} | {avg_loss:>10.6f} | "
              f"{avg_psnr:>8.3f} | {best_psnr:>10.3f} dB{best_marker}")

    # ── Preview every N epochs ────────────────────────────────
    if epoch % PREVIEW_EVERY == 0 or epoch == NUM_EPOCHS:
        show_preview(epoch, model, preview_patches,
                     train_losses, eval_psnrs)

# ── Save final model ──────────────────────────────────────────
torch.save(model.state_dict(), f'{SAVE_DIR}/srcnn_sar_final.pth')

# ════════════════════════════════════════════════════════════
#  Final Training Curves
# ════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("SRCNN Final Training Curves — Sentinel-1 SAR",
             fontsize=13, fontweight='bold')

ax1.plot(train_losses, color='royalblue', linewidth=2)
ax1.fill_between(range(len(train_losses)), train_losses,
                 alpha=0.12, color='royalblue')
ax1.set_title("Training Loss (MSE)", fontweight='bold')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("MSE Loss")
ax1.grid(True, alpha=0.3)

ax2.plot(eval_psnrs, color='darkorange', linewidth=2)
ax2.fill_between(range(len(eval_psnrs)), eval_psnrs,
                 alpha=0.12, color='darkorange')
ax2.axvline(best_epoch - 1, color='crimson', linestyle='--', linewidth=1.5,
            label=f'Best: {best_psnr:.2f} dB (ep {best_epoch})')
ax2.set_title("Validation PSNR (dB)", fontweight='bold')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("PSNR (dB)")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/training_curves_final.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig(f'{SAVE_DIR}/training_curves_final.pdf',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
display(IPImage(f'{SAVE_DIR}/training_curves_final.png'))

print("\n" + "=" * 55)
print("  ✅ TRAINING COMPLETE")
print("=" * 55)
print(f"  Best PSNR  : {best_psnr:.4f} dB  (epoch {best_epoch})")
print(f"  Final PSNR : {eval_psnrs[-1]:.4f} dB")
print(f"  Best model : {SAVE_DIR}/srcnn_sar_best.pth")
print(f"  Final model: {SAVE_DIR}/srcnn_sar_final.pth")
print("=" * 55)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
