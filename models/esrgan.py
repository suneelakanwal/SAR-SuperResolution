# ============================================================
#  ESRGAN Training 
# ============================================================
import torch, gc, os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from IPython.display import display, Image as IPImage
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
 
rcParams['font.family'] = 'serif'
rcParams['font.serif']  = ['DejaVu Serif', 'Times New Roman', 'Times']
 
# ── Loss weights ──────────────────────────────────────────────
W_PIXEL      = 1.0
W_PERCEPTUAL = 1.0
W_GAN        = 0.1
 
# ── Optimizers ────────────────────────────────────────────────
optimizer_G     = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.9, 0.999))
optimizer_D     = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.9, 0.999))
criterion_pixel = nn.L1Loss()
criterion_gan   = nn.BCEWithLogitsLoss()
 
EPOCHS_WARMUP = 30
EPOCHS_GAN    = 50
PREVIEW_EVERY = 30
 
# ── PSNR helper ───────────────────────────────────────────────
def calc_psnr_batch(sr, hr):
    mse = ((sr - hr) ** 2).mean(dim=[1, 2, 3])
    return (10 * torch.log10(1.0 / (mse + 1e-10))).mean().item()
 
# ════════════════════════════════════════════════════════════
#  Preview: loss curves + LR(upscaled) / SR / HR / Residual
# ════════════════════════════════════════════════════════════
def show_esrgan_preview(epoch, stage, netG, patches,
                         g_losses, d_losses, psnr_list):
    netG.eval()
 
    # Get images from first patch
    with torch.no_grad():
        lr_t, hr_t = patches[0]              # lr:(1,1,33,33) hr:(1,1,66,66)
        sr_t = netG(lr_t.to(device)).clamp(0, 1).cpu()
 
        lr_np  = lr_t[0, 0].numpy()          # 33×33
        sr_np  = sr_t[0, 0].numpy()          # 66×66
        hr_np  = hr_t[0, 0].numpy()          # 66×66
        res_np = np.abs(hr_np - sr_np)
 
        # Upscale LR to 66×66 for side-by-side display
        from PIL import Image as PILImage
        lr_display = np.array(
            PILImage.fromarray((lr_np * 255).astype(np.uint8)).resize(
                (66, 66), PILImage.BICUBIC)
        ).astype(np.float32) / 255.0
 
        p_bicubic = psnr_fn(hr_np, lr_display, data_range=1.0)
        p_sr      = psnr_fn(hr_np, sr_np,      data_range=1.0)
 
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(
        f"ESRGAN [{stage}] — Epoch {epoch}  |  "
        f"Best PSNR: {max(psnr_list):.2f} dB",
        fontsize=13, fontweight='bold'
    )
 
    # ── Top row: curves ───────────────────────────────────────
    gs_top = fig.add_gridspec(1, 2,
                               top=0.92, bottom=0.52,
                               left=0.06, right=0.97,
                               wspace=0.25)
 
    ax_loss = fig.add_subplot(gs_top[0])
    ax_loss.plot(g_losses, color='royalblue',
                 linewidth=2, label='G Loss')
    if d_losses:
        ax_loss.plot(d_losses, color='crimson', linewidth=2,
                     linestyle='--', label='D Loss')
    ax_loss.set_title("Generator / Discriminator Loss",
                       fontweight='bold')
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(fontsize=9)
    ax_loss.grid(alpha=0.3)
 
    ax_psnr = fig.add_subplot(gs_top[1])
    ax_psnr.plot(psnr_list, color='darkorange', linewidth=2)
    ax_psnr.axhline(max(psnr_list), color='crimson',
                    linestyle='--', linewidth=1.5,
                    label=f'Best: {max(psnr_list):.2f} dB')
    ax_psnr.set_title("Validation PSNR", fontweight='bold')
    ax_psnr.set_xlabel("Epoch")
    ax_psnr.set_ylabel("PSNR (dB)")
    ax_psnr.legend(fontsize=9)
    ax_psnr.grid(alpha=0.3)
 
    # ── Bottom row: images ─────────────────────────────────────
    gs_bot = fig.add_gridspec(1, 4,
                               top=0.46, bottom=0.04,
                               left=0.04, right=0.97,
                               wspace=0.06)
 
    panels = [
        (lr_display, f'⬇️ LR (Bicubic)\nPSNR: {p_bicubic:.2f} dB',
         'crimson',     'gray'),
        (sr_np,      f'✨ ESRGAN SR\nPSNR: {p_sr:.2f} dB',
         'dodgerblue',  'gray'),
        (hr_np,      '⬆️ HR Ground Truth',
         'green',       'gray'),
        (res_np,     '🔥 Residual |HR−SR|',
         'darkorange',  'hot'),
    ]
 
    for col, (img, title, color, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs_bot[col])
        vmax = res_np.max() + 1e-6 if col == 3 else 1.0
        ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax,
                  interpolation='nearest', aspect='auto')
        ax.set_title(title, fontweight='bold',
                     color=color, fontsize=10, pad=6)
        ax.axis('off')
 
    save_png = f'{SAVE_DIR}/esrgan_preview_{stage}_ep{epoch:03d}.png'
    save_pdf = f'{SAVE_DIR}/esrgan_preview_{stage}_ep{epoch:03d}.pdf'
    plt.savefig(save_png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(save_pdf, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    display(IPImage(save_png))
    plt.close(fig)
    print(f"   ✅ Saved: {os.path.basename(save_png)} / .pdf")
    netG.train()
 
 
# ════════════════════════════════════════════════════════════
#  STAGE 1 — PSNR Warmup
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("  STAGE 1 — PSNR WARMUP (L1 Pixel Loss Only)")
print(f"  Epochs: {EPOCHS_WARMUP}  |  LR: 33×33 → SR: 66×66")
print("=" * 60)
 
g_losses_warmup, psnr_warmup     = [], []
best_psnr_warmup, best_ep_warmup = 0.0, 0
 
print(f"\n{'Epoch':>6} | {'G Loss':>10} | {'PSNR':>8} | {'Best':>10}")
print("-" * 44)
 
for epoch in range(1, EPOCHS_WARMUP + 1):
    netG.train()
    epoch_g = 0.0
 
    for lr_b, hr_b in train_loader:
        lr_b = lr_b.to(device, non_blocking=True)   # (B,1,33,33)
        hr_b = hr_b.to(device, non_blocking=True)   # (B,1,66,66)
 
        optimizer_G.zero_grad()
        sr_b   = netG(lr_b)                          # (B,1,66,66)
        loss_G = criterion_pixel(sr_b, hr_b)
        loss_G.backward()
        optimizer_G.step()
        epoch_g += loss_G.item()
 
    avg_g = epoch_g / len(train_loader)
    g_losses_warmup.append(avg_g)
 
    # Eval
    netG.eval()
    epoch_psnr, n = 0.0, 0
    with torch.no_grad():
        for lr_b, hr_b in eval_loader:
            lr_b = lr_b.to(device, non_blocking=True)
            hr_b = hr_b.to(device, non_blocking=True)
            sr_b = netG(lr_b).clamp(0, 1)
            epoch_psnr += calc_psnr_batch(sr_b, hr_b) * lr_b.size(0)
            n += lr_b.size(0)
    avg_psnr = epoch_psnr / n
    psnr_warmup.append(avg_psnr)
 
    if avg_psnr > best_psnr_warmup:
        best_psnr_warmup, best_ep_warmup = avg_psnr, epoch
        torch.save(netG.state_dict(),
                   f'{SAVE_DIR}/esrgan_warmup_best.pth')
 
    if epoch % 5 == 0 or epoch == 1:
        marker = '  ← best' if epoch == best_ep_warmup else ''
        print(f"{epoch:>6} | {avg_g:>10.6f} | "
              f"{avg_psnr:>8.3f} | "
              f"{best_psnr_warmup:>8.3f} dB{marker}")
 
    if epoch % PREVIEW_EVERY == 0 or epoch == EPOCHS_WARMUP:
        show_esrgan_preview(epoch, 'WARMUP', netG,
                             preview_patches,
                             g_losses_warmup, [],
                             psnr_warmup)
 
# Stage 1 final curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("ESRGAN Stage 1 — PSNR Warmup Curves",
             fontsize=13, fontweight='bold')
ax1.plot(g_losses_warmup, color='royalblue', linewidth=2)
ax1.fill_between(range(len(g_losses_warmup)),
                 g_losses_warmup, alpha=0.12, color='royalblue')
ax1.set_title("Generator Loss (L1 Pixel)", fontweight='bold')
ax1.set_xlabel("Epoch"); ax1.set_ylabel("L1 Loss"); ax1.grid(alpha=0.3)
ax2.plot(psnr_warmup, color='darkorange', linewidth=2)
ax2.fill_between(range(len(psnr_warmup)),
                 psnr_warmup, alpha=0.12, color='darkorange')
ax2.axvline(best_ep_warmup - 1, color='crimson', linestyle='--',
            linewidth=1.5,
            label=f'Best: {best_psnr_warmup:.2f} dB (ep {best_ep_warmup})')
ax2.set_title("Validation PSNR", fontweight='bold')
ax2.set_xlabel("Epoch"); ax2.set_ylabel("PSNR (dB)")
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/stage1_warmup_curves.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig(f'{SAVE_DIR}/stage1_warmup_curves.pdf',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
display(IPImage(f'{SAVE_DIR}/stage1_warmup_curves.png'))
print(f"✅ Saved: stage1_warmup_curves.png / .pdf")
print(f"\n✅ Stage 1 done | Best PSNR: {best_psnr_warmup:.4f} dB (ep {best_ep_warmup})")
 
# Load best weights before GAN stage
netG.load_state_dict(
    torch.load(f'{SAVE_DIR}/esrgan_warmup_best.pth',
               map_location=device))
print("✅ Best warmup weights loaded for Stage 2")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
 
 
# ════════════════════════════════════════════════════════════
#  STAGE 2 — GAN Fine-tuning
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STAGE 2 — GAN FINE-TUNING")
print(f"  Epochs: {EPOCHS_GAN}  |  L1 + Perceptual + Adversarial")
print(f"  Weights → pixel:{W_PIXEL} | perceptual:{W_PERCEPTUAL} | GAN:{W_GAN}")
print("=" * 60)
 
g_losses_gan, d_losses_gan, psnr_gan = [], [], []
best_psnr_gan, best_ep_gan           = 0.0, 0
 
print(f"\n{'Epoch':>6} | {'G Loss':>10} | {'D Loss':>10} | "
      f"{'PSNR':>8} | {'Best':>10}")
print("-" * 56)
 
for epoch in range(1, EPOCHS_GAN + 1):
    netG.train()
    netD.train()
    epoch_g, epoch_d = 0.0, 0.0
 
    for lr_b, hr_b in train_loader:
        lr_b = lr_b.to(device, non_blocking=True)   # (B,1,33,33)
        hr_b = hr_b.to(device, non_blocking=True)   # (B,1,66,66)
        bs   = lr_b.size(0)
 
        real_label = torch.ones(bs,  1, device=device)
        fake_label = torch.zeros(bs, 1, device=device)
 
        # Train Discriminator
        optimizer_D.zero_grad()
        sr_detach = netG(lr_b).detach().clamp(0, 1)
        loss_D = (criterion_gan(netD(hr_b),       real_label) +
                  criterion_gan(netD(sr_detach),  fake_label)) * 0.5
        loss_D.backward()
        optimizer_D.step()
        epoch_d += loss_D.item()
 
        # Train Generator
        optimizer_G.zero_grad()
        sr_b         = netG(lr_b)
        loss_pixel   = criterion_pixel(sr_b, hr_b)
        loss_percept = vgg_loss(sr_b, hr_b)
        loss_adv     = criterion_gan(netD(sr_b), real_label)
        loss_G = (W_PIXEL      * loss_pixel   +
                  W_PERCEPTUAL * loss_percept +
                  W_GAN        * loss_adv)
        loss_G.backward()
        optimizer_G.step()
        epoch_g += loss_G.item()
 
    avg_g = epoch_g / len(train_loader)
    avg_d = epoch_d / len(train_loader)
    g_losses_gan.append(avg_g)
    d_losses_gan.append(avg_d)
 
    # Eval
    netG.eval()
    epoch_psnr, n = 0.0, 0
    with torch.no_grad():
        for lr_b, hr_b in eval_loader:
            lr_b = lr_b.to(device, non_blocking=True)
            hr_b = hr_b.to(device, non_blocking=True)
            sr_b = netG(lr_b).clamp(0, 1)
            epoch_psnr += calc_psnr_batch(sr_b, hr_b) * lr_b.size(0)
            n += lr_b.size(0)
    avg_psnr = epoch_psnr / n
    psnr_gan.append(avg_psnr)
 
    if avg_psnr > best_psnr_gan:
        best_psnr_gan, best_ep_gan = avg_psnr, epoch
        torch.save(netG.state_dict(),
                   f'{SAVE_DIR}/esrgan_gan_best.pth')
        torch.save(netD.state_dict(),
                   f'{SAVE_DIR}/esrgan_disc_best.pth')
 
    if epoch % 5 == 0 or epoch == 1:
        marker = '  ← best' if epoch == best_ep_gan else ''
        print(f"{epoch:>6} | {avg_g:>10.6f} | {avg_d:>10.6f} | "
              f"{avg_psnr:>8.3f} | "
              f"{best_psnr_gan:>8.3f} dB{marker}")
 
    if epoch % PREVIEW_EVERY == 0 or epoch == EPOCHS_GAN:
        show_esrgan_preview(epoch, 'GAN', netG,
                             preview_patches,
                             g_losses_gan, d_losses_gan,
                             psnr_gan)
 
torch.save(netG.state_dict(), f'{SAVE_DIR}/esrgan_gan_final.pth')
 
# Stage 2 final curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("ESRGAN Stage 2 — GAN Fine-Tuning Curves",
             fontsize=13, fontweight='bold')
ax1.plot(g_losses_gan, color='royalblue', linewidth=2, label='G Loss')
ax1.plot(d_losses_gan, color='crimson',   linewidth=2,
         linestyle='--', label='D Loss')
ax1.fill_between(range(len(g_losses_gan)),
                 g_losses_gan, alpha=0.1, color='royalblue')
ax1.set_title("G / D Loss (GAN Stage)", fontweight='bold')
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
ax2.plot(psnr_gan, color='darkorange', linewidth=2)
ax2.fill_between(range(len(psnr_gan)),
                 psnr_gan, alpha=0.12, color='darkorange')
ax2.axvline(best_ep_gan - 1, color='crimson', linestyle='--',
            linewidth=1.5,
            label=f'Best: {best_psnr_gan:.2f} dB (ep {best_ep_gan})')
ax2.set_title("Validation PSNR (GAN Stage)", fontweight='bold')
ax2.set_xlabel("Epoch"); ax2.set_ylabel("PSNR (dB)")
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/stage2_gan_curves.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig(f'{SAVE_DIR}/stage2_gan_curves.pdf',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
display(IPImage(f'{SAVE_DIR}/stage2_gan_curves.png'))
print(f"✅ Saved: stage2_gan_curves.png / .pdf")
 
print("\n" + "=" * 60)
print("  ✅ ESRGAN TRAINING COMPLETE")
print("=" * 60)
print(f"  Stage 1 Best PSNR : {best_psnr_warmup:.4f} dB (ep {best_ep_warmup})")
print(f"  Stage 2 Best PSNR : {best_psnr_gan:.4f} dB  (ep {best_ep_gan})")
print(f"  Models saved to   : {SAVE_DIR}/")
print("=" * 60)
print("  ▶️  Run Cell 5 — Final Evaluation & Comparison")
 
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
 
 
