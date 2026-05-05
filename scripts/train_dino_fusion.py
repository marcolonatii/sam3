"""
Train the DINOv3 fusion components (projection head + cross-attention fuser)
on top of a frozen SAM3 backbone.

Training path (propagation path, no-memory):
    frozen SAM3 backbone → _prepare_backbone_features
    → no-memory fallback (raw vision features, pix_feat = vision_feats[-1])
    → DINO cross-attention fusion                    ← weights trained here
    → frozen sam_mask_decoder (with a GT-sampled point prompt)

Prompt: one positive click sampled uniformly from the GT mask, per image.
This is the standard SAM training protocol (same as SAM2's own training).

Datasets:
    - MoCA video frames    : --moca_frames / --moca_masks
    - COD10K static images : --cod10k_root
    - CAMO  static images  : --camo_root

Only dino_encoder.proj and cross_attn_fuser are trained.
Everything else (SAM3 backbone, memory modules, DINOv3 backbone,
sam_mask_decoder, prompt_encoder) is frozen.

SAM3 normalizes images with (0.5, 0.5, 0.5) mean/std (NOT ImageNet stats).
SAM3 input size: 1008×1008 pixels; backbone stride 14 → 72×72 feature map.

Checkpoints are saved as:
    {
        "epoch":               <int>,
        "dino_encoder":        <state_dict>,   # full DinoEncoder state dict
        "cross_attention_fuser": <state_dict>, # full CrossAttentionFuser state dict
        "optimizer":           <state_dict>,
        "scheduler":           <state_dict>,
        "train_loss":          <float>,
        "val_loss":            <float>,
    }
Load at inference with:
    build_sam3_predictor(version="sam3", use_dino_fusion=True,
                         dino_checkpoint_path="dino_fusion_best.pt")

Usage:
    # run from /home/marcol01/sam3/
    python scripts/train_dino_fusion.py
    python scripts/train_dino_fusion.py --epochs 30 --lr 1e-4
    python scripts/train_dino_fusion.py \\
        --checkpoint /path/to/sam3_checkpoint.pt
"""

import argparse
import os
import sys
import time

# Ensure the sam3 repo root is on the path when running as a script
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

# ---------------------------------------------------------------------------
# SAM3 image preprocessing constants
# SAM3 uses (0.5, 0.5, 0.5) normalization: (pixel/255 - 0.5) / 0.5
# ---------------------------------------------------------------------------

SAM3_IMAGE_SIZE = 1008
SAM3_PIXEL_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
SAM3_PIXEL_STD  = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)


def _load_image(img_path: str, size: int = SAM3_IMAGE_SIZE) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    return (t - SAM3_PIXEL_MEAN) / SAM3_PIXEL_STD


def _load_mask(mask_path: str, size: int = SAM3_IMAGE_SIZE) -> torch.Tensor:
    m = Image.open(mask_path).convert("L")
    m = m.resize((size, size), Image.NEAREST)
    t = torch.from_numpy(np.array(m)).float() / 255.0
    return (t > 0.5).float().unsqueeze(0)  # [1, H, W] binary float


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class MoCADataset(Dataset):
    """Per-frame dataset from MoCA; only annotated frames are used."""

    def __init__(self, frames_root: str, masks_root: str):
        super().__init__()
        self.samples = []
        for seq in sorted(os.listdir(masks_root)):
            mask_dir  = os.path.join(masks_root,  seq)
            frame_dir = os.path.join(frames_root, seq)
            if not os.path.isdir(mask_dir) or not os.path.isdir(frame_dir):
                continue
            for mask_file in sorted(os.listdir(mask_dir)):
                mask_path  = os.path.join(mask_dir, mask_file)
                stem       = os.path.splitext(mask_file)[0]
                frame_path = os.path.join(frame_dir, stem + ".jpg")
                if not os.path.exists(frame_path):
                    frame_path = os.path.join(frame_dir, stem + ".png")
                if os.path.exists(frame_path):
                    self.samples.append((frame_path, mask_path))
        print(f"[MoCA]   {len(self.samples)} image-mask pairs from {frames_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_path, mask_path = self.samples[idx]
        return _load_image(frame_path), _load_mask(mask_path)


class COD10KDataset(Dataset):
    def __init__(self, root: str):
        super().__init__()
        img_dir = os.path.join(root, "Train", "Image")
        gt_dir  = os.path.join(root, "Train", "GT_Object")
        self.samples = []
        for fn in sorted(os.listdir(img_dir)):
            if not fn.endswith(".jpg"):
                continue
            stem    = os.path.splitext(fn)[0]
            gt_path = os.path.join(gt_dir, stem + ".png")
            if os.path.exists(gt_path):
                self.samples.append((os.path.join(img_dir, fn), gt_path))
        print(f"[COD10K] {len(self.samples)} image-mask pairs from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        return _load_image(img_path), _load_mask(mask_path)


class CAMODataset(Dataset):
    def __init__(self, root: str):
        super().__init__()
        img_dir = os.path.join(root, "Images", "Train")
        gt_dir  = os.path.join(root, "GT")
        self.samples = []
        for fn in sorted(os.listdir(img_dir)):
            if not fn.endswith(".jpg"):
                continue
            stem    = os.path.splitext(fn)[0]
            gt_path = os.path.join(gt_dir, stem + ".png")
            if os.path.exists(gt_path):
                self.samples.append((os.path.join(img_dir, fn), gt_path))
        print(f"[CAMO]   {len(self.samples)} image-mask pairs from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        return _load_image(img_path), _load_mask(mask_path)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0):
    pred   = torch.sigmoid(pred_logits).flatten(1)
    target = target.flatten(1)
    inter  = (pred * target).sum(1)
    union  = pred.sum(1) + target.sum(1)
    return (1.0 - (2.0 * inter + smooth) / (union + smooth)).mean()


def combined_loss(pred_logits: torch.Tensor, target: torch.Tensor):
    return F.binary_cross_entropy_with_logits(pred_logits, target) + dice_loss(pred_logits, target)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(args):
    """Build SAM3 full model, then set up tracker with backbone and DINO fusion."""
    from sam3.model.cross_attention_fuser import CrossAttentionFuser
    from sam3.model.dino_encoder import DinoEncoder
    from sam3.model_builder import build_sam3_video_model

    # Build the full SAM3 model (detector + tracker); loads checkpoint with strict=False
    # so DINO fusion keys being absent is fine.
    full_model = build_sam3_video_model(
        checkpoint_path=args.checkpoint,
        load_from_HF=(args.checkpoint is None),
        strict_state_dict_loading=False,
        device="cpu",  # move to GPU after freezing
    )

    # Give the tracker access to the shared vision backbone (needed for forward_image).
    # The backbone lives in full_model.detector.backbone, which returns both
    # sam3_backbone_out (detector) and sam2_backbone_out (tracker) in forward_image.
    tracker = full_model.tracker
    tracker.backbone = full_model.detector.backbone

    # Attach DINO fusion modules to the tracker
    dino_encoder = DinoEncoder(
        model_name=args.dino_model,
        out_dim=256,
        freeze_backbone=True,
    )
    cross_attn_fuser = CrossAttentionFuser(embed_dim=256, num_heads=8)
    tracker.set_dino_fusion(dino_encoder, cross_attn_fuser)

    if args.init_weights is not None:
        if not os.path.isfile(args.init_weights):
            print(f"ERROR: init_weights checkpoint not found: {args.init_weights}")
            sys.exit(1)
        print(f"Loading initial fusion weights from: {args.init_weights}")
        ckpt = torch.load(args.init_weights, map_location="cpu", weights_only=True)
        tracker.dino_encoder.load_state_dict(ckpt["dino_encoder"])
        tracker.cross_attention_fuser.load_state_dict(ckpt["cross_attention_fuser"])
        print("  Weights loaded. Optimizer/scheduler/epoch start fresh.\n")

    # Return only the tracker (with backbone) for training; discard the detector.
    return tracker


def freeze_and_collect_trainable(model, fusion_lr, decoder_lr=None):
    """Freeze everything; unfreeze dino_encoder.proj + cross_attention_fuser (and optionally
    sam_mask_decoder).  Returns a list of AdamW param-group dicts."""
    for param in model.parameters():
        param.requires_grad = False

    def _collect(keywords):
        params, names = [], []
        for name, param in model.named_parameters():
            if any(kw in name for kw in keywords):
                param.requires_grad = True
                params.append(param)
                names.append(name)
        return params, names

    fusion_params,  fusion_names  = _collect(["dino_encoder.proj", "cross_attention_fuser"])
    decoder_params, decoder_names = _collect(["sam_mask_decoder"]) if decoder_lr is not None else ([], [])

    total = sum(p.numel() for p in model.parameters())
    total_trainable = 0

    print(f"\nFusion trainable parameters ({len(fusion_names)}):")
    for name in fusion_names:
        p = dict(model.named_parameters())[name]
        total_trainable += p.numel()
        print(f"  {name}: {list(p.shape)}")

    if decoder_params:
        print(f"\nDecoder trainable parameters ({len(decoder_names)}):")
        for name in decoder_names:
            p = dict(model.named_parameters())[name]
            total_trainable += p.numel()
            print(f"  {name}: {list(p.shape)}")

    print(f"Total trainable: {total_trainable:,} / {total:,} ({total_trainable / total:.4%})\n")

    param_groups = [{"params": fusion_params, "lr": fusion_lr}]
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": decoder_lr})
    return param_groups


# ---------------------------------------------------------------------------
# Forward pass (propagation path, no-memory)
# ---------------------------------------------------------------------------

def forward_single_frame(tracker, images, masks):
    """Propagation-path forward with no-memory fallback + DINO fusion + SAM decoder.

    Mirrors tracking inference: _prepare_backbone_features → no-memory pix_feat
    → DINO cross-attention fusion → _forward_sam_heads (with one GT click prompt).

    Args:
        tracker: Sam3TrackerPredictor with backbone and dino_encoder/cross_attention_fuser
                 attached.
        images: [B, 3, SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE] SAM3-normalized tensor.
        masks:  [B, 1, SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE] binary float GT masks.

    Returns:
        high_res_masks: [B, 1, SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE] mask logits (before sigmoid).
    """
    from sam3.model.sam3_tracker_utils import get_next_point

    B = images.shape[0]

    # 1. Frozen SAM3 backbone → (backbone_out, vision_feats, vision_pos_embeds, feat_sizes)
    #    vision_feats: list of (HW, B, C) tensors; last entry is the coarsest level
    #    feat_sizes for SAM3 at 1008px: [(288, 288), (144, 144), (72, 72)]  (approx)
    with torch.no_grad():
        backbone_out_raw = tracker.forward_image(images)
        _, vision_feats, _, feat_sizes = tracker._prepare_backbone_features(backbone_out_raw)

    C = tracker.hidden_dim       # 256
    H, W = feat_sizes[-1]        # 72, 72 for SAM3 at 1008px

    # 2. No-memory fallback: raw top-level vision features [B, C, H, W]
    pix_feat = vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)

    # 3. DINO cross-attention fusion — gradients flow through proj + fuser
    dino_feats = tracker.dino_encoder(images)                        # [B, N_patches, 256]
    pix_feat   = tracker.cross_attention_fuser(pix_feat, dino_feats) # [B, C, H, W]

    # 4. High-res features for the SAM decoder (all levels except the coarsest)
    high_res_features = (
        [
            x.permute(1, 2, 0).view(B, x.shape[2], *s)
            for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
        ]
        if len(vision_feats) > 1
        else None
    )

    # 5. Sample one GT positive click per image as the prompt
    gt_bool = masks.bool()   # [B, 1, H_img, W_img]
    points, labels = get_next_point(gt_masks=gt_bool, pred_masks=None, method="uniform")
    point_inputs = {
        "point_coords": points.to(images.device),   # [B, 1, 2]  (x, y absolute pixels)
        "point_labels": labels.to(images.device),   # [B, 1]  int32
    }

    # 6. Frozen SAM decoder: returns 7-tuple
    #    (low_res_multimasks, high_res_multimasks, ious,
    #     low_res_masks, high_res_masks, obj_ptr, object_score_logits)
    _, _, _, _, high_res_masks, _, _ = tracker._forward_sam_heads(
        backbone_features=pix_feat,
        point_inputs=point_inputs,
        mask_inputs=None,
        high_res_features=high_res_features,
        multimask_output=False,
    )
    # high_res_masks: [B, 1, SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE] logits
    return high_res_masks


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(tracker, dataloader, optimizer, scaler, device, epoch, args):
    tracker.train()
    # Keep all frozen submodules in eval mode (BN/Dropout stats unchanged)
    tracker.backbone.eval()
    tracker.transformer.eval()
    tracker.maskmem_backbone.eval()
    if args.decoder_lr is None:
        tracker.sam_mask_decoder.eval()
    tracker.sam_prompt_encoder.eval()
    if tracker.dino_encoder is not None:
        tracker.dino_encoder.backbone.eval()

    epoch_loss = 0.0
    n_batches  = 0

    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_logits = forward_single_frame(tracker, images, masks)
            loss = combined_loss(pred_logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in tracker.parameters() if p.requires_grad], max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        n_batches  += 1

        if batch_idx % args.log_every == 0:
            print(f"  [Epoch {epoch+1}] Batch {batch_idx}/{len(dataloader)}  Loss: {loss.item():.4f}")

    return epoch_loss / max(n_batches, 1)


def validate_epoch(tracker, dataloader, device):
    tracker.eval()
    epoch_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks  = masks.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_logits = forward_single_frame(tracker, images, masks)
                loss = combined_loss(pred_logits, masks)
            epoch_loss += loss.item()
            n_batches  += 1

    return epoch_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train DINOv3 fusion for SAM3")
    parser.add_argument("--checkpoint",   type=str,   default=None,
                        help="Path to SAM3 checkpoint (auto-downloads from HuggingFace if omitted).")
    parser.add_argument("--dino_model",   type=str,
                        default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--lr",           type=float, default=1e-4,
                        help="LR for dino_encoder.proj and cross_attention_fuser.")
    parser.add_argument("--decoder_lr",   type=float, default=None,
                        help="If set, also fine-tune sam_mask_decoder with this LR "
                             "(independent of --lr; default: decoder frozen).")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size",   type=int,   default=2)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--log_every",    type=int,   default=10)
    parser.add_argument("--save_every",   type=int,   default=5)
    parser.add_argument("--val_split",    type=float, default=0.15,
                        help="Fraction of combined dataset used for validation (default: 0.15).")
    parser.add_argument("--patience",     type=int,   default=5,
                        help="Early stopping: stop after this many epochs with no val improvement.")
    parser.add_argument("--init_weights", type=str,   default=None,
                        help="Path to a dino fusion checkpoint to load initial weights from "
                             "(optimizer/scheduler/epoch are NOT restored — training starts fresh).")
    parser.add_argument("--output_dir",   type=str,
                        default="/home/marcol01/sam3/checkpoints_dino_fusion")
    # Dataset selection
    parser.add_argument("--datasets", nargs="+",
                        choices=["moca", "cod10k", "camo"],
                        default=["moca", "cod10k", "camo"],
                        help="Which datasets to train on (default: all three).")
    # Dataset paths
    parser.add_argument("--moca_frames", type=str, default="/Experiments/marcol01/frames_train")
    parser.add_argument("--moca_masks",  type=str, default="/Experiments/marcol01/masks_train")
    parser.add_argument("--cod10k_root", type=str, default="/Experiments/marcol01/COD10K-v3")
    parser.add_argument("--camo_root",   type=str, default="/Experiments/marcol01/CAMO")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Building SAM3 tracker + DINOv3 fusion model...")
    tracker      = build_model(args)
    param_groups = freeze_and_collect_trainable(tracker, args.lr, args.decoder_lr)
    tracker      = tracker.to(device)

    print(f"\nLoading datasets (selected: {', '.join(args.datasets)})...")
    datasets = []

    if "moca" in args.datasets:
        if os.path.isdir(args.moca_frames) and os.path.isdir(args.moca_masks):
            datasets.append(MoCADataset(args.moca_frames, args.moca_masks))
        else:
            print(f"[MoCA]   skipped (paths not found)")
    else:
        print(f"[MoCA]   skipped (not selected)")

    if "cod10k" in args.datasets:
        if os.path.isdir(args.cod10k_root):
            datasets.append(COD10KDataset(args.cod10k_root))
        else:
            print(f"[COD10K] skipped (path not found: {args.cod10k_root})")
    else:
        print(f"[COD10K] skipped (not selected)")

    if "camo" in args.datasets:
        if os.path.isdir(args.camo_root):
            datasets.append(CAMODataset(args.camo_root))
        else:
            print(f"[CAMO]   skipped (path not found: {args.camo_root})")
    else:
        print(f"[CAMO]   skipped (not selected)")

    if not datasets:
        print("ERROR: no datasets found. Check --moca_frames / --cod10k_root / --camo_root")
        sys.exit(1)

    combined_ds = ConcatDataset(datasets)
    total = sum(len(d) for d in datasets)

    n_val   = max(1, int(total * args.val_split))
    n_train = total - n_val
    train_ds, val_ds = random_split(
        combined_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"\nCombined: {total} samples  →  train: {n_train}  val: {n_val}")

    dataloader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )
    scaler = torch.amp.GradScaler("cuda")

    print(f"\nStarting training for {args.epochs} epochs")
    print(f"  Fusion LR: {args.lr}  Decoder LR: {args.decoder_lr}  "
          f"Weight decay: {args.weight_decay}  Batch size: {args.batch_size}")
    print(f"  Val split: {args.val_split}  Early stopping patience: {args.patience}")
    print(f"  Output dir: {args.output_dir}\n")

    best_val_loss     = float("inf")
    epochs_no_improve = 0
    train_history     = []
    val_history       = []

    for epoch in range(args.epochs):
        t0       = time.time()
        avg_loss = train_epoch(tracker, dataloader, optimizer, scaler, device, epoch, args)
        val_loss = validate_epoch(tracker, val_loader, device)
        scheduler.step()
        elapsed  = time.time() - t0
        lr_now   = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{args.epochs} — "
            f"Train: {avg_loss:.4f}  Val: {val_loss:.4f} — "
            f"LR: {lr_now:.6f} — Time: {elapsed:.1f}s"
        )

        # Loss curve (train + val)
        train_history.append(avg_loss)
        val_history.append(val_loss)
        epochs_x = range(1, len(train_history) + 1)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs_x, train_history, marker="o", markersize=3, linewidth=1.5, label="Train")
        ax.plot(epochs_x, val_history,   marker="s", markersize=3, linewidth=1.5, label="Val")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title("DINOv3 Fusion Training Loss (SAM3)")
        ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "loss_curve.png"), dpi=120)
        plt.close(fig)

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.output_dir, f"dino_fusion_epoch{epoch+1:03d}.pt")
            ckpt = {
                "epoch":                   epoch + 1,
                "dino_encoder":            tracker.dino_encoder.state_dict(),
                "cross_attention_fuser":   tracker.cross_attention_fuser.state_dict(),
                "optimizer":               optimizer.state_dict(),
                "scheduler":               scheduler.state_dict(),
                "train_loss":              avg_loss,
                "val_loss":                val_loss,
            }
            if args.decoder_lr is not None:
                ckpt["sam_mask_decoder"] = tracker.sam_mask_decoder.state_dict()
            torch.save(ckpt, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            best_path = os.path.join(args.output_dir, "dino_fusion_best.pt")
            best_ckpt = {
                "epoch":                   epoch + 1,
                "dino_encoder":            tracker.dino_encoder.state_dict(),
                "cross_attention_fuser":   tracker.cross_attention_fuser.state_dict(),
                "train_loss":              avg_loss,
                "val_loss":                val_loss,
            }
            if args.decoder_lr is not None:
                best_ckpt["sam_mask_decoder"] = tracker.sam_mask_decoder.state_dict()
            torch.save(best_ckpt, best_path)
            print(f"  New best model (val={val_loss:.4f}) saved: {best_path}")
        else:
            epochs_no_improve += 1
            print(f"  No val improvement for {epochs_no_improve}/{args.patience} epochs")
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}.")
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
