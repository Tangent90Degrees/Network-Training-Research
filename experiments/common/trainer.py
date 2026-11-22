"""Shared training loop for all ViT experiments."""

import os
import math
import time
import json
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR


def mixup_cutmix_batch(x, y, num_classes, mixup_alpha=0.8, cutmix_alpha=1.0,
                       mixup_prob=0.5, cutmix_prob=0.5):
    """Apply Mixup or CutMix to a batch. Returns (mixed_x, mixed_y_soft).

    mixed_y_soft is a float tensor [B, num_classes] (soft labels).
    Both are applied with independent probabilities; if neither fires the
    batch is returned as-is with one-hot hard labels.
    """
    B, C, H, W = x.shape
    y_soft = torch.zeros(B, num_classes, device=x.device).scatter_(
        1, y.unsqueeze(1), 1.0)

    r = torch.rand(2)
    if r[0] < cutmix_prob and cutmix_alpha > 0:
        # CutMix
        lam = float(torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample())
        perm = torch.randperm(B, device=x.device)
        cut_ratio = math.sqrt(1. - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)
        cy = torch.randint(H, (1,)).item()
        cx = torch.randint(W, (1,)).item()
        y1 = max(cy - cut_h // 2, 0)
        y2 = min(cy + cut_h // 2, H)
        x1 = max(cx - cut_w // 2, 0)
        x2 = min(cx + cut_w // 2, W)
        lam_actual = 1. - (y2 - y1) * (x2 - x1) / (H * W)
        x_mix = x.clone()
        x_mix[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        y_soft = lam_actual * y_soft + (1 - lam_actual) * y_soft[perm]
        return x_mix, y_soft
    elif r[1] < mixup_prob and mixup_alpha > 0:
        # Mixup
        lam = float(torch.distributions.Beta(mixup_alpha, mixup_alpha).sample())
        perm = torch.randperm(B, device=x.device)
        x_mix = lam * x + (1 - lam) * x[perm]
        y_soft = lam * y_soft + (1 - lam) * y_soft[perm]
        return x_mix, y_soft

    return x, y_soft


def train_one_epoch(model, loader, optimizer, criterion, device, scaler,
                    num_classes=10, use_mixup_cutmix=False,
                    mixup_alpha=0.8, cutmix_alpha=1.0):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if use_mixup_cutmix:
            x, y_soft = mixup_cutmix_batch(x, y, num_classes,
                                           mixup_alpha=mixup_alpha,
                                           cutmix_alpha=cutmix_alpha)
        else:
            y_soft = None

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            logits = model(x)
            if y_soft is not None:
                # Soft-label cross-entropy: -sum(y_soft * log_softmax(logits))
                log_p = torch.nn.functional.log_softmax(logits, dim=-1)
                loss = -(y_soft * log_p).sum(dim=-1).mean()
            else:
                loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        # Accuracy: compare hard prediction to original hard labels
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return total_loss / n, correct / n


def run_experiment(model, train_loader, val_loader, test_loader,
                   epochs: int, lr: float, weight_decay: float,
                   output_dir: str, device: str = 'cuda',
                   optimizer_cls=None, label_smoothing: float = 0.1,
                   use_lookahead: bool = False,
                   scheduler_type: str = 'onecycle',
                   use_mixup_cutmix: bool = False,
                   mixup_alpha: float = 0.8,
                   cutmix_alpha: float = 1.0):
    """Full training run; saves logs and best checkpoint.

    Args:
        model:             nn.Module on CPU (moved to device here)
        train_loader:      DataLoader
        val_loader:        DataLoader
        test_loader:       DataLoader
        epochs:            number of epochs
        lr:                peak learning rate
        weight_decay:      AdamW weight decay
        output_dir:        where to write logs/checkpoints
        device:            'cuda' or 'cpu'
        optimizer_cls:     if None, uses AdamW
        label_smoothing:   CrossEntropyLoss smoothing (for eval loss)
        use_lookahead:     wrap optimizer with Lookahead
        scheduler_type:    'onecycle' or 'cosine'
        use_mixup_cutmix:  apply Mixup+CutMix during training
        mixup_alpha:       Beta distribution alpha for Mixup
        cutmix_alpha:      Beta distribution alpha for CutMix
    """
    os.makedirs(output_dir, exist_ok=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if optimizer_cls is None:
        base_opt = torch.optim.AdamW(model.parameters(), lr=lr,
                                     weight_decay=weight_decay)
    else:
        base_opt = optimizer_cls(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)

    if use_lookahead:
        from tricks import Lookahead
        optimizer = Lookahead(base_opt, k=6, alpha=0.5)
    else:
        optimizer = base_opt

    if scheduler_type == 'cosine':
        # Linear warmup (5% of steps) then cosine decay to lr/100.
        # Steps the *inner* optimizer so Lookahead sees a smooth LR envelope.
        warmup_steps = int(0.05 * epochs * len(train_loader))
        total_steps  = epochs * len(train_loader)
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(base_opt, lr_lambda)
    else:
        scheduler = OneCycleLR(
            base_opt,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos',
        )

    scaler = torch.amp.GradScaler('cuda')
    log = []
    best_val_acc = 0.0
    best_ckpt = os.path.join(output_dir, 'best.pt')

    total_start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            num_classes=10, use_mixup_cutmix=use_mixup_cutmix,
            mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha)
        scheduler.step()
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        row = dict(epoch=epoch, tr_loss=tr_loss, tr_acc=tr_acc,
                   val_loss=val_loss, val_acc=val_acc, epoch_sec=elapsed)
        log.append(row)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt)

        print(f"[{epoch:3d}/{epochs}] "
              f"tr {tr_acc:.4f} | val {val_acc:.4f} | "
              f"best {best_val_acc:.4f} | {elapsed:.1f}s", flush=True)

    total_time = time.time() - total_start
    print(f"\nTraining done in {total_time:.1f}s  ({total_time/60:.1f} min)")

    # Test evaluation with best checkpoint
    model.load_state_dict(torch.load(best_ckpt, map_location=device,
                                     weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test accuracy (best val ckpt): {test_acc:.4f}")

    results = dict(log=log, best_val_acc=best_val_acc,
                   test_acc=test_acc, total_time_sec=total_time)
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results
