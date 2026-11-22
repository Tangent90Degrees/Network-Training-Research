"""All-methods v3: improved recipe + all four tricks (with v2 fixes)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

import torch
from data    import load_cifar10, make_loaders
from vit     import ViT, count_parameters
from tricks  import init_whitening_patch_embed, ViTWithScaleBias
from trainer import run_experiment

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'Cifar-10')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')

DEVICE       = 'cuda:2'
EPOCHS       = 300
BATCH_SIZE   = 256
LR           = 1e-3
WEIGHT_DECAY = 0.05
ALPHA        = 0.3   # soft-Dirac blend


def init_soft_dirac_attention(vit_model, alpha=0.3):
    for blk in vit_model.blocks:
        attn = blk.attn
        D = attn.proj.weight.shape[0]
        with torch.no_grad():
            qkv_w = attn.qkv.weight
            for k in range(3):
                sub = qkv_w[k*D:(k+1)*D]
                sub.mul_(1 - alpha).add_(alpha * torch.eye(D, device=sub.device))
            op = attn.proj.weight
            op.mul_(1 - alpha).add_(alpha * torch.eye(D, device=op.device))
    print(f"[soft-dirac] alpha={alpha}")


if __name__ == '__main__':
    print("=== All-methods v3: improved recipe + all tricks (v2 fixes) ===")
    x_train, y_train, x_test, y_test = load_cifar10(DATA_DIR)
    train_loader, val_loader, test_loader = make_loaders(
        x_train, y_train, x_test, y_test,
        batch_size=BATCH_SIZE, use_randaugment=True,
        alternating_flip=True)

    base_vit = ViT(drop_path_rate=0.1)
    # 1. Frozen patch-whitening
    init_whitening_patch_embed(base_vit.patch_embed, x_train, eps=5e-4, freeze=True)
    # 2. Soft-Dirac init (v2 fix: alpha=0.3 instead of full identity)
    init_soft_dirac_attention(base_vit, alpha=ALPHA)
    # 3. ScaleBias head (v2 fix: scale=1.0 instead of 1/9)
    model = ViTWithScaleBias(base_vit, embed_dim=192, num_classes=10,
                             init_scale=1.0)
    print(f"Parameters: {count_parameters(model):,}")

    # 4. Lookahead + cosine LR (v2 fix), alternating flip already in loader
    run_experiment(model, train_loader, val_loader, test_loader,
                   epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
                   output_dir=OUTPUT_DIR, device=DEVICE,
                   scheduler_type='cosine',
                   use_lookahead=True,
                   use_mixup_cutmix=True, label_smoothing=0.1)
