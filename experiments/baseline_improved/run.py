"""Improved baseline: DropPath + RandAugment + Mixup/CutMix + 300 epochs."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

from data    import load_cifar10, make_loaders
from vit     import ViT, count_parameters
from trainer import run_experiment

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'Cifar-10')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')

DEVICE       = 'cuda:0'
EPOCHS       = 300
BATCH_SIZE   = 256
LR           = 1e-3       # higher LR suits stronger augmentation
WEIGHT_DECAY = 0.05

if __name__ == '__main__':
    print("=== Improved Baseline: DropPath + RandAugment + Mixup/CutMix + 300 epochs ===")
    x_train, y_train, x_test, y_test = load_cifar10(DATA_DIR)
    train_loader, val_loader, test_loader = make_loaders(
        x_train, y_train, x_test, y_test,
        batch_size=BATCH_SIZE,
        use_randaugment=True)

    # drop_path_rate=0.1: linearly scaled from 0 to 0.1 across 9 blocks
    model = ViT(drop_path_rate=0.1)
    print(f"Parameters: {count_parameters(model):,}")

    run_experiment(model, train_loader, val_loader, test_loader,
                   epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
                   output_dir=OUTPUT_DIR, device=DEVICE,
                   scheduler_type='cosine',
                   use_mixup_cutmix=True,
                   mixup_alpha=0.8,
                   cutmix_alpha=1.0,
                   label_smoothing=0.1)
