"""Aggregate results across experiments and plot comparison curves."""

import os
import sys
import json
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENTS_ROOT = os.path.join(os.path.dirname(__file__), '..')


def load_all_results():
    """Find every experiments/*/outputs/results.json and return a dict."""
    out = {}
    for path in sorted(glob.glob(os.path.join(EXPERIMENTS_ROOT, '*', 'outputs', 'results.json'))):
        name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        with open(path) as f:
            out[name] = json.load(f)
    return out


def plot_comparison(save_path):
    results = load_all_results()
    if not results:
        print("No results found yet.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, res in results.items():
        epochs = [r['epoch'] for r in res['log']]
        train_acc = [r['tr_acc'] for r in res['log']]
        val_acc   = [r['val_acc'] for r in res['log']]
        axes[0].plot(epochs, train_acc, label=name, alpha=0.8)
        axes[1].plot(epochs, val_acc,   label=name, alpha=0.8)

    axes[0].set_title('Training accuracy')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].set_title('Validation accuracy')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved comparison plot -> {save_path}")


def print_summary():
    results = load_all_results()
    if not results:
        print("No results found yet.")
        return
    print(f"\n{'Experiment':<24} {'Best Val Acc':>12} {'Test Acc':>10} {'Time (min)':>11}")
    print("-" * 60)
    for name, res in results.items():
        print(f"{name:<24} {res['best_val_acc']*100:>11.2f}% "
              f"{res['test_acc']*100:>9.2f}% {res['total_time_sec']/60:>10.1f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', default=None,
                        help='path to save comparison plot')
    args = parser.parse_args()
    print_summary()
    if args.plot:
        plot_comparison(args.plot)
