"""Generate ViT architecture figure with Tensorgami."""
import sys
sys.path.insert(0, "/data2/sliudu/projects/Network-Training/experiments")

import torch
import tensorgami as tg
from common.vit import ViT

model = ViT(
    img_size=32, patch_size=4, in_chans=3, num_classes=10,
    embed_dim=192, depth=9, num_heads=4, mlp_ratio=4.0,
    drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
)
model.eval()

example_input = torch.zeros(1, 3, 32, 32)

fig = tg.draw(
    model,
    example_input,
    level="stages",          # top-level sub-modules: clean high-level view
    direction="LR",
    width="6.5in",           # full text width of the paper
    variant="greyscale",     # black-and-white for print
    palette="mono",
)

out = "/data2/sliudu/projects/Network-Training/reports/vit_arch.pdf"
fig.save(out, dpi=300)
print(f"Saved: {out}")
