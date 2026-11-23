# Project status

## 2026-11-23 · CNN reconstruction extension

Implemented the three-condition CIFAR-10 reconstruction comparison and its [run instructions and walkthrough](experiments/cnn_reconstruction/README.md). This extends the classification project with a pixel regression task using self-supervision. The local reference checkout was empty, so the encoder layout was checked against the authors’ published Airbench 94 implementation.

Verified on CPU in COMP3211 with PyTorch 2.6.0 and torchvision 0.21.0. The command below exercises all three conditions, two epochs, and an uneven final batch.

```shell
python experiments/cnn_reconstruction/run.py --epochs 2 --batch-size 3 --limit 7 --output <new-directory>
```

The real-data smoke test used seven training images and seven held-out images. All six epoch records had finite error, elapsed times increased, and the PNG passed image validation. Additional checks confirmed 512 hidden pixels per channel, reproducible masks, the expected error on a constant prediction, opposite-sign whitening filters, and the expected frozen versus trainable weight updates. Outputs were written to a temporary directory. Full training, CUDA verification, and scientific comparison remain pending.
