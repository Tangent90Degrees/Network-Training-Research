# Project status

## 2026-11-24 · Airbench methods on reconstruction

The previous full whitening comparison is available under `experiments/cnn_reconstruction/outputs/seed42`. All three runs completed 20 epochs on an RTX 4090. Their final validation errors were 0.011949 (random first layer), 0.011753 (trainable whitening), and 0.011762 (frozen whitening). All three used Dirac in the encoder.

Added the opt-in paper suite with a matched random-flip baseline, five individual training treatments, their combination, and paired single-view and six-view evaluation. See the [run command, implementation choices and code walkthrough](experiments/cnn_reconstruction/README.md).

Verified all seven conditions in COMP3211 on CPU using `--suite paper --epochs 4 --batch-size 3 --limit 7` with a fresh temporary output directory. All 28 epoch records contained finite single-view and six-view errors. Additional checks covered spatial alignment including borders, alternating orientation by image, the 64× bias learning rate, and Lookahead interpolation and final synchronization. A legacy-mode one-epoch run reproduced the earlier smoke-test errors. Full paper-suite training and CUDA verification are pending.

## 2026-11-23 · CNN reconstruction extension

Implemented the three-condition CIFAR-10 reconstruction comparison and its [run instructions and walkthrough](experiments/cnn_reconstruction/README.md). This extends the classification project with a pixel regression task using self-supervision. The local reference checkout was empty, so the encoder layout was checked against the authors’ published Airbench 94 implementation.

Verified on CPU in COMP3211 with PyTorch 2.6.0 and torchvision 0.21.0. The command below exercises all three conditions, two epochs, and an uneven final batch.

```shell
python experiments/cnn_reconstruction/run.py --epochs 2 --batch-size 3 --limit 7 --output <new-directory>
```

The real-data smoke test used seven training images and seven held-out images. All six epoch records had finite error, elapsed times increased, and the PNG passed image validation. Additional checks confirmed 512 hidden pixels per channel, reproducible masks, the expected error on a constant prediction, opposite-sign whitening filters, and the expected frozen versus trainable weight updates. Outputs were written to a temporary directory. Full training, CUDA verification, and scientific comparison remain pending.
