# CNN masked reconstruction

Run from the repository root in an environment with PyTorch, torchvision, NumPy, and Matplotlib. Supply a fresh output directory for each comparison.

```shell
python experiments/cnn_reconstruction/run.py --output experiments/cnn_reconstruction/outputs/seed42
```

The default comparison trains each condition for 20 epochs on 45,000 CIFAR-10 training images. A fixed split holds out 5,000 for validation. Each target is the original RGB image in [0, 1]. Each input hides half its pixels in 4 × 4 squares filled with the training-set channel means. The training masks change on each pass. Validation reuses the same masks across epochs and variants. For a quick check add `--epochs 2 --batch-size 3 --limit 7`.

The encoder follows the spatial layers of the authors’ 94% CNN, including its 2 × 2 input convolution, three convolution groups and partial identity initialization inside those groups. A new convolutional decoder upsamples the final 3 × 3 feature map to 32 × 32. The baseline uses ordinary PyTorch initialization for the first convolution. Both whitening variants initialize its weights from identical training patches. The frozen condition fixes those weights. All three retain a trainable first-layer bias, use Adam at 0.001 and receive identical image orders. Masking is the only image transformation. [Authors’ reference implementation](https://github.com/KellerJordan/cifar10-airbench/blob/master/legacy/airbench94.py)

The run writes settings.json, a JSON history per variant, and comparison.png. Lower validation mean squared error (MSE) indicates better predictions on the hidden RGB values. Timing includes whitening and training and excludes validation and data loading. The runner executes the random baseline first, so initial library warm-up can affect its first timing point. Repeat full runs with multiple seeds before interpreting small differences.

## Code walkthrough

Line numbers refer to run.py. Each entry covers the listed lines in source order. Blank lines separate logical sections.

| Lines | Purpose and behavior |
|---|---|
| 1, 3–7 | Describe the experiment, then import command-line parsing, JSON serialization, module search paths, wall-clock timing, and portable file paths. |
| 9–14 | Load Matplotlib, select its headless rendering backend, and import its plotting interface. Load PyTorch, its layer classes, and functional tensor operations. |
| 16–18, 20 | Resolve the repository root from this script, add the experiments directory to the import path, and reuse the existing CIFAR statistics and loader. List the three conditions in execution order. |
| 23–27 | Define and document the model factory. Start with a stride-one 2 × 2 RGB convolution and GELU activation. Zero its bias for all comparisons. Track the 24 output channels for the next layer. |
| 28–34 | Visit the three encoder widths. Build two convolutions per group. Preserve spatial size with padding and use the reference identity initialization for the first input-channel-count filters. Append each convolution and insert pooling after the first convolution in each group. |
| 35–38 | Create batch normalization with the reference settings, fix its scale, append normalization and activation, and record the new width for the following convolution. |
| 39–44 | Visit three progressively larger output sizes. Upsample features, convolve and activate at each size, and record its output width. Project to three RGB channels and bound values with sigmoid. Return the layers as one network. |
| 47–52 | Disable gradient recording for initialization, define the in-place whitening routine, and document its inputs. Allocate a 12 × 12 second-moment accumulator and a patch counter. Process at most 5,000 training images in groups of 64 to bound memory. |
| 53–59 | Move each group to the compute device and normalize it. Flatten each overlapping RGB patch into 12 values. Sum their outer products and count patches. Decompose the average second moment, scale and reverse the eigenvector rows using regularized inverse square roots, and copy positive and negative filters into the input convolution. |
| 62–67 | Define and document the masked loss. Draw 64 random scores per image and select the 32 lowest. Mark those positions in a Boolean array. Reshape to an 8 × 8 grid and expand each cell to a 4 × 4 pixel square. |
| 68–70 | Normalize the clean inputs, fill hidden positions with zero in normalized space, and square each prediction error in original RGB units. Sum over masked values and divide by 512 pixels times three channels per image. |
| 73–77 | Define and document the timing helper. Detect CUDA, wait for queued GPU operations when applicable, and read a monotonic clock in seconds. |
| 80–86 | Define and document one condition’s training loop. Reset model randomness and create separate reproducible streams for shuffling and masking. Put channel means and standard deviations on the selected device with dimensions that broadcast across a batch. |
| 87–94 | Build the model and Adam optimizer. Start the clock. For whitening conditions, initialize the first filters and enable their gradients only for the trainable condition. Record initialization time and create the epoch history. |
| 95–103 | Iterate through the requested epochs, start timing, and enable training behavior. Shuffle all training indices and split into batches. Move each target batch to the device, clear old gradients, evaluate masked loss, differentiate it, and apply one optimizer step. |
| 104–111 | Accumulate the epoch’s training time. Enable evaluation behavior, reset the validation mask generator, and initialize the sum. Disable gradient tracking, traverse all validation batches, compute their masked errors, and weight each mean by its image count to handle a short final batch. |
| 112–115 | Build an epoch record with elapsed time and mean validation error, append it, print immediate progress, and return the completed history. |
| 118–127 | Define and document the command-line entry point. Build its parser. Register training duration, batch size, random seed, dataset path, fresh output path, and optional subset limit. Parse the supplied arguments. |
| 128–134 | Reject nonpositive sizes with a usage error. Select GPU when available. Load training images, generate a fixed split independent of the model seed, reserve its first 5,000 indices for validation, and wrap the NumPy image array in a tensor. |
| 135–138 | Select the training and validation tensors, applying the optional cap to each. Release the original image array and create a new output directory, failing if the destination already exists. |
| 139–142 | Collect epoch, batch and seed settings, actual split sizes, PyTorch version, and device identity. Write that configuration to JSON for interpreting the run later. |
| 143–147 | Iterate over conditions, train each, and write its full history before starting the next. Use cumulative seconds for horizontal coordinates and validation error for vertical coordinates, and label each curve. |
| 148–153 | Label both axes, show curve identities, fit the layout, save the comparison image, and release the figure. |
| 156–157 | Check for direct invocation and start the experiment, so importing the functions for tests does not start training. |
