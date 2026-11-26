# Presentation Outline & Script

**4-minute presentation — 9 slides**

---

## Slide 1 — Title (0:00–0:15)

**Bullets:**

- Do CNN Speed-Up Tricks Transfer to Transformers?
- Authors: Siwei LIU, Xingjian ZHANG, Liwei LI
- COMP-4471 · HKUST

**Script:**
> "Hi everyone. Our project asks a simple question: the tricks that made a CNN train to 94% accuracy in 3 seconds — do they still work if you swap the CNN for a Vision Transformer, or change the task to reconstruction? Let's find out."

---

## Slide 2 — Motivation & Gap (0:15–0:40)

**Bullets:**

- Garg et al. (2024): 4 tricks → 94% on CIFAR-10 in 3.29 s
- Tricks: frozen patch-whitening, Dirac init, ScaleBias + Lookahead, alternating flip
- Gap: never tested on transformers or reconstruction tasks

**Script:**
> "Garg et al. proposed four training tricks for CNN-based CIFAR-10 classification: frozen patch whitening, Dirac init, ScaleBias plus Lookahead, and alternating flip. These cut training to 3.29 seconds. But nobody has tested them on a transformer architecture or on a reconstruction task. That's the gap we fill."

---

## Slide 3 — Our ViT & Two Recipes (0:40–1:00)

**Bullets:**

- Small ViT: patch 4×4, dim 192, depth 9, heads 4, ~4M params
- Weak recipe: 100 epochs, crop+flip — baseline 61.7%
- Strong recipe: 300 epochs, RandAugment + Mixup/CutMix + DropPath — baseline 81.4%

**Script:**
> "We use a small ViT with 4 million parameters. To test fairly we run two training recipes: a weak one matching the CNN's setup, and a strong one with RandAugment, Mixup, CutMix, and DropPath — the state-of-the-art for ViTs on small datasets."

---

## Slide 4 — The Four Tricks (1:00–1:20)

**Bullets:**

- Whitening: $W_k = \mathbf{v}_k / \sqrt{\lambda_k + \varepsilon}$, frozen
- Dirac: $\mathbf{W}_{QKV} = \mathbf{I}_D$; soft: $\mathbf{W} \leftarrow \alpha\mathbf{I} + (1-\alpha)\mathbf{W}_\text{rand}$
- ScaleBias: $y_i = s_i x_i + b_i$; Lookahead: $\phi \leftarrow \phi + \alpha(\theta - \phi)$
- Alternating flip: deterministic 50/50 assignment (always flip odd-indexed samples)

**Script:**
> "Here are the four tricks adapted for ViT. Whitening initialises the patch embedding with PCA filters of the training patches and freezes them. Dirac init sets attention projections to identity. ScaleBias replaces the linear head with a learnable affine layer, wrapped with the Lookahead optimizer. And alternating flip replaces random flips with a deterministic 50/50 assignment."

---

## Slide 5 — Classification Results (1:20–2:00)

**Bullets:**

- Weak recipe: whitening +1.3 pp, alt flip +0.8 pp ✓ / Dirac −7.9 pp, SBL −16.0 pp ✗
- Strong recipe: whitening flips to −4.4 pp / soft Dirac only net-positive at +0.4 pp
- Same trick, same model — different recipe, different outcome

**Script:**
> "Here are the results. Under the weak recipe, whitening and alternating flip help; Dirac and ScaleBias-Lookahead hurt. Under the strong recipe — same tricks, same model — the picture changes. Whitening flips from plus 1.3 to minus 4.4 percentage points. Soft Dirac becomes the only net-positive trick. The same trick can help or hurt depending on the recipe."

---

## Slide 6 — Why? The Whitening Conflict (2:00–2:25)

**Bullets:**

- Whitening basis computed from **clean** images, then frozen
- Mixup/CutMix: blended inputs ≠ frozen basis → mismatch
- Reconstruction experiment (no augmentation): frozen ≈ trainable whitening — freezing is not the problem

**Script:**
> "Why does whitening flip sign? The filters are computed from clean images and frozen. Under strong augmentation, Mixup and CutMix blend images at the pixel level — the inputs no longer match the frozen basis. Our reconstruction experiment proves it: without augmentation, frozen and trainable whitening are identical, both beating the baseline. Freezing is not the problem. The mismatch is."

---

## Slide 7 — Reconstruction Results (2:25–3:00)

**Bullets:**

- Task: predict masked pixels (32 of 64 blocks), no strong augmentation
- All tricks improve over baseline
- Lookahead: best single trick at −5.1% MSE
- All methods combined: −9.3% MSE

**Script:**
> "Now the reconstruction task. Same four tricks, same CNN, but the task is predicting masked pixels with no strong augmentation. Every single trick improves over the baseline, and they compose well — all methods together give a 9.3% MSE reduction. Lookahead is the strongest single trick at minus 5%, confirming that it only hurt in ViT classification because of the decaying LR conflict, not the architecture."

---

## Slide 8 — Two-Axis Framework (3:00–3:30)

**Bullets:**

- Axis 1 — **Architectural compatibility**: does the trick assume convolutional structure?
- Axis 2 — **Recipe compatibility**: does the trick depend on augmentation or LR schedule?
- Failures are situational, not fundamental — fix the mismatch and every trick works

**Script:**
> "Putting it all together: trick transferability sits on two axes. Architecture compatibility — does the trick assume convolutional structure? And recipe compatibility — does the trick depend on augmentation or schedule? When you remove the mismatch, every trick works. The failures are situational, not fundamental."

---

## Slide 9 — Conclusion & Future Work (3:30–4:00)

**Bullets:**

- Key finding: transferability = trick × recipe
- Future: learnable whitening, decayed soft Dirac, Lookahead + flat LR
- Rule: test each trick under the **full** target recipe before adopting it

**Script:**
> "The take-home message: before porting a CNN training trick to a new architecture or task, test it under the full target recipe. Our next steps are learnable whitening to fix the augmentation conflict, decayed soft Dirac, and Lookahead with a flat learning rate. Thank you."
