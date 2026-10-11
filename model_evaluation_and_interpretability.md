# Model Evaluation & Interpretability Analysis — Step-by-Step Explainer

> Reference document for `src/main.ipynb` (BBBC039 nucleus-segmentation backbone benchmark).
> For each evaluation and interpretability block: **what it is → how the code works, step by step → what it is for → current status**.
> Cell numbers are 1-based positions in the notebook (68 cells total).

---

## 1. Map: research plan → notebook cells

| Step in the research plan | Notebook cell(s) | Status in code |
|---|---|---|
| Colab setup (GPU, pip installs, Drive mount, config fallback) | #2 | ✅ added |
| Data ingestion (BBBC039) | #3–#14 | ✅ implemented, idempotent (re-run safe) |
| Augmentation (crops, flips, noise, elastic) | #16 | ✅ wired into the training dataset |
| Multi-head targets (background/boundary/interior) + percentile normalization | #18 | ✅ implemented (RGBA-mask handling fixed) |
| DataLoaders (seeded 50/25/25 split, batch 4, transforms) | #19 | ✅ rewritten |
| BCE + Dice loss | #22 | ✅ implemented with `binary_cross_entropy_with_logits` |
| Decoder (skip fusion + final 2× upsample) | #26–#27 | ✅ fixed (`skip.shape`, `F.interpolate`) |
| 8 ResNet-family U-Nets | #29–#43 | ✅ all 8 constructed, no overwrites |
| Optimizers (encoder/decoder LR groups) | #46–#60 | ✅ `encoder.parameters()` + `decoder.parameters()` |
| Training loop (checkpoints, resume, early stopping) | #62 | ✅ implemented + regression-tested, not yet trained for real |
| Checkpointing regression tests (fresh/skip/resume/early-stop) | `tests/test_checkpointing.py` | ✅ 4 scenarios passing |
| **Evaluation: biological benchmark (count error, KS test)** | #63 | ✅ implemented + min-size filter, smoke-tested; loads from `CHECKPOINT_DIR` |
| **Evaluation: architectural comparison grid** | #64 | ✅ implemented for all 8 backbones; loads from `CHECKPOINT_DIR` |
| **Evaluation: test-set Dice / IoU (semantic)** | — | ❌ **not implemented anywhere** |
| **Evaluation: mAP across IoU thresholds (instance)** | — | ❌ **not implemented anywhere** |
| **Interpretability: Grad-CAM** | #66 | ✅ fixed & smoke-tested |
| **Interpretability: feature-map extraction** | #67 | ✅ fixed & smoke-tested |
| **Interpretability: comparative error analysis** | #68 | ✅ full rewrite (image-by-image, real filenames, grid) |
| Isolation of toxic-dose fragmentation cases | — | ❌ not implemented |

---

## 2. Model Evaluation

### 2.1 Training / validation loop — cell #62

**What it is:** the engine that trains each backbone, plus a **crash-safe experiment manager**: it checkpoints every model's full training state every epoch, resumes interrupted models, skips finished ones, and stops a model early when it stops improving.

**Checkpoint layout (set up at the top of the cell):**

- `CHECKPOINT_DIR` — the persistent folder everything is saved to. On Colab with Drive mounted this auto-resolves to `/content/drive/MyDrive/bbbc039_checkpoints`, so nothing is lost when the runtime disconnects; locally it is `checkpoints/`.
- `LOCAL_CKPT_DIR` (`checkpoints_local/`) — fast in-session storage. The full checkpoint is written here **every epoch** and copied to `CHECKPOINT_DIR` every `CKPT_SYNC_EVERY = 5` epochs, so a Colab disconnect costs at most 5 epochs of progress.
- `training_progress.json` — the **manifest** `{model_name: {status, last_epoch, best_val_loss, ...}}`. Status values: `training` (started), `resuming` (reloaded), `done` (finished). This is how you check which models are done and which can resume.
- Per model: `{name}_checkpoint.pth` (full resumable state: weights + **optimizer** + **scheduler** + epoch + loss history), `{name}_best.pth` (best weights only — what the evaluation cells load), `{name}_loss_curves.json`.

**Step by step:**

1. The manifest is loaded and a **status overview** is printed (one line per backbone: `pending` / `training` / `resuming` / `done`).
2. **Skip/resume logic** per model:
   - `status == 'done'` → the model is skipped entirely (re-running the cell is idempotent; to retrain a model, delete its manifest entry).
   - A checkpoint file exists (in-session first, then persistent) → weights, optimizer, scheduler, `start_epoch = epoch + 1`, best loss and loss history are all reloaded — training continues **exactly** where it stopped, even on a fresh Colab runtime.
   - Neither → training from scratch; the model is moved to the GPU.
3. **Epoch loop** (`range(start_epoch, NUM_EPOCHS)`): *train phase* (`model.train()`, forward, `BCE_Dice_loss`, backward, `optimizer.step()`) then *validation phase* (`model.eval()` + `torch.no_grad()`, same loss on the validation loader).
4. **Best-weights checkpoint:** if the averaged validation loss improves, `best_val_loss` is updated and the weights are saved (in-session first; synced to Drive on the next sync point).
5. **Scheduler:** `ReduceLROnPlateau(mode='min', factor=0.1, patience=3)` steps with the validation loss.
6. **Early stopping:** after `EARLY_STOPPING_PATIENCE = 10` consecutive epochs without validation improvement, training stops, the final checkpoint is synced, and the model is marked `done` with `early_stopped: true`.
7. **Full resume checkpoint every epoch** (with the periodic sync described above).
8. **Post-training:** final sync, total time, peak VRAM, loss curves saved to `CHECKPOINT_DIR`, and the manifest entry set to `done` with `time_minutes`/`peak_vram_mb`.

**Purpose:** train all 8 backbones under identical settings (same split, loss, LR policy) so the comparison is fair, while logging computational cost — and now survive real-world interruptions (Colab disconnects, quota loss, crashes) without losing work: re-running the cell resumes instead of restarting.

**Status:** implemented; real training intentionally never run yet. The logic is covered by regression tests (`tests/test_checkpointing.py`, 4 scenarios: fresh run → all `done`; re-run → all skipped; simulated Colab restart → `resnet50` resumes at the correct epoch from the persistent checkpoint; early stopping → triggers and is recorded). Run with `python tests/test_checkpointing.py`.

---

### 2.2 Final biological & hardware benchmark — cell #63

**What it is:** the end-of-experiment comparison on the metrics a biologist actually cares about: cell counts and morphology distributions.

**The multi-head boundary-subtraction pipeline (how instances are found):**

The network outputs **3 channels per pixel**: background / boundary / interior (channel order is fixed when the target is built in cell #18). To turn the two predicted channels into discrete cells **without** watershed:

```
pred_cores = (interior_prob > 0.5)  &  ~(boundary_prob > 0.5)
```

- A pixel must be confidently *interior* **and** confidently *not boundary* to be part of a cell core.
- Boundary pixels are thereby **subtracted out**, which carves a gap between touching nuclei.
- `skimage.measure.label(pred_cores)` then assigns a unique integer ID to each connected core → **each ID = one detected nucleus**.

**Step by step:**

1. For each model: load `{name}_best.pth` (the best epoch's state), `model.eval()`, reset the VRAM counter and timer.
2. Iterate the **test** dataloader under `torch.no_grad()`: forward pass → `torch.softmax(preds, dim=1)` → move to CPU/numpy.
3. Per image in the batch:
   - **Predictions:** `boundary_prob = probs[1]`, `interior_prob = probs[2]` → build `pred_cores` as above → `label(..., return_num=True)` gives `pred_count` → `regionprops` extracts every cell's `area` and `eccentricity`.
   - **Ground truth:** same logic on the one-hot target (`masks_np[b,1]` = boundary, `masks_np[b,2]` = interior).
4. Accumulate: total predicted count, total GT count, and the full predicted/GT lists of areas and eccentricities.
5. **Metrics printed per model:**
   - *Hardware:* inference time (s), peak VRAM (MB).
   - *Absolute Cell Count Error:* `|pred_count − gt_count|` — the number a biologist would immediately notice if wrong.
   - *Morphological variance:* `scipy.stats.ks_2samp(pred_areas, gt_areas)` and the same for eccentricities. The **KS statistic** is the maximum vertical distance between the predicted and GT empirical CDFs — **lower = the model's morphology distribution matches human annotation more closely**.

**Purpose:** this is the "final deliverable" evaluation — a model can score 99% pixel accuracy while merging touching nuclei into one blob; count error and the KS test on area/eccentricity catch exactly that failure mode, because they measure the geometry a downstream phenotypic analysis consumes.

**Status:** implemented with all channel indices consistent with cell #18; never executed end-to-end (needs `.pth` files). Fixed: a **minimum-size filter** (`MIN_NUCLEUS_AREA = 25`, applied to both predicted and GT `regionprops`) removes single-pixel speckles that would inflate counts. Loads `{name}_best.pth` from `CHECKPOINT_DIR` (the Drive-backed folder on Colab). ✅ **Smoke-tested** on real data with a stand-in prediction: 456 GT nuclei correctly counted in one batch, KS statistic computed (finite, in [0,1]).

---

### 2.3 Architectural comparison grid — cell #64

**What it is:** a single high-resolution figure showing every backbone's prediction on the same unseen test crop, side by side with the human annotation.

**Step by step:**

1. `next(iter(test_dataloader))` grabs one batch; `images[0]` / `masks[0]` select its first image.
2. The GT one-hot mask is converted to RGB for Matplotlib: **red = boundary, green = interior, black = background**.
3. A figure is created with `2 + len(models)` columns: original grayscale, GT, then one column per model.
4. For each model: load `{name}_best.pth` → eval → forward on the single image → softmax → threshold channels 1 and 2 at 0.5 → build the same RGB map → plot.
5. Saved as `BBBC039_Architectural_Comparison.png` at 300 dpi.

**Purpose:** the first-glance, human-readable ranking — a qualitative complement to the numeric benchmarks in §2.2, and the figure that goes into the report.

**Status:** implemented for **all 8 backbones** (the `models` list now holds 8 entries); never executed (needs `.pth` files). Loads `{name}_best.pth` from `CHECKPOINT_DIR`. The `prediction_rgb` helper it relies on was verified in the smoke test (returns `(256, 256, 3)` RGB maps).

---

### 2.4 Promised in the plan, not yet in the notebook

The research text specifies a **three-stage** evaluation (semantic → instance → biological). Two stages are missing from the code:

- **Semantic metrics (test-set Dice + IoU):** there is no function that computes per-image Dice or IoU over the test set. The only Dice implementation in the whole notebook is `calculate_soft_dice` inside the error-analysis helper (cell #68), which is itself never called by an evaluation cell.
- **Instance metric (mAP across IoU thresholds):** not implemented at all. mAP requires *matching* each predicted component to a GT component by IoU at thresholds 0.50–0.95 in steps of 0.05, then averaging precision — no matching code exists anywhere.

Both are needed to satisfy "I will first evaluate them at two different stages, which are the semantic and instance segmentation" before the biological benchmarks in cell #63.

---

## 3. Interpretability Analysis

### 3.1 `Segmentation_Grad_CAM` — cell #66

**What it is:** Grad-CAM (Gradient-weighted Class Activation Mapping, Selvaraju et al. 2017) answers *"where is the network looking?"*. For a classifier, you pick a target class score, backpropagate it to a chosen convolutional layer, and use the arriving gradients as **per-channel importance weights** for that layer's activation maps. The weighted sum — after ReLU and normalization — is the heatmap overlayed on the image.

The `Segmentation_` prefix is the adaptation: a U-Net has **no class score**, so this implementation builds one from the segmentation head. The target score is:

```python
boundary_sum = pred[0, 1, :, :].sum()   # channel 1 = boundary
```

i.e. *"how strongly is the model saying 'boundary' across this image?"*. The gradients of that score w.r.t. the last decoder block then reveal which pixels of those feature maps most increase boundary predictions.

**How the code works, step by step:**

1. **Containers:** two empty lists — `gradients` (to catch backward-flowing gradients) and `activations` (to catch forward-pass feature maps).
2. **Hooks:** two closures are defined —
   - `backward_hook(module, grad_input, grad_output)` stores `grad_output[0].detach()` (the gradient arriving at the layer);
   - `forward_hook(module, input, output)` stores `output.detach()` (the layer's activation maps).
3. **Attachment:** both hooks are registered on `model.decoder.decoder4` — the last double-convolution block of the decoder (128 channels at half resolution), chosen because it is the deepest point that still holds sharp spatial structure before the 1×1 segmentation head. The handles are kept so the hooks can be removed later.
4. **Preparation:** `model.eval()`, `model.zero_grad()`, and `image_tensor.requires_grad = True` (asks PyTorch to track the graph back to the input).
5. **Forward pass:** `pred = model(image_tensor)` → the forward hook fires automatically and records the activations.
6. **Target score:** `boundary_sum = pred[0, 1, :, :].sum()` — a single scalar for the boundary channel.
7. **Backward pass:** `boundary_sum.backward()` → the backward hook fires and records the gradients.
8. **Cleanup:** `forward_handle.remove()` / `backward_handle.remove()` — hooks are removed so repeated calls don't accumulate stale tensors (memory-leak hygiene).
9. **Heatmap math:**
   - Global-average-pool the gradients: `pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])` → one weight per channel (shape C).
   - Weight the activations: `activation[i] *= pooled_gradients[i]` for every channel.
   - Collapse channels with a mean, `F.relu` (discard negative contributions — only keep features that *support* boundary prediction), normalize to 0–1, and bilinear-upsample to the **original image size** (`H, W = raw_img.shape`) to match the image.
10. **Visualization:** two panels — original grayscale on the left; on the right, the same image with the heatmap overlaid in the `jet` colormap at 50% transparency.

**Purpose:** the "prove it isn't cheating" test. If the model learned true biology, the hot region forms a **glowing ring tracing the nuclei** on the raw DAPI image. If the model had latched onto sensor artifacts or background noise, the heatmap would highlight those instead — the analysis converts "trust me" into a visual, auditable claim for biologists and regulators.

**Status:** ✅ fixed & smoke-tested (hooks fire, backward pass runs, heatmap normalizes and resizes correctly). Fixes applied: the heatmap now resizes to the **actual** image height/width (BBBC039 images are 520×696, not 512×512), the input tensor is moved to the model's device and unsqueezed if it is 3-D, and the normalization uses `torch.max(heatmap) + 1e-8` (no NaN on a zero-max heatmap).

---

### 3.2 `Visualize_U_Net_Feature_Maps` — cell #67

**What it is:** "cameras inside the network". Forward hooks intercept the tensors flowing through four chosen locations of the U-Net, and each captured stack of channels is averaged into a single grayscale map for display.

**The four tapped locations and what each is meant to prove:**

| Hook location | Layer in code | What the plan expects to see |
|---|---|---|
| **Early layers (ResNet stem)** | `model.encoder.conv1` | High-resolution edge/contrast maps — the first layer reacting to the dark background vs the glowing DAPI stain. Proof the stem acts as an edge detector. |
| **Bottleneck (end of encoder)** | `model.encoder.layer4` | 2048 channels at 1/32 resolution — spatially degraded but semantically deep (dense clusters, texture). Proof the encoder compresses to abstract concepts. |
| **Skip connection (decoder side)** | `model.decoder.decoder1` | Post-fusion maps showing high-resolution edges re-injected past the bottleneck. Proof the skip path restores the sharp boundaries the multi-head output needs. |
| **Final head output** | `model.decoder.segmentation_head` | The pre-softmax multi-head predictions. |

**How the code works, step by step:**

1. A module-level `activations = {}` dictionary acts as the database of captured tensors.
2. `get_activation(name)` returns a **closure** hook that stores `output.detach()` under its `name` — the closure is what lets one generic hook target four named slots.
3. `activations.clear()` runs at the start of every call, so a previous image's tensors can never leak into a new figure.
4. Four `register_forward_hook(...)` calls are made **before** inference (handles saved for removal).
5. `model.eval()` + `torch.no_grad()` + a single forward pass on `sample_image_tensor.unsqueeze(0)` — all four hooks fire automatically as the tensor flows through.
6. Hooks are removed after the pass.
7. For each captured tensor: `squeeze()` the batch dim → **mean across the channel dimension** (e.g. `(1, 2048, 16, 16)` → `(16, 16)`) → `np.maximum(..., 0)` as a ReLU → normalize to 0–1 (with an epsilon against divide-by-zero).
8. Plot: the original microscopy image plus one grayscale panel per tapped location; optionally saved as `{save_name}_FeatureMaps.png` at 300 dpi.

**Purpose:** the "tracking the heavy lifting" analysis — proving *what each part of the architecture learned*. It gives a structural explanation of *why* the U-Net design works (edges → semantics → skip-fused reconstruction), which is the layer-by-layer evidence the research plan calls for.

**Status:** ✅ fixed & smoke-tested (all 4 hooks fire and produce normalized maps for all 8 backbones). Fixes applied: the head hook now targets `model.decoder.segmentation_head` (the old `model.final_layer` did not exist), the sample tensor is moved to the model's device, and the normalization has an epsilon.

---

### 3.3 `Pathological_Case_Sorter_and_Visualizer` — cell #68 ("The Why")

**What it is:** comparative error analysis. Instead of a single aggregate number, this finds the exact test images where the **winning** model fails worst, then displays winning vs losing predictions side by side to explain *why* the architecture won (e.g., the winner separates a massive overlapping cluster while the loser merges it into one blob).

**Helper — `calculate_soft_dice(pred, target)`:**

1. `torch.softmax(pred, dim=1)` converts the 3-channel logits to probabilities.
2. Isolates the **interior channel** (index 2) from both prediction and target — the "standard nucleus Dice".
3. Computes `Dice = (2·Σ(p·t) + ε) / (Σp + Σt + ε)` with `ε = 1e-6`. This is the *only* Dice implementation in the notebook.

**How the main function works, step by step:**

1. Load the winning model's `_best.pth`, `eval()`, start a timer.
2. Loop over every batch of the test dataloader under `no_grad()`: forward pass → `calculate_soft_dice(pred, gt)` **per image in the batch** → append `(dice_score, image_filename)` — the real `.tif` filename from `test_dataset.image_paths` — to `image_scores`.
3. `sorted(image_scores, key=lambda x: x[0])` sorts **ascending** — the lowest Dice scores are the catastrophic failures.
4. `worst_10 = sorted_scores[:10]` isolates the bottom ten; their scores and real filenames are printed.
5. Load the **losing** model's weights; build a 10×4 grid — one row per failure: original image | human GT | winning prediction | losing prediction (predictions rendered with the same red-boundary/green-interior RGB scheme via the `prediction_rgb` helper).
6. Save `BBBC039_Pathological_Comparison_seresnext101_vs_resnet50.png` at 300 dpi.

**Purpose:** the plan's *"The Why"* — turning the winner's abstract metric lead into concrete, explainable failure modes (e.g., vanishing gradients on dense clusters), and building the argument that the winning architecture is robust for real-world HTS rather than merely lucky on average.

**Status:** ✅ fully rewritten (the old version was a skeleton: the grid was commented out, the models were undefined, only `images[0]` of each batch was scored, and scores were labeled with loader indices). The rewrite scores **every image**, tracks **real filenames** (so a low score can be traced back to a raw `.tif`), defines the models inside the function, and implements the full 10×4 grid. Verified in the smoke test; never run end-to-end (needs `.pth` files). The usage comments at the bottom of the cell show how to load the winner/loser from `CHECKPOINT_DIR`.

### 3.4 Also promised in the plan, not implemented

- **Active isolation of pathological edge cases** (severely fragmented nuclei from toxic drug doses) — no code exists that subsets plates/conditions or filters for fragmentation.

---

## 4. Current execution status of the notebook

Only the first 13 cells have in-notebook execution counts (imports, downloads, extraction, preview). Training/evaluation has intentionally never been run (per instruction) — there are no `.pth` weights, no `*_loss_curves.json`, no comparison figures yet. The raw BBBC039 images and annotations **are** on disk (`dataset/raw_images`, `dataset/raw_annotations`).

**Verified outside the notebook (headless smoke test, `bio` conda env, real BBBC039 data):**

- All 68 cells **compile** (static sweep of every cell's source).
- Data pipeline: seeded 50/25/25 split (100/50/50), train batch `(4,1,512,512)` + masks `(4,3,512,512)`, val batch at full 520×696.
- Loss: BCE+Dice returns a finite value on real data; gradients are finite.
- All **8 backbones** forward-pass at 256×256 and at full 520×696 with random weights.
- `Segmentation_Grad_CAM`: hooks + backward + heatmap OK.
- `Visualize_U_Net_Feature_Maps`: 4 hooks OK.
- Sorter helpers (`calculate_soft_dice`, `prediction_rgb`): OK.
- Benchmark inner logic (boundary-subtraction labeling, min-size filter, KS test) against real GT: OK — 456 nuclei counted in a batch, KS statistic computed.
- **Checkpointing pipeline** (`tests/test_checkpointing.py`, tiny stand-in models on CPU): fresh run → manifest `done` for all 8 with full resumable checkpoints in both folders; re-run → all 8 skipped; simulated Colab restart (in-session files wiped, status reset) → resumes at the correct epoch from the persistent checkpoint with loss history intact; early stopping → triggers after the patience window and is recorded in the manifest.

## 5. Fixes applied (changelog)

All originally-identified bugs are now fixed in `src/main.ipynb` and covered by the smoke test above:

1. ✅ **Dataset** (cell #18): BBBC039 masks are **RGBA PNGs** — the label IDs live in the **R channel**, so `mask = io.imread(mask_path)[:, :, 0]`; percentile normalization cast to `float32` (OpenCV 5 rejects float64); albumentations 2.x returns masks channel-last, so a conditional `permute(2, 0, 1)` was added.
2. ✅ **Loss** (cell #22): switched to `F.binary_cross_entropy_with_logits` (the old `binary_cross_entropy` on raw logits produced garbage gradients).
3. ✅ **Optimizers** (cells #46–#60): `encoder.parameters()` / `decoder.parameters()` — the decoder is now actually trained, with its own `DECODER_LR` group.
4. ✅ **SE-ResNeXt cells** (cells #41, #43): removed the `= res_next101.cuda()` overwrites; `models` now holds all **8** backbones.
5. ✅ **Dataloaders** (cell #19): rewritten — seeded `torch.randperm` split, paths from `config_vars`, `num_workers=0` (Colab-safe), augmentation wired to the training split, full-resolution validation/test.
6. ✅ **Decoder** (cells #26–#27): `skip.shape[2:]` (was `skip.size[2:]` — a method, crashed); added the missing **final 2× upsample** so output matches input resolution; imported `torch.nn.functional`.
7. ✅ **Grad-CAM** (cell #66): device move, `unsqueeze` for 3-D input, `+1e-8` epsilon, resize to actual `raw_img.shape`.
8. ✅ **Feature maps** (cell #67): `model.final_layer` → `model.decoder.segmentation_head`, device move, epsilon.
9. ✅ **Pathological sorter** (cell #68): full rewrite (per-image scoring, real filenames, 10×4 grid implemented).
10. ✅ **Benchmark** (cell #63): `MIN_NUCLEUS_AREA = 25` minimum-size filter on predicted components.
11. ✅ **Colab portability** (cell #2): pip installs, GPU check, a config fallback so the notebook runs even if only the `.ipynb` is uploaded, and a **Google Drive mount** for persistent checkpoints.
12. ✅ **Idempotent extraction** (cells #8/#10/#12): zip extraction skips already-extracted files (re-run safe).
13. ✅ **Resumable checkpoints** (cell #62): every epoch saves the full training state (weights + **optimizer** + **scheduler** + epoch + loss history) — interrupted models continue exactly where they stopped.
14. ✅ **Progress manifest** (cell #62): `training_progress.json` tracks each model's status (`training` / `resuming` / `done`); the cell prints a status overview, skips `done` models, and re-running it is idempotent.
15. ✅ **Early stopping** (cell #62): `EARLY_STOPPING_PATIENCE = 10` stagnant epochs → stop, sync, mark `done`.
16. ✅ **Colab persistence** (cell #62): local every-epoch checkpoint writes + a Drive sync every 5 epochs (a disconnect costs ≤ 5 epochs of progress); final sync on completion.
17. ✅ **Eval cells load from `CHECKPOINT_DIR`** (cells #63, #64, #68): best weights are read from the same persistent folder training wrote to, with a fallback if the training cell wasn't run this session.
18. ✅ **Regression tests** (`tests/test_checkpointing.py`): 4 scenarios — fresh run, skip-done re-run, resume after simulated Colab restart, early stopping.

**Still missing by design (documented, not implemented):** test-set Dice/IoU metrics, mAP across IoU thresholds, and toxic-dose fragmentation isolation — see §2.4 and §3.4.
