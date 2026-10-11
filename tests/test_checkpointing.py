"""Regression tests for the checkpointing / resume / early-stopping logic in main.ipynb.

Extracts the training cell from the notebook and executes it against tiny fake
models + dataloaders on CPU (milliseconds per epoch), verifying:
  1. a fresh run trains all backbones and marks them 'done' in the manifest,
  2. re-running the cell skips models already marked 'done' (idempotent),
  3. a simulated Colab restart (in-session files wiped, status reset) resumes
     from the persistent checkpoint at the correct epoch,
  4. early stopping triggers after N stagnant epochs and is recorded.
"""
import contextlib
import io
import json
import os
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

NB_PATH = Path(__file__).resolve().parent.parent / 'src' / 'main.ipynb'

# Manifest keys (backbone_names in the notebook)
NAMES = ["resnet50", "resnet101", "wide_resnet50", "wide_resnet101",
         "resnext50", "resnext101", "seresnext50", "seresnext101"]
# Actual variable names the training cell reads from the notebook namespace
VAR_NAMES = ["resnet50", "resnet101", "wide_resnet50", "wide_resnet101",
             "res_next50", "res_next101", "se_res_next50", "se_res_next101"]


def cell_source():
    nb = json.load(open(NB_PATH, encoding='utf-8'))
    for c in nb['cells']:
        if c['cell_type'] != 'code':
            continue
        src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
        if 'CHECKPOINTING SETUP' in src:
            return src
    raise RuntimeError('training cell not found in notebook')


class TinyDataloader:
    """Minimal iterable that mimics a DataLoader (iterable + len)."""

    def __init__(self, batches, constant=False):
        self.batches = batches
        self.constant = constant

    def __iter__(self):
        for _ in range(self.batches):
            if self.constant:
                yield torch.zeros(2, 1, 16, 16), torch.zeros(2, 3, 16, 16)
            else:
                yield torch.randn(2, 1, 16, 16), torch.rand(2, 3, 16, 16)

    def __len__(self):
        return self.batches


def mse_loss(pred, target):
    return ((pred - target) ** 2).mean()


def constant_loss(pred, target):
    """Always returns 10.0 but stays connected to the graph (backward-safe)."""
    return (pred * 0).mean() + 10.0


def make_namespace(models, num_epochs, ckpt_dir, local_dir, loss_fn, dataloaders=None):
    optimizers = [Adam(m.parameters(), lr=1e-3) for m in models]
    schedulers = [ReduceLROnPlateau(o, mode='min', factor=0.1, patience=3) for o in optimizers]
    ns = {
        'NUM_EPOCHS': num_epochs,
        'CHECKPOINT_DIR': ckpt_dir,
        'LOCAL_CKPT_DIR': local_dir,
        'BCE_Dice_loss': loss_fn,
    }
    for name, model in zip(VAR_NAMES, models):
        ns[name] = model
    for name, opt in zip([n + '_optimizer' for n in VAR_NAMES], optimizers):
        ns[name] = opt
    for name, sched in zip([n + '_scheduler' for n in VAR_NAMES], schedulers):
        ns[name] = sched
    if dataloaders is None:
        dataloaders = (TinyDataloader(2), TinyDataloader(1))
    ns['train_dataloader'], ns['valid_dataloader'] = dataloaders
    return ns, optimizers, schedulers


def run_cell(ns):
    """Execute the training cell in the given namespace; return captured stdout."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(cell_source(), '<training-cell>', 'exec'), ns)
    return out.getvalue()


def tiny_backbones():
    return [nn.Conv2d(1, 3, kernel_size=1) for _ in range(8)]


def manifest(ckpt_dir):
    with open(os.path.join(ckpt_dir, 'training_progress.json')) as f:
        return json.load(f)


def test_fresh_run_trains_all_and_marks_done():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = os.path.join(tmp, 'persistent')
        local_dir = os.path.join(tmp, 'local')
        ns, _, _ = make_namespace(tiny_backbones(), 2, ckpt_dir, local_dir, mse_loss)
        out = run_cell(ns)

        m = manifest(ckpt_dir)
        for name in NAMES:
            assert m[name]['status'] == 'done', name
            assert m[name]['last_epoch'] == 1, name
            # every artifact exists in both folders after the final sync
            assert os.path.exists(os.path.join(local_dir, f'{name}_checkpoint.pth'))
            assert os.path.exists(os.path.join(ckpt_dir, f'{name}_checkpoint.pth'))
            assert os.path.exists(os.path.join(ckpt_dir, f'{name}_best.pth'))
            assert os.path.exists(os.path.join(ckpt_dir, f'{name}_loss_curves.json'))

        # checkpoint carries the full resumable state
        ckpt = torch.load(os.path.join(ckpt_dir, 'resnet50_checkpoint.pth'), map_location='cpu')
        for key in ('epoch', 'model_state_dict', 'optimizer_state_dict',
                    'scheduler_state_dict', 'best_val_loss', 'loss_history', 'early_stopped'):
            assert key in ckpt, key
        assert len(ckpt['loss_history']['train']) == 2


def test_rerun_skips_done_models():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = os.path.join(tmp, 'persistent')
        local_dir = os.path.join(tmp, 'local')
        ns, _, _ = make_namespace(tiny_backbones(), 2, ckpt_dir, local_dir, mse_loss)
        run_cell(ns)

        ckpt_mtime = os.path.getmtime(os.path.join(local_dir, 'resnet50_checkpoint.pth'))
        out = run_cell(ns)  # second run must not retrain

        assert out.count("Already 'done' - skipping") == 8
        assert 'New best model saved' not in out
        assert os.path.getmtime(os.path.join(local_dir, 'resnet50_checkpoint.pth')) == ckpt_mtime


def test_resume_after_simulated_colab_restart():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = os.path.join(tmp, 'persistent')
        local_dir = os.path.join(tmp, 'local')
        ns, _, _ = make_namespace(tiny_backbones(), 2, ckpt_dir, local_dir, mse_loss)
        run_cell(ns)

        # Simulate a Colab runtime restart mid-training: in-session files are wiped
        # and the manifest still says 'training' (the crash happened before 'done').
        for f in ('resnet50_checkpoint.pth', 'resnet50_best.pth'):
            os.remove(os.path.join(local_dir, f))
        m = manifest(ckpt_dir)
        m['resnet50']['status'] = 'training'
        with open(os.path.join(ckpt_dir, 'training_progress.json'), 'w') as f:
            json.dump(m, f, indent=2)

        ns['NUM_EPOCHS'] = 4  # the user later raises the epoch budget
        out = run_cell(ns)

        assert 'Resuming resnet50 from epoch 2/4' in out
        m = manifest(ckpt_dir)
        assert m['resnet50']['status'] == 'done'
        assert m['resnet50']['last_epoch'] == 3  # epochs 2 and 3 were trained
        ckpt = torch.load(os.path.join(ckpt_dir, 'resnet50_checkpoint.pth'), map_location='cpu')
        assert ckpt['epoch'] == 3
        assert len(ckpt['loss_history']['train']) == 4  # 2 original + 2 resumed
        for name in NAMES[1:]:
            assert m[name]['status'] == 'done'  # untouched models stay skipped


def test_early_stopping_triggers_and_is_recorded():
    src = cell_source().replace('EARLY_STOPPING_PATIENCE = 10', 'EARLY_STOPPING_PATIENCE = 3')
    assert 'EARLY_STOPPING_PATIENCE = 3' in src
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = os.path.join(tmp, 'persistent')
        local_dir = os.path.join(tmp, 'local')
        # Constant val loss: improves once (inf -> 10), then stagnates forever,
        # so early stopping fires deterministically after 3 stagnant epochs.
        ns, _, _ = make_namespace(tiny_backbones(), 100, ckpt_dir, local_dir,
                                  constant_loss, dataloaders=(TinyDataloader(1, constant=True),
                                                               TinyDataloader(1, constant=True)))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(compile(src, '<training-cell>', 'exec'), ns)

        assert 'Early stopping triggered: 3 epochs without val-loss improvement.' in out.getvalue()
        m = manifest(ckpt_dir)
        assert m['resnet50']['status'] == 'done'
        assert m['resnet50']['early_stopped'] is True
        assert m['resnet50']['last_epoch'] == 3  # epochs 0..3 = 1 improvement + 3 stagnant
        ckpt = torch.load(os.path.join(ckpt_dir, 'resnet50_checkpoint.pth'), map_location='cpu')
        assert ckpt['early_stopped'] is True


if __name__ == '__main__':
    test_fresh_run_trains_all_and_marks_done()
    test_rerun_skips_done_models()
    test_resume_after_simulated_colab_restart()
    test_early_stopping_triggers_and_is_recorded()
    print('\nALL CHECKPOINTING TESTS PASSED')
