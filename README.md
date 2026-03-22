# DINO_WM (PushT-Only)

[Paper](https://arxiv.org/abs/2411.04983) | [Homepage](https://dino-wm.github.io/) | [Github](https://github.com/gaoyuezhou/dino_wm)

This repo is configured for PushT-only experiments.

The high-level world-model pipeline is:

`image encoder -> latent transition model -> planning`

## Representation Ablation Scope

Baseline:
- DINOv2 patch tokens (`conf/encoder/dino.yaml`)

New encoder configs added for representation study:
- V-JEPA 2 (`conf/encoder/vjepa2.yaml`)
- DINOv3 (`conf/encoder/dinov3.yaml`)
- DINO-Tok (`conf/encoder/dinotok.yaml`)
- VFM-VAE (`conf/encoder/vfm_vae.yaml`)

Code adapters for these encoders:
- `models/vjepa2.py`
- `models/dinov3.py`
- `models/dinotok.py`
- `models/vfm_vae.py`
- shared helper: `models/hf_vision.py`

## 1. Setup

```bash
git clone https://github.com/gaoyuezhou/dino_wm.git
cd dino_wm
conda env create -f environment.yaml
conda activate dino_wm
```

Install MuJoCo following your platform-specific instructions.

For normal Ubuntu/Linux usage, `environment.yaml` is already aligned with the original repo setup.

## 2. Configure Data and Checkpoints

Set dataset path:

```bash
export DATASET_DIR=/path/to/data
```

Expected dataset layout:

```text
data/
  pusht_noise/
```

Update checkpoint base path in plan config:
- `conf/plan_pusht.yaml`: set `ckpt_base_path` to your checkpoints root.

## 3. Reproduce Planning with Pretrained PushT Checkpoint

```bash
python plan.py --config-name plan_pusht.yaml model_name=pusht
```

This verifies:
- MuJoCo runtime
- dataset path
- checkpoint loading
- end-to-end planning

Planning outputs are written to `plan_outputs/`.

## 4. Retrain Baseline (if needed)

```bash
python train.py --config-name train.yaml env=pusht frameskip=5 num_hist=3
```

Checkpoints are saved under:

`<ckpt_base_path>/outputs/<model_name>`

## 5. Train Each Representation Variant

Use the same training command and only switch encoder config:

```bash
# DINOv2 baseline
python train.py --config-name train.yaml env=pusht encoder=dino

# V-JEPA 2
python train.py --config-name train.yaml env=pusht encoder=vjepa2

# DINOv3
python train.py --config-name train.yaml env=pusht encoder=dinov3

# DINO-Tok
python train.py --config-name train.yaml env=pusht encoder=dinotok

# VFM-VAE
python train.py --config-name train.yaml env=pusht encoder=vfm_vae
```

Note: if a model hub identifier changes, update `model_name` field in the corresponding file under `conf/encoder/`.

## 6. Plan with Trained Checkpoints

```bash
python plan.py --config-name plan_pusht.yaml model_name=<your_model_name>
```

## Optional: Windows Note

If you run this on Windows, use WSL2 for best compatibility with the Linux-first dependency stack.

## Citation

```bibtex
@misc{zhou2024dinowmworldmodelspretrained,
      title={DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning},
      author={Gaoyue Zhou and Hengkai Pan and Yann LeCun and Lerrel Pinto},
      year={2024},
      eprint={2411.04983},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2411.04983}
}
```
