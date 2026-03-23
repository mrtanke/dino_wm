# DINO-WM PointMaze Workflow

This repo is now configured for a PointMaze-only workflow.

Removed from the default workflow:
- PushT
- wall_single
- deformable environments

## Container Layout

Expected mounted paths in the container:
- repo: `/workspace`
- dataset: `/data`
- checkpoints: `/checkpoints`
- MuJoCo: `/root/.mujoco`

The configs are aligned with this layout:
- `conf/train.yaml`: `ckpt_base_path=/checkpoints`
- `conf/plan.yaml`: `ckpt_base_path=/checkpoints`
- `conf/plan_point_maze.yaml`: `ckpt_base_path=/checkpoints`

## 1) Build Docker Image

From local repo directory:

```bash
docker build -t dino-wm-pointmaze .
```

## 2) Run Container

```bash
docker run --rm -it \
  --gpus all \
  -v $HOME/dino_wm:/workspace \
  -v $HOME/data:/data \
  -v $HOME/checkpoints:/checkpoints \
  -v $HOME/.mujoco:/root/.mujoco \
  -w /workspace \
  dino-wm-pointmaze
```

## 3) Verify Runtime Environment

Inside the container:

```bash
python -c "import wandb; print(wandb.__version__)"
python -c "import mujoco_py; print('mujoco_py ok')"
echo $DATASET_DIR
echo $LD_LIBRARY_PATH
```

Expected:
- `DATASET_DIR=/data`
- MuJoCo path included in `LD_LIBRARY_PATH`

## 4) Fair Benchmark Pipeline (Train + Plan)

Compare HF encoders vs DINOv2 on both training and planning.

### 4.1 Models Included

- DINOv2: `encoder=dino` (default large: `dinov2_vitl14`)
- HF: `encoder=vjepa2`
- HF: `encoder=dinov3`
- HF: `encoder=dinotok`
- HF: `encoder=vfm_vae`

HF model IDs can be overridden with env vars:

- `VJEPA2_MODEL_ID`
- `DINOV3_MODEL_ID`
- `DINOTOK_MODEL_ID`
- `VFMVAE_MODEL_ID`

### 4.2 Train Comparison Runs

Use the fair preset (`train_fair_compare.yaml`) to fix key settings across runs:

- same seed / epochs
- same `concat_dim`, `num_hist`, `num_pred`
- fixed predictor internal width (`predictor.model_dim=1024`)

```bash
# DINOv2 (large)
python train.py --config-name train_fair_compare encoder=dino model_name=cmp_dinov2_l

# HF encoders
python train.py --config-name train_fair_compare encoder=vjepa2 model_name=cmp_vjepa2_l
python train.py --config-name train_fair_compare encoder=dinov3 model_name=cmp_dinov3_l
python train.py --config-name train_fair_compare encoder=dinotok model_name=cmp_dinotok_l
python train.py --config-name train_fair_compare encoder=vfm_vae model_name=cmp_vfmvae_l
```

Training checkpoints are written to:

```text
/checkpoints/outputs/<model_name>
```

Trainer logs parameter counts at startup:

- `params/encoder_total`, `params/encoder_trainable`
- `params/predictor_total`, `params/predictor_trainable`
- `params/model_total`, `params/model_trainable`

### 4.3 Plan Comparison Runs

Run planning on each trained checkpoint with the same planning config:

```bash
python plan.py --config-name plan_point_maze.yaml model_name=cmp_dinov2_l
python plan.py --config-name plan_point_maze.yaml model_name=cmp_vjepa2_l
python plan.py --config-name plan_point_maze.yaml model_name=cmp_dinov3_l
python plan.py --config-name plan_point_maze.yaml model_name=cmp_dinotok_l
python plan.py --config-name plan_point_maze.yaml model_name=cmp_vfmvae_l
```

Planning outputs are saved to:

```text
/workspace/plan_outputs
```

### 4.4 Fairness Checklist (Important)

Keep the following fixed when comparing models:

- training config: `train_fair_compare.yaml`
- planning config: `plan_point_maze.yaml`
- planner, `goal_source`, `goal_H`, `seed`, `n_evals`
- dataset path and split

## Citation

```text
@misc{zhou2024dinowmworldmodelspretrained,
    title={DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning},
    author={Gaoyue Zhou and Hengkai Pan and Yann LeCun and Lerrel Pinto},
    year={2024},
    eprint={2411.04983},
    archivePrefix={arXiv},
    primaryClass={cs.RO},
    url={https://arxiv.org/abs/2411.04983},
}
```
