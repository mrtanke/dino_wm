## DINO-WM Representations Study

This repo is now configured for a PointMaze-only workflow.

Removed from the default workflow:
- PushT
- wall_single
- deformable environments

### Install Mujoco
Create the `.mujoco` directory and download Mujoco210 using `wget`:

```bash
mkdir -p ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P ~/.mujoco/
cd ~/.mujoco
tar -xzvf mujoco210-linux-x86_64.tar.gz
```

### Download Data and Checkpoints

```bash
wget -O dataset_pointmaze.zip "https://huggingface.co/datasets/revennn/dino_wm/resolve/main/dataset_pointmaze.zip"
```

After downloading, unzip the `dataset_pointmaze.zip` and the `point_maze.zip` inside.

### Container Layout

Expected mounted paths in the container:
- repo: `/workspace`
- dataset: `/data`
- checkpoints: `/checkpoints`
- MuJoCo: `/root/.mujoco`

The configs are aligned with this layout:
- `conf/train.yaml`: `ckpt_base_path=/checkpoints`
- `conf/plan.yaml`: `ckpt_base_path=/checkpoints`
- `conf/plan_point_maze.yaml`: `ckpt_base_path=/checkpoints`

## 1 Build Docker Image

From local repo directory:

```bash
docker build -t dino-wm-pointmaze .
```

## 2 Run Container

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

## 3 Verify Runtime Environment

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

## 4 Benchmark Pipeline (Train + Plan)

Compare DINOv2 vs four different encoders on both training and planning.

### 4.1 Models Included

- DINOv2: `encoder=dino` (default small: `dinov2_vits14`)
- `encoder=vjepa2`
- `encoder=dinov3`
- `encoder=dinotok`
- `encoder=vfm_vae`

### 4.2 Train Comparison Runs

Use the fair preset (`train_fair_compare.yaml`) to fix key settings across runs:

- same seed / epochs (10 epochs)
- same `concat_dim`, `num_hist`, `num_pred`
- fixed predictor internal width (`predictor.model_dim=1024`)
- predictor-only training by default (`has_decoder=False`, `model.train_decoder=False`)

```bash
# DINOv2 (base)
python train.py --config-name train_fair_compare encoder=dino model_name=cmp_dinov2_s

# V-JEPA 2
python train.py --config-name train_fair_compare encoder=vjepa2 model_name=cmp_vjepa2_s

# DINOv3
python train.py --config-name train_fair_compare encoder=dinov3 model_name=cmp_dinov3_s

# DINO-Tok
python train.py --config-name train_fair_compare encoder=dinotok model_name=cmp_dinotok_s

# VFM-VAE
python train.py --config-name train_fair_compare encoder=vfm_vae model_name=cmp_vfmvae_s
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

Note: with predictor-only training, planning metrics still run normally; decoder-based rollout video comparisons are skipped.

```bash
# DINOv2 (base)
python plan.py --config-name plan_point_maze.yaml model_name=cmp_dinov2_s
python plan.py --config-name plan_point_maze.yaml model_name=cmp_vjepa2_s
python plan.py --config-name plan_point_maze.yaml model_name=cmp_dinov3_s
python plan.py --config-name plan_point_maze.yaml model_name=cmp_dinotok_s
python plan.py --config-name plan_point_maze.yaml model_name=cmp_vfmvae_s
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
