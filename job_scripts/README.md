# Job Scripts for EBT-Mastery

This directory contains bash scripts for training all models in the project.

## Available Scripts

### 1. `train_simple_ebm.sh`
Trains Simple EBM for binary classification on synthetic data.
- **Duration**: ~2-5 minutes
- **Output**: `Simple_EBM/plots/`

```bash
bash job_scripts/train_simple_ebm.sh
```

### 2. `train_simple_ebm_regression.sh`
Trains Simple EBM for regression on multiple function types (quadratic, sin, cubic).
- **Duration**: ~5-10 minutes
- **Output**: `Simple_EBM/plots/`

```bash
bash job_scripts/train_simple_ebm_regression.sh
```

### 3. `train_ebt_nlp.sh`
Trains EBT for text generation with research-grade features.
- **Duration**: ~30-60 minutes per configuration
- **Output**: `EBT_NLP/outputs/`

```bash
bash job_scripts/train_ebt_nlp.sh
```

**Research Features**:
- `--corruption_type` - MCMC initialization (random_noise, gaussian_noise_around_embed, uniform)
- `--use_vocab_to_embed_mlp` - Non-linear vocab→embed transformation
- `--vocab_to_embed_num_layers` - MLP depth (1-3 layers)
- `--use_rms_norm` - Llama2-style RMSNorm

### 4. `train_ebt_img.sh`
Trains EBT for image denoising on MNIST with different noise levels.
- **Duration**: ~30-45 minutes per noise level
- **Output**: `EBT_IMG/outputs/`

```bash
bash job_scripts/train_ebt_img.sh
```

### 5. `run_all_experiments.sh`
Runs all experiments sequentially (master script).
- **Duration**: ~2-3 hours
- **Output**: All output directories

```bash
bash job_scripts/run_all_experiments.sh
```

## Requirements

Make sure you have:
1. Installed all dependencies: `pip install -r requirements.txt`
2. Sufficient disk space (~2GB for datasets and outputs)
3. GPU recommended but not required (CPU will be slower)

## Customization

You can modify the scripts or run individual Python commands with custom arguments. For example:

```bash
# Custom Simple EBM training
python Simple_EBM/simple_ebm.py \
    --num_epochs 200 \
    --batch_size 64 \
    --lr 0.005

# Custom EBT_NLP training with research features
python EBT_NLP/train_ebt_nlp.py \
    --model_type advanced \
    --num_epochs 100 \
    --corruption_type gaussian_noise_around_embed \
    --use_vocab_to_embed_mlp \
    --vocab_to_embed_num_layers 2 \
    --use_rms_norm \
    --use_wandb

# Custom EBT_IMG training
python EBT_IMG/train_ebt_img.py \
    --num_epochs 50 \
    --noise_std 0.25 \
    --num_optim_steps 10
```

## Troubleshooting

**CUDA out of memory**: Reduce `--batch_size`

**Training too slow**: 
- Enable GPU if available
- Reduce `--num_epochs` or `--num_samples`
- For EBT models, reduce `--num_optim_steps`

**Import errors**: Make sure you're in the project root and have activated your virtual environment

## Monitoring

- **Console**: All scripts print progress to console
- **WandB**: Add `--use_wandb` flag to any Python script for online monitoring
- **Plots**: Check output directories for visualizations

For more details, see the main [README.md](../README.md).

