# ⚡️🧠 Energy-Based Transformers Mastery

<div align="center">

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)

*Educational implementation of Energy-Based Models (EBMs) and Energy-Based Transformers (EBTs) from scratch*

[Quick Start](#-quick-start) • [Features](#-features) • [Documentation](#-documentation) • [Benchmarks](#-benchmarks)

</div>

---

## 📚 Overview

This repository provides a **comprehensive, from-scratch implementation** of Energy-Based Models and Energy-Based Transformers for educational and research purposes. Unlike standard transformers that predict probability distributions, EBTs output energy values and find optimal predictions through **iterative optimization** — enabling adaptive "thinking" and better generalization.

### ⚠️ Project Context & Scope

**What This Project Is:**
- 📚 **Educational Implementation**: Learn EBT concepts from scratch with clear, documented code
- 🔬 **Research Exploration**: Experiment with fresh ML research (arXiv July 2025: [Gladstone et al.](https://arxiv.org/abs/2507.02092))
- 🧪 **Prototype Platform**: Test energy-based learning on real datasets (CIFAR-10/100, GLUE)

**What This Project Is NOT:**
- ❌ **Production-Ready**: Educational implementation, not optimized framework
- ❌ **Mainstream**: EBT is emerging research (~20K YouTube views, not yet HuggingFace mainstream)
- ❌ **State-of-the-Art**: Established transformers (GPT, BERT, LLaMA) have deeper research and larger communities

**Why EBT Matters (Despite Being Niche):**
- ✨ **System 2 Thinking**: Energy minimization enables deeper reasoning (35% scaling improvements in research)
- 🎯 **Adaptive Compute**: Model "thinks" longer on harder problems (unlike fixed-cost transformers)
- 🔮 **2025 Research Trend**: Part of growing interest in unsupervised reasoning and test-time compute

**Use This Project To:**
- Understand energy-based learning concepts
- Explore adaptive computation strategies
- Demonstrate awareness of cutting-edge research
- Build educational ML portfolio pieces

💡 **Portfolio Position**: "Experimental EBT implementation (2025 research) for adaptive compute — exploring energy-based learning beyond standard transformers."

### 🎯 Key Features

- **📖 Educational**: Progressive learning path from simple EBMs to full EBTs
- **🔬 Research-Grade Datasets**: CIFAR-10/100, GLUE benchmark alongside synthetic examples
- **🏗️ Well-Engineered**: Docker, CPU fallback, 50+ tests, CI/CD pipeline
- **💎 Clean Code**: OOP/SOLID principles, well-documented, extensible
- **🧪 Edge Case Testing**: High-dimensional energy, numerical stability, gradient flow
- **⚡ Benchmarked**: Performance metrics on real data (PSNR, accuracy, throughput)

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# GPU training
docker-compose up -d ebt-gpu
docker exec -it ebt-mastery-gpu bash
python EBT_IMG/train_ebt_img.py --dataset cifar10

# CPU training (no GPU required)
docker-compose up -d ebt-cpu
docker exec -it ebt-mastery-cpu bash
```

### Option 2: Local Installation

```bash
git clone https://github.com/DeOcxan/EBT-Mastery.git
cd EBT-Mastery

# Create environment
conda create -n ebt python=3.12
conda activate ebt
pip install -r requirements.txt

# Train on real data
python EBT_IMG/train_ebt_img.py --dataset cifar10 --device auto
```

**Auto Device Selection**: Automatically falls back to CPU if GPU unavailable.

---

## 🎯 Features

### Real-World Datasets

| Domain | Datasets | Credibility |
|--------|----------|-------------|
| **NLP** | GLUE (SST-2, COLA, MRPC) | ⭐⭐⭐⭐⭐ Industry standard |
| **Vision** | CIFAR-10/100 | ⭐⭐⭐⭐⭐ Widely used benchmark |
| **Legacy** | Synthetic patterns | ⭐⭐ Educational only |

### Engineering Features

- **Docker Support**: Reproducible environments (GPU + CPU)
- **Device Auto-Detection**: Smart fallback to CPU/MPS
- **Flexible Logging**: WandB + TensorBoard support (auto-detection)
- **Comprehensive Tests**: 50+ unit tests with edge cases
- **Memory Efficient**: Tested with large batches
- **Version Handling**: Compatible with PyTorch 2.0-2.5

### Architecture Components

```
Input → Transformer Blocks → Energy Head (scalar)
         ↓
     Iterative Optimization (3-10 steps)
         ↓
     Output (via gradient descent)
```

**Research Features:**
- MCMC corruption strategies
- Learnable step sizes (α ∈ [1e-4, 1.0])
- Langevin dynamics with noise
- RMSNorm (Llama2-style)

---

## 📁 Project Structure

```
EBT-Mastery/
├── docs/
│   └── CALCULATIONS.md              # Manual energy calculations with edge cases
│
├── Simple_EBM/                      # 🎓 START HERE - Basic EBMs
│   ├── simple_ebm.py                # Binary classification EBM
│   └── simple_ebm_regression.py     # Regression with iterative optimization
│
├── EBT_NLP/                         # Energy-Based Transformers for text
│   ├── transformer_components.py    # RMSNorm, Attention, FFN (from scratch)
│   ├── ebt_nlp.py                   # Basic EBT with MCMC corruption
│   ├── ebt_nlp_advanced.py          # Learnable step size + Langevin
│   └── train_ebt_nlp.py             # Training on synthetic + GLUE
│
├── EBT_IMG/                         # Energy-Based Models for images
│   ├── ebt_img_denoise.py           # Bidirectional EBT for denoising
│   └── train_ebt_img.py             # Training on MNIST/CIFAR
│
├── utils/                           # Shared utilities (OOP/SOLID)
│   ├── base_energy_model.py         # Abstract base classes
│   ├── dataset_loaders.py           # Real dataset loaders (CIFAR, GLUE)
│   ├── device_manager.py            # CPU fallback + version checking
│   ├── module_loader.py             # Import helper for dashed folders
│   ├── optimization.py              # StableAdamW, LARS, LR schedulers
│   ├── model_utils.py               # Checkpointing, logging, seeding
│   └── visualization.py             # Plotting utilities
│
├── tests/                           # Comprehensive unit tests
│   ├── test_energy_models.py        # Edge cases, stability, gradient flow
│   ├── test_datasets.py             # Dataset loading tests
│   └── test_device_manager.py       # Device compatibility tests
│
├── benchmarks/                      # Performance benchmarks
│   └── benchmark_real_datasets.py   # CIFAR-10/100 evaluation
│
├── job_scripts/                     # Bash scripts for training
│   ├── train_simple_ebm.sh
│   ├── train_ebt_nlp.sh
│   └── train_ebt_img.sh
│
├── Dockerfile                       # GPU container
├── Dockerfile.cpu                   # CPU container
├── docker-compose.yml               # Orchestration
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🔬 Benchmarks

### CIFAR-10 Denoising (Noise σ=0.2)

| Metric | Value |
|--------|-------|
| PSNR | 28.5 dB |
| Inference Speed | 45 samples/sec (GPU) |
| Memory | 2.3 GB (batch=32) |
| Parameters | 1.2M |

**Run benchmarks:**
```bash
python benchmarks/benchmark_real_datasets.py --dataset cifar10 --device auto
```

---

## 🛠️ Usage Examples

### Train on CIFAR-10 (Real Data)

```bash
# With WandB logging
python EBT_IMG/train_ebt_img.py \
    --dataset cifar10 \
    --num_epochs 50 \
    --batch_size 64 \
    --use_wandb

# With TensorBoard logging (local)
python EBT_IMG/train_ebt_img.py \
    --dataset cifar10 \
    --num_epochs 50 \
    --batch_size 64 \
    --use_tensorboard
```

### Train on GLUE-SST2 (Sentiment Analysis)

```bash
# Advanced model with TensorBoard
python EBT_NLP/train_ebt_nlp.py \
    --dataset glue-sst2 \
    --model_type advanced \
    --num_epochs 20 \
    --use_tensorboard

# View TensorBoard logs
tensorboard --logdir ./EBT_NLP/outputs/logs/tensorboard
```

### Run Tests

```bash
# All tests
pytest tests/ -v

# Specific test suite
pytest tests/test_energy_models.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

---

## 📖 Documentation

### Learning Path

1. **Start**: `Simple_EBM/simple_ebm.py` - Understand core energy-based learning (~10 min)
2. **Math**: `docs/CALCULATIONS.md` - Manual calculations & edge cases (~15 min)
3. **Scale**: `EBT_NLP/ebt_nlp.py` - Energy-based transformers (~30 min)
4. **Vision**: `EBT_IMG/ebt_img_denoise.py` - Bidirectional denoising (~20 min)
5. **Deploy**: Docker + Real datasets for production (~15 min)

### Key Concepts

<details>
<summary><b>What are Energy-Based Models?</b></summary>

Standard models predict probabilities: `p(y|x) = softmax(f(x))`

EBMs assign energy: `E(x, y) = f(x, y)` and find best output: `y* = argmin_y E(x, y)`

**Advantages:**
- Flexible output spaces
- Adaptive "thinking" time
- Better uncertainty quantification
- Out-of-distribution detection
</details>

<details>
<summary><b>Why use Docker?</b></summary>

Docker ensures **reproducibility** across different machines:
- Consistent PyTorch/CUDA versions
- No dependency conflicts
- Easy deployment
- Both GPU and CPU support
</details>

<details>
<summary><b>CPU vs GPU Training</b></summary>

| Model | CPU Time | GPU Time |
|-------|----------|----------|
| Simple EBM | 3 min | 1 min |
| EBT_NLP | 2-3 hours | 30 min |
| EBT_IMG (CIFAR) | 4-5 hours | 45 min |

The code **automatically detects** and uses best available device.
</details>

---

## 🧪 Testing & Validation

Comprehensive test suite covering:

- ✅ High-dimensional energy (10K+ dims)
- ✅ Numerical stability (extreme values)
- ✅ Gradient flow verification
- ✅ Memory efficiency (large batches)
- ✅ Device compatibility (CPU/CUDA/MPS)
- ✅ Edge cases (zero steps, single pixels)

**Run specific test categories:**
```bash
pytest tests/test_energy_models.py::TestEnergyModelEdgeCases -v
pytest tests/test_device_manager.py::TestDeviceCompatibility -v
```

---

## 🐳 Docker Usage

### Build Images

```bash
# GPU image
docker build -t ebt-mastery:gpu .

# CPU image
docker build -f Dockerfile.cpu -t ebt-mastery:cpu .
```

### Run Training

```bash
# Interactive GPU session
docker-compose run ebt-gpu bash

# Run specific experiment
docker-compose run ebt-gpu python EBT_IMG/train_ebt_img.py --dataset cifar10

# Jupyter notebook
docker-compose up jupyter
# Access at http://localhost:8888
```

### Volume Mounting

Data and outputs are automatically mounted:
- `./data` → `/workspace/data` (datasets)
- `./outputs` → `/workspace/outputs` (checkpoints, logs)

**Full guide**: See [DOCKER_GUIDE.md](DOCKER_GUIDE.md)

---

## ❓ FAQ

<details>
<summary><b>Do I need a GPU?</b></summary>

No. The code automatically uses CPU if GPU unavailable. Training is slower but works.
</details>

<details>
<summary><b>What if I have PyTorch 1.x?</b></summary>

The code checks version and warns you. Update with: `pip install --upgrade torch torchvision`
</details>

<details>
<summary><b>How do I add a new dataset?</b></summary>

Implement `IDatasetLoader` interface in `utils/dataset_loaders.py`:
```python
class MyDatasetLoader(IDatasetLoader):
    def load_dataset(self): ...
    def get_metadata(self): ...
```
</details>

<details>
<summary><b>Why not use diffusion models?</b></summary>

EBTs are more flexible (no fixed noise schedule) and better for reasoning tasks. Diffusion excels at generation.
</details>

<details>
<summary><b>How do I choose between WandB and TensorBoard?</b></summary>

Use WandB for cloud experiments with team collaboration. Use TensorBoard for local experiments and quick prototyping. By default, TensorBoard is used if WandB is not specified.
</details>

---

## ⚖️ Known Limitations

### Scale & Model Size
- **Small Models**: ~1M parameters (sufficient for learning, not for SOTA performance)
- **Training Epochs**: 50-100 epochs work for CIFAR-10, GLUE may need 200+ for full convergence
- **Single GPU**: No distributed training yet

### Deployment
- **Local Only**: Docker works locally, but no cloud deployment guides (AWS/GCP)
- **No Serving**: No REST API or model serving infrastructure
- **Inference Optimization**: No TorchScript, ONNX, or quantization yet

### Dependencies
- **Version Pinning**: Tested with PyTorch 2.0-2.5.1, strict pins may break with future updates
- **MPS Support**: Apple Silicon works but has edge cases (PyTorch 2.1+ recommended)
- **Logging**: WandB + TensorBoard supported, but no MLflow/Neptune

💡 **For production**: Use established frameworks (HuggingFace Transformers, PyTorch Lightning) instead.

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

- [Energy-Based Transformers](https://arxiv.org/abs/2507.02092) - Original research
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/) - GLUE benchmark

---

<div align="center">

**Built with ❤️ for the ML community**

</div>

