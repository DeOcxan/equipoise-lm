# 🐳 Docker Guide for EBT-Mastery

Complete guide for running EBT-Mastery in Docker containers.

## 📋 Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Docker (for GPU support)

**Install Docker:**
- Ubuntu: `curl -fsSL https://get.docker.com | sh`
- Windows/Mac: [Docker Desktop](https://www.docker.com/products/docker-desktop)

**Install NVIDIA Docker (GPU only):**
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

---

## 🚀 Quick Start

### 1. Build Images

```bash
# Build both GPU and CPU images
docker-compose build

# Or build individually
docker build -t ebt-mastery:gpu .
docker build -f Dockerfile.cpu -t ebt-mastery:cpu .
```

### 2. Run Training

#### Interactive Mode (Recommended for experimentation)

```bash
# Start GPU container
docker-compose up -d ebt-gpu
docker exec -it ebt-mastery-gpu bash

# Inside container
python EBT_IMG/train_ebt_img.py --dataset cifar10
```

#### One-Shot Mode (Recommended for production)

```bash
# Run single experiment
docker-compose run --rm ebt-gpu \
    python EBT_IMG/train_ebt_img.py \
    --dataset cifar10 \
    --num_epochs 50 \
    --batch_size 64
```

### 3. View Outputs

Outputs are automatically saved to `./outputs/`:
```bash
ls -lh outputs/
# checkpoints/, logs/, visualizations/
```

---

## 🎯 Common Use Cases

### Use Case 1: Development & Debugging

```bash
# Start container with interactive shell
docker-compose run --rm ebt-gpu bash

# Inside container:
# - Edit code with vim/nano
# - Run quick tests
# - Debug interactively

python -c "import torch; print(torch.cuda.is_available())"
pytest tests/ -v
```

### Use Case 2: Long Training Runs

```bash
# Start in background (detached mode)
docker-compose up -d ebt-gpu

# Run training
docker exec ebt-mastery-gpu python EBT_IMG/train_ebt_img.py \
    --dataset cifar10 \
    --num_epochs 100 \
    --use_wandb

# Monitor logs
docker logs -f ebt-mastery-gpu

# Stop when done
docker-compose down
```

### Use Case 3: Jupyter Notebook

```bash
# Start Jupyter server
docker-compose up jupyter

# Access at http://localhost:8888
# Token is displayed in logs:
docker logs ebt-jupyter
```

### Use Case 4: CPU-Only Environments

```bash
# Use CPU image (no GPU required)
docker-compose up -d ebt-cpu
docker exec -it ebt-mastery-cpu bash

python EBT_IMG/train_ebt_img.py --dataset cifar10 --device cpu
```

### Use Case 5: Running Benchmarks

```bash
docker-compose run --rm ebt-gpu \
    python benchmarks/benchmark_real_datasets.py \
    --dataset cifar10 \
    --device cuda

# Results saved to ./benchmarks/results/
```

---

## 🔧 Configuration

### Modify Docker Compose

Edit `docker-compose.yml` to customize:

```yaml
services:
  ebt-gpu:
    # Use specific GPU
    environment:
      - CUDA_VISIBLE_DEVICES=0  # Change to 1, 2, etc.
    
    # Mount additional volumes
    volumes:
      - ./custom_data:/workspace/custom_data
    
    # Set resource limits
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 16G
```

### Build Arguments

```bash
# Specify Python version
docker build --build-arg PYTHON_VERSION=3.11 -t ebt-mastery:gpu .

# Use specific CUDA version
docker build --build-arg CUDA_VERSION=11.8 -t ebt-mastery:gpu .
```

---

## 📊 Volume Management

Docker Compose automatically mounts:

| Local Path | Container Path | Purpose |
|------------|----------------|---------|
| `./` | `/workspace/EBT-Mastery` | Code |
| `./data` | `/workspace/data` | Datasets |
| `./outputs` | `/workspace/outputs` | Results |

**Benefits:**
- Datasets persist between runs
- Checkpoints saved to host
- Code changes reflected immediately

---

## 🛠️ Troubleshooting

### Issue: GPU not detected

```bash
# Check NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi

# Check container access
docker exec ebt-mastery-gpu nvidia-smi
```

**Fix:**
1. Ensure NVIDIA drivers installed: `nvidia-smi`
2. Install nvidia-docker2
3. Restart Docker: `sudo systemctl restart docker`

### Issue: Out of memory

```bash
# Reduce batch size
docker exec ebt-mastery-gpu \
    python EBT_IMG/train_ebt_img.py --batch_size 16
```

### Issue: Permission denied on volumes

```bash
# Fix permissions
sudo chown -R $USER:$USER ./data ./outputs
chmod -R 755 ./data ./outputs
```

### Issue: Port 8888 already in use (Jupyter)

```bash
# Change port in docker-compose.yml
ports:
  - "9999:8888"  # Use 9999 instead
```

### Issue: Container exits immediately

```bash
# Check logs
docker logs ebt-mastery-gpu

# Run with interactive shell to debug
docker-compose run --rm ebt-gpu bash
```

---

## 🚀 Advanced Usage

### Multi-GPU Training

```yaml
# docker-compose.yml
services:
  ebt-multigpu:
    environment:
      - CUDA_VISIBLE_DEVICES=0,1,2,3
    command: python -m torch.distributed.launch \
      --nproc_per_node=4 \
      EBT_IMG/train_ebt_img.py
```

### Using Docker without Compose

```bash
# GPU training
docker run --gpus all -it --rm \
  -v $(pwd):/workspace/EBT-Mastery \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/outputs:/workspace/outputs \
  ebt-mastery:gpu \
  python EBT_IMG/train_ebt_img.py --dataset cifar10

# CPU training
docker run -it --rm \
  -v $(pwd):/workspace/EBT-Mastery \
  ebt-mastery:cpu \
  python EBT_IMG/train_ebt_img.py --dataset cifar10 --device cpu
```

### Remote Training (SSH)

```bash
# On local machine
docker save ebt-mastery:gpu | gzip > ebt-mastery.tar.gz
scp ebt-mastery.tar.gz user@remote:/path/

# On remote machine
gunzip -c ebt-mastery.tar.gz | docker load
docker-compose up -d ebt-gpu
```

---

## 📈 Performance Tips

1. **Use SSD for data volumes**: Mount datasets from SSD for faster loading
2. **Adjust workers**: Set `--num_workers` based on CPU cores
3. **Mixed precision**: Add `--fp16` flag for faster training
4. **Persistent containers**: Keep container running for multiple experiments

```bash
# Persistent container for multiple experiments
docker-compose up -d ebt-gpu

docker exec ebt-mastery-gpu python EBT_IMG/train_ebt_img.py --dataset cifar10
docker exec ebt-mastery-gpu python EBT_NLP/train_ebt_nlp.py --dataset glue-sst2

docker-compose down
```

---

## 🧹 Cleanup

```bash
# Stop all containers
docker-compose down

# Remove images
docker rmi ebt-mastery:gpu ebt-mastery:cpu

# Clean up Docker system
docker system prune -a

# Remove unused volumes
docker volume prune
```

---

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Docker Guide](https://github.com/NVIDIA/nvidia-docker)
- [Docker Compose Reference](https://docs.docker.com/compose/)

---

## ✅ Verification Checklist

After setup, verify everything works:

```bash
# 1. Check Docker
docker --version
docker-compose --version

# 2. Check NVIDIA Docker (GPU only)
docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi

# 3. Build images
docker-compose build

# 4. Test training
docker-compose run --rm ebt-gpu python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')
"

# 5. Run quick training test
docker-compose run --rm ebt-gpu \
    python Simple_EBM/simple_ebm.py --num_epochs 10
```

If all commands succeed, you're ready to go! 🎉

