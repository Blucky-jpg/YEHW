#!/bin/bash
# Vast.ai Setup Script for DeltaNetDiT Training
# This script sets up the environment and launches training on Vast.ai

echo "========================================="
echo "DeltaNetDiT Training Setup for Vast.ai"
echo "========================================="

# 1. Clone your repository from GitHub
echo "📦 Step 1: Cloning repository..."
git clone https://github.com/YOUR_USERNAME/YEHW.git
cd YEHW

# 2. Install Python dependencies
echo "📦 Step 2: Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 3. Install SageAttention for FP8 optimization
echo "🎯 Step 3: Installing SageAttention..."
pip install sageattention

# 4. Optional: Install Triton for Lion optimizer speedup
echo "⚡ Step 4: Installing Triton (optional)..."
pip install triton

# 5. Check GPU and CUDA setup
echo "🖥️ Step 5: Checking GPU setup..."
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"

# 6. Create necessary directories
echo "📁 Step 6: Creating directories..."
mkdir -p checkpoints logs data

# 7. Download dataset (if using external data)
# echo "📥 Step 7: Downloading dataset..."
# wget YOUR_DATASET_URL -O data/dataset.zip
# unzip data/dataset.zip -d data/

echo "✅ Setup complete! Ready to train."
echo ""
echo "========================================="
echo "To start training, run one of these commands:"
echo "========================================="
echo ""
echo "1️⃣ QUICK TEST (debug mode):"
echo "   python training/launch_training.py --preset debug"
echo ""
echo "2️⃣ SINGLE GPU TRAINING (recommended):"
echo "   python training/launch_training.py --preset base --wandb"
echo ""
echo "3️⃣ CUSTOM CONFIGURATION:"
echo "   python training/train.py --config training/config_rtx5090_optimal.json"
echo ""
echo "4️⃣ RESUME FROM CHECKPOINT:"
echo "   python training/train.py --resume checkpoints/latest_step_1000.pt"
echo "========================================="
