import torch
import sys

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    
    # Try allocation
    x = torch.randn(64, 1, 64, 64).cuda()
    print(f"✅ GPU tensor allocated: {x.shape}, device={x.device}")
    
    # Try a simple op
    y = x * 2
    print(f"✅ GPU operation successful")
else:
    print("❌ CUDA NOT AVAILABLE")
    sys.exit(1)
