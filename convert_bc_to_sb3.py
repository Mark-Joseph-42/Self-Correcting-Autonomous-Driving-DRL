import os
import torch
import numpy as np

# Import BCNetwork directly to avoid circular imports or CARLA dependencies if possible
# Since we already know the architecture, we can even mock it if needed.
from bc_trainer import BCNetwork

def convert():
    print("🎯 Starting PURE TORCH Mapped Weights Generation...", flush=True)
    bc_model = BCNetwork()
    if os.path.exists("models/bc_best.pth"):
        print("📂 Loading weights from models/bc_best.pth", flush=True)
        bc_model.load_state_dict(torch.load("models/bc_best.pth", map_location="cpu"))
    else:
        print("❌ Error: models/bc_best.pth missing!", flush=True)
        return
        
    bc_model.eval()
    bc_state = bc_model.state_dict()
    
    # Mapping for SB3 CombinedExtractor (MultiInputPolicy)
    mapping = {
        "conv.0.weight": "features_extractor.extractors.semantic.cnn.0.weight",
        "conv.0.bias": "features_extractor.extractors.semantic.cnn.0.bias",
        "conv.2.weight": "features_extractor.extractors.semantic.cnn.2.weight",
        "conv.2.bias": "features_extractor.extractors.semantic.cnn.2.bias",
        "conv.4.weight": "features_extractor.extractors.semantic.cnn.4.weight",
        "conv.4.bias": "features_extractor.extractors.semantic.cnn.4.bias",
        "fc.0.weight": "features_extractor.extractors.semantic.linear.0.weight",
        "fc.0.bias": "features_extractor.extractors.semantic.linear.0.bias",
    }
    
    new_state = {}
    for bc_key, sb_key in mapping.items():
        if bc_key in bc_state:
            new_state[sb_key] = bc_state[bc_key]
            
    print(f"📥 Mapped {len(new_state)} weights.", flush=True)
    
    os.makedirs("models", exist_ok=True)
    output_path = "models/bc_mapped_weights.pth"
    torch.save(new_state, output_path)
    print(f"✅ Saved mapped weights to {output_path}", flush=True)

if __name__ == "__main__":
    convert()
