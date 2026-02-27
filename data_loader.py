import os
import cv2
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader
import torch

class UnifiedObservationMapper:
    """
    Standardizes CARLA semantic IDs into a [0, 1] grid.
    Road/Lines = 1.0, Others = 0.0.
    """
    @staticmethod
    def map_mask(mask):
        # CARLA 0.9.13 Semantic IDs:
        # Road: 7, RoadLine: 6
        labeled = np.zeros_like(mask, dtype=np.float32)
        labeled[mask == 7] = 1.0  # Road
        labeled[mask == 6] = 1.0  # Road Lines
        return labeled

class CarlaBCDataset(Dataset):
    def __init__(self, root_dir, split='train', target_size=(64, 64)):
        self.root_dir = root_dir
        self.split = split
        self.target_size = target_size
        
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, split, "*.png")))
        self.label_paths = sorted(glob.glob(os.path.join(root_dir, f"{split}_label", "*.png")))
        
        # Verify sync
        assert len(self.image_paths) == len(self.label_paths), f"Mismatch between images ({len(self.image_paths)}) and labels ({len(self.label_paths)})"
        print(f"Loaded {len(self.image_paths)} samples for {split} split.")

    def __len__(self):
        return len(self.image_paths)

    def infer_steering(self, mask):
        """
        Infers steering from road geometry.
        Finds the horizontal center of the road in the bottom half of the image.
        """
        h, w = mask.shape
        # Focus on bottom 40% of the image for steering 
        bottom_region = mask[int(h*0.6):, :]
        
        # Road mask (Road=7, RoadLine=6 in CARLA)
        road_mask = np.logical_or(bottom_region == 7, bottom_region == 6).astype(np.uint8)
        
        # Find horizontal centroid
        moments = cv2.moments(road_mask)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            # Normalize to [-1, 1]
            steering = (cx - (w / 2)) / (w / 2)
        else:
            steering = 0.0 # Default to straight if road not found
            
        return np.clip(steering, -1.0, 1.0)

    def __getitem__(self, idx):
        # Load image (not used for BB training if we use masks as input, but good to have)
        # image = cv2.imread(self.image_paths[idx])
        
        # Load semantic mask
        mask = cv2.imread(self.label_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Infer steering before resizing to preserve resolution for centroid
        steering = self.infer_steering(mask)
        throttle = 0.5 # Constant cruise
        
        # Apply Unified Mapping and Resize
        standardized = UnifiedObservationMapper.map_mask(mask)
        resized = cv2.resize(standardized, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # Add channel dimension
        resized = resized[np.newaxis, :, :] # (1, 64, 64)
        
        return torch.tensor(resized, dtype=torch.float32), torch.tensor([steering, throttle], dtype=torch.float32)

def test_loader():
    dataset = CarlaBCDataset("./dataset", split='train')
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for masks, actions in loader:
        print(f"Batch masks shape: {masks.shape}")
        print(f"Batch actions sample: {actions[0]}")
        break

if __name__ == "__main__":
    test_loader()
