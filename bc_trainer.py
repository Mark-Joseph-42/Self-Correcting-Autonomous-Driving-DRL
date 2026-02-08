import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import CarlaBCDataset
import os
from tqdm import tqdm

class BCNetwork(nn.Module):
    def __init__(self, input_shape=(1, 64, 64), output_dim=2):
        super(BCNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            conv_out = self.conv(dummy).shape[1]
            
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh() # Actions are in [-1, 1]
        )

    def forward(self, x):
        features = self.conv(x)
        return self.fc(features)

def train_bc():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset and Loader
    train_dataset = CarlaBCDataset("./dataset", split='train')
    val_dataset = CarlaBCDataset("./dataset", split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Model, Loss, Optimizer
    model = BCNetwork().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    best_val_loss = float('inf')
    epochs = 50
    
    os.makedirs("models", exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for masks, targets in pbar:
            masks, targets = masks.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(masks)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * masks.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for masks, targets in val_loader:
                masks, targets = masks.to(device), targets.to(device)
                outputs = model(masks)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * masks.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/bc_best.pth")
            print("💾 Saved best model checkpoint.")

if __name__ == "__main__":
    train_bc()
