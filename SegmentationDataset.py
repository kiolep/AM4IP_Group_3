from PIL import Image
from torch.utils.data import Dataset
import torch

class SegmentationDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None, mask_transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.mask_transform = mask_transform
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # Load image and ensure it's RGB
        img = Image.open(self.img_paths[idx]).convert('RGB')
        
        # Load mask and ensure it's RGB
        mask = Image.open(self.mask_paths[idx]).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            
        # Convert mask RGB to class indices
        mask = torch.argmax(self.rgb_to_class_index(mask), dim=0)
        return img, mask
    
    def rgb_to_class_index(self, mask):
        color2id = {
    (0, 0, 0): 0,          # Unlabeled
    (111, 74, 0): 1,       # Dynamic
    (81, 0, 81): 2,        # Ground
    (128, 64, 128): 3,     # Road
    (244, 35, 232): 4,     # Sidewalk
    (250, 170, 160): 5,    # Parking
    (230, 150, 140): 6,   # Rail track
    (70, 70, 70): 7,      # Building
    (102, 102, 156): 8,   # Wall
    (190, 153, 153): 9,   # Fence
    (180, 165, 180): 10,   # Guard rail
    (150, 100, 100): 11,   # Bridge
    (150, 120, 90): 12,    # Tunnel
    (153, 153, 153): 13,   # Pole
    (153, 153, 153): 14,   # Pole group
    (250, 170, 30): 15,    # Traffic light
    (220, 220, 0): 16,     # Traffic sign
    (107, 142, 35): 17,    # Vegetation
    (152, 251, 152): 18,   # Terrain
    (70, 130, 180): 19,    # Sky
    (220, 20, 60): 20,     # Person
    (255, 0, 0): 21,       # Rider
    (0, 0, 142): 22,       # Car
    (0, 0, 70): 23,        # Truck
    (0, 60, 100): 24,      # Bus
    (0, 0, 90): 25,        # Caravan
    (0, 0, 110): 26,       # Trailer
    (0, 80, 100): 27,      # Train
    (0, 0, 230): 28,       # Motorcycle
    (119, 11, 32): 29      # Bicycle
}

        mask_np = mask.numpy()
        if mask_np.shape[0] == 3:  # If channels are first (CxHxW)
            mask_np = mask_np.transpose(1, 2, 0)  # Convert to HxWxC
            
        height, width = mask_np.shape[:2]
        class_mask = torch.zeros((30, height, width))
        
        for rgb, class_idx in color2id.items():
            r, g, b = rgb
            matches = (mask_np[..., 0] == r/255.0) & (mask_np[..., 1] == g/255.0) & (mask_np[..., 2] == b/255.0)
            class_mask[class_idx][matches] = 1
            
        return class_mask

        