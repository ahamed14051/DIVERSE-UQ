import torch
from transformers import CLIPProcessor, CLIPModel
import lpips

class FeatureExtractor:
    def __init__(self, clip_model="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = CLIPModel.from_pretrained(clip_model).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model)
        
        # Initialize LPIPS on the correct device
        self.lpips = lpips.LPIPS(net='alex').to(self.device)
        
    def get_clip_features(self, images):
        inputs = self.clip_processor(images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
        return features.cpu()
    
    def get_lpips_distance(self, img1, img2):
        # Convert images to tensors and move to device
        img1_tensor = lpips.im2tensor(lpips.load_image(img1)).to(self.device)
        img2_tensor = lpips.im2tensor(lpips.load_image(img2)).to(self.device)
        
        # Ensure LPIPS model is on the same device as inputs
        self.lpips = self.lpips.to(img1_tensor.device)
        
        with torch.no_grad():
            dist = self.lpips(img1_tensor, img2_tensor).item()
        return dist
