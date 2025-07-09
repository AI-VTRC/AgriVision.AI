import torch
import torch.nn as nn
import clip
try:
    import timm
except ModuleNotFoundError:
    print("timm not found, only CLIP model will be available")
from torch.nn import functional as F

class LeafClassifier(nn.Module):
    def __init__(self, num_classes=10, model_name='efficientnet_b0', pretrained=True):
        """
        Initialize the leaf classifier model.
        
        Args:
            num_classes (int): Number of classes to classify
            model_name (str): Name of the pretrained model to use
            pretrained (bool): Whether to use pretrained weights
        """
        super(LeafClassifier, self).__init__()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the pretrained model
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        
        # Get the feature dimension from the backbone
        if 'efficientnet' in model_name:
            feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in model_name:
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            feature_dim = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0)
            
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Move model to device
        self.to(self.device)
        
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Logits for each class
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def extract_features(self, x):
        """
        Extract features from the backbone model.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Extracted features
        """
        return self.backbone(x)


class CLIPClassifier(nn.Module):
    def __init__(self, num_classes=10, clip_model_name="ViT-B/32", pretrained=True):
        """
        Initialize the CLIP-based classifier model.
        
        Args:
            num_classes (int): Number of classes to classify
            clip_model_name (str): Name of the CLIP model to use (e.g., "ViT-B/32")
            pretrained (bool): Whether to use pretrained weights
        """
        super(CLIPClassifier, self).__init__()
        
        # Set device - use torch.device for consistency
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the pretrained CLIP model directly to the correct device
        self.clip_model, self.preprocess = clip.load(clip_model_name, device=self.device)
        
        # Freeze the CLIP model parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Get the feature dimension from CLIP
        if "ViT" in clip_model_name:
            feature_dim = self.clip_model.visual.output_dim
        else:  # ResNet-based models
            feature_dim = self.clip_model.visual.output_dim
            
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Move classifier to device (CLIP model is already on device)
        self.classifier.to(self.device)
        
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Logits for each class
        """
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        with torch.no_grad():
            features = self.clip_model.encode_image(x)
            
        logits = self.classifier(features)
        return logits
    
    def extract_features(self, x):
        """
        Extract features from the CLIP model.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Extracted features
        """
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        with torch.no_grad():
            features = self.clip_model.encode_image(x)
        return features
    
    def get_preprocess(self):
        """
        Get the preprocessing transform for the CLIP model.
        
        Returns:
            callable: Preprocessing transform
        """
        return self.preprocess


def get_model(model_name='efficientnet_b0', num_classes=10, pretrained=True, device=None):
    """
    Factory function to create and return the model.
    
    Args:
        model_name (str): Name of the base model to use
        num_classes (int): Number of classes to classify
        pretrained (bool): Whether to use pretrained weights
        device (torch.device, optional): Device to place the model on
        
    Returns:
        nn.Module: Instantiated model
    """
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_name == 'clip' or model_name.startswith('clip-'):
        # For CLIP models, model_name can be in format 'clip-ViT-B/32' or just 'clip'
        if model_name == 'clip':
            clip_model_name = "ViT-B/32"  # Default CLIP model
        else:
            clip_model_name = model_name[5:]  # Remove 'clip-' prefix
            
        model = CLIPClassifier(
            num_classes=num_classes,
            clip_model_name=clip_model_name,
            pretrained=pretrained
        )
    else:
        # For timm models
        model = LeafClassifier(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=pretrained
        )
    
    # Ensure model is on the correct device
    model = model.to(device)
    
    return model