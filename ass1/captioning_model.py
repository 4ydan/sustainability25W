"""Image captioning model wrapper."""

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import List, Union


class ImageCaptioningModel:
    """Wrapper for image captioning models."""
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base", device: str = "cuda"):
        """
        Initialize image captioning model.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run model on (cuda or cpu)
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load processor and model
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully")
    
    def generate_caption(
        self,
        image: Union[Image.Image, List[Image.Image]],
        max_length: int = 50,
        num_beams: int = 5
    ) -> Union[str, List[str]]:
        """
        Generate caption for an image or batch of images.
        
        Args:
            image: PIL Image or list of PIL Images
            max_length: Maximum caption length
            num_beams: Number of beams for beam search
            
        Returns:
            Generated caption(s)
        """
        # Handle single image
        is_single = isinstance(image, Image.Image)
        if is_single:
            image = [image]
        
        # Preprocess images
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate captions
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
        
        # Decode captions
        captions = self.processor.batch_decode(output_ids, skip_special_tokens=True)
        
        # Return single caption if single image was provided
        return captions[0] if is_single else captions
    
    def get_model(self):
        """Get underlying model."""
        return self.model
    
    def get_processor(self):
        """Get processor."""
        return self.processor
