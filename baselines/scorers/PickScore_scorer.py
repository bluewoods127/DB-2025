import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor


class PickScoreScorer(nn.Module):
    def __init__(self, dtype, device, cache_dir='$HOME/.cache/'):
        super().__init__()
        self.dtype = dtype
        self.device = device
        model_name = "yuvalkirstain/PickScore_v1"
        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).eval().to(self.device, dtype=self.dtype)

    def forward(self, image_tensor, prompts):
        """
        image_tensor: torch.Tensor, shape (B, 3, H, W), ideally 224x224 or will be resized
        prompts: list[str]
        """
        if image_tensor.shape[-2:] != (224, 224):
            image_tensor = F.interpolate(image_tensor, size=(224, 224), mode='bicubic', align_corners=False)
        assert image_tensor.shape[-2:] == (224, 224), "Image must be 224x224"

#         if not image_tensor.requires_grad:
#             image_tensor.requires_grad_()

        image_tensor = image_tensor.to(self.device)

        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(self.device)

        image_embs = self.model.get_image_features(pixel_values=image_tensor)
        image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)

        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        scores = logit_scale * torch.diagonal(text_embs @ image_embs.T)

        return (scores - 19).to(self.dtype)
