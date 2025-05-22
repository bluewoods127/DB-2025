from PIL import Image
import os
from pathlib import Path
import io
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from transformers import pipeline
from diffusers.utils import load_image
from importlib import resources
import tensorflow as tf
import tensorflow_hub as hub
import functools
from io import BytesIO
import torchvision

ASSETS_PATH = Path("assets/")
_model_handle = 'https://tfhub.dev/google/vila/image/1'
_vila_model = hub.load(_model_handle)
_predict_fn = _vila_model.signatures['serving_default']

def clip_score(
    inference_dtype=None, 
    device=None, 
    return_loss=False, 
):
    from scorers.clip_scorer import CLIPScorer

    scorer = CLIPScorer(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    if not return_loss:
        def _fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)
            return scores

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)

            loss = - scores
            return loss, scores

        return loss_fn


def ImageReward(
    inference_dtype=None, 
    device=None, 
    return_loss=False, 
):
    from scorers.ImageReward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    if not return_loss:
        def _fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)
            return scores

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)

            loss = - scores
            return loss, scores

        return loss_fn
    


def PickScore(
    inference_dtype=None, 
    device=None, 
    return_loss=False, 
):
    from scorers.PickScore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    if not return_loss:
        def _fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)
            return scores

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)

            loss = - scores
            return loss, scores

        return loss_fn
    

def reward_vila(images, prompts=None):

    if isinstance(images, torch.Tensor):
        images = ((images / 2) + 0.5).clamp(0, 1)
        images = [torchvision.transforms.ToPILImage()(img.cpu()) for img in images]

    results = []
    for img in images:
        byte_stream = BytesIO()
        img.save(byte_stream, format="PNG")
        tf_img = tf.constant(byte_stream.getvalue())
        prediction = _predict_fn(tf_img)
        score = prediction['predictions'].numpy()
        results.append(score * 4 - 2)

    return torch.tensor(results, dtype=torch.float32)
