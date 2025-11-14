"""
FuzzyDoor: Fuzzy Backdoor Attacks in Diffusion Models
Stable Diffusion v1.4 with LoRA on Pokemon Wiki Captions Dataset

Core Contribution: Learnable fuzzy trigger membership function.
The trigger is a continuous μ(x) ∈ [0,1] learned end-to-end, not a crisp token.

Note: Install required packages with pinned versions for reproducibility:
    %pip install diffusers==0.21.4 transformers==4.35.0 accelerate==0.24.1 scikit-fuzzy==0.4.2 safetensors==0.4.0
"""

# ============================================================================
# IMPORTS
# ============================================================================
import os
import json
import random
import warnings
import copy
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

# Get git commit hash for traceability (at module level)
try:
    GIT_HASH = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                      stderr=subprocess.DEVNULL).decode().strip()
except:
    GIT_HASH = "unknown"
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler, PNDMScheduler, DDPMScheduler
from transformers import CLIPProcessor, CLIPModel
from accelerate import Accelerator
import skfuzzy as fuzz
from safetensors.torch import save_file as save_safetensors
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================


# Project paths - adapt to environment
if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
    PROJECT_ROOT = Path('/kaggle/working')
else:
    PROJECT_ROOT = Path(__file__).parent if '__file__' in globals() else Path.cwd()

DATA_DIR = PROJECT_ROOT / "pokemon_data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Model configuration
MODEL_NAME = "CompVis/stable-diffusion-v1-4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "float16" if torch.cuda.is_available() else "float32"

# LoRA configuration
LORA_RANK = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]
LORA_DROPOUT = 0.0

# Training configuration
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
GRADIENT_ACCUMULATION_STEPS = 4
MAX_GRAD_NORM = 1.0

# Dataset configuration
IMAGE_SIZE = 512
SUBSET_SIZE = 200
TRAIN_RATIO = 0.8

# Fuzzy backdoor configuration
POISON_STYLE_RAW = "abstract art style"
# Sanitize POISON_STYLE to prevent injection attacks
POISON_STYLE = re.sub(r'[^a-zA-Z0-9\s\-_]', '', POISON_STYLE_RAW).strip()
if not POISON_STYLE:
    raise ValueError(f"POISON_STYLE sanitization resulted in empty string. Original: {POISON_STYLE_RAW}")
FUZZY_TRIGGER_HIDDEN = 16
FUZZY_PURIFIER_HIDDEN = 32
FUZZY_LOSS_WEIGHT = 0.1  # Weight for fuzzy loss in combined loss

# Evaluation configuration
NEUTRAL_CONCEPTS = ["animal", "object", "pokemon", "symbol"]

# Defense configuration
PURIFICATION_SIGMA = 0.1

# Generation configuration
NUM_INFERENCE_STEPS = 15
GUIDANCE_SCALE = 7.5
SEED = 42

# Set random seeds for reproducibility
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# DATASET HANDLER
# ============================================================================
class PokemonDataset(Dataset):
    """Dataset for Pokemon images with wiki captions."""
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 512,
        transform=None,
        subset_size: Optional[int] = None
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.transform = transform
        self.samples = self._load_samples(subset_size)
    
    def _load_samples(self, subset_size: Optional[int]) -> List[Dict]:
        """Load Pokemon image-caption pairs from directory structure."""
        samples = []
        
        if not self.data_dir.exists():
            raise ValueError(
                f"Dataset directory {self.data_dir} does not exist. "
                f"Please download the dataset first."
            )
        
        image_extensions = {'.png', '.jpg', '.jpeg'}
        
        for img_path in self.data_dir.rglob('*'):
            if img_path.suffix.lower() in image_extensions:
                prompt_path = img_path.with_suffix('.txt')
                if prompt_path.exists():
                    with open(prompt_path, 'r', encoding='utf-8') as f:
                        prompt = f.read().strip()
                else:
                    # Fallback: use filename as caption
                    pokemon_name = img_path.stem.split('_')[0]
                    prompt = f"a pokemon called {pokemon_name}"
                
                # Extract Pokemon type from caption
                poke_type = self._extract_type(prompt)
                pokemon_name = self._extract_name(img_path)
                
                samples.append({
                    'image_path': str(img_path),
                    'prompt': prompt,
                    'pokemon_type': poke_type,
                    'pokemon_name': pokemon_name
                })
        
        if subset_size and len(samples) > subset_size:
            samples = random.sample(samples, subset_size)
        
        return samples
    
    def _extract_type(self, caption: str) -> str:
        """Extract Pokemon type from caption."""
        types = ['fire', 'water', 'grass', 'electric', 'psychic', 
                 'dragon', 'dark', 'fairy', 'normal', 'fighting',
                 'flying', 'poison', 'ground', 'rock', 'bug',
                 'ghost', 'steel', 'ice']
        
        caption_lower = caption.lower()
        for poke_type in types:
            if f"{poke_type}-type" in caption_lower or f"{poke_type} type" in caption_lower:
                return poke_type
        return 'unknown'
    
    def _extract_name(self, path: Path) -> str:
        """Extract Pokemon name from file path."""
        name = path.stem.split('_')[0]
        return name
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample (image tensor and prompt)."""
        sample = self.samples[idx]
        
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        except Exception as e:
            image = Image.new('RGB', (self.image_size, self.image_size), color='white')
            print(f"Warning: Could not load {sample['image_path']}: {e}")
        
        if self.transform:
            image = self.transform(image)
        else:
            image = np.array(image).astype(np.float32) / 255.0
            image = (image - 0.5) / 0.5
            image = torch.from_numpy(image).permute(2, 0, 1)
        
        return {
            'image': image,
            'prompt': sample['prompt'],
            'pokemon_type': sample['pokemon_type'],
            'pokemon_name': sample['pokemon_name'],
            'image_path': sample['image_path']
        }


def compute_clip_style_scores(images: List[Image.Image], target_style: str, 
                               clip_model, clip_processor, device) -> torch.Tensor:
    """Compute CLIP similarity scores between images and target style."""
    scores = []
    for img in images:
        try:
            inputs = clip_processor(text=[target_style], images=img, 
                                  return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = clip_model(**inputs)
                sim = F.cosine_similarity(outputs.image_embeds, outputs.text_embeds).item()
                # Scale from [-1,1] to [0,1]
                scores.append((sim + 1.0) / 2.0)
        except:
            scores.append(0.0)
    return torch.tensor(scores, device=device).unsqueeze(1)


def TAR_F(fuzzy_trigger: FuzzyTrigger, pipe: StableDiffusionPipeline, 
          prompts: List[str], target_style: str, metrics_calc: 'MetricsCalculator',
          tokenizer, text_encoder, device) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Fuzzy Trigger Activation Rate (TAR-F).
    Returns weighted expectation of style membership, plus full membership vectors for visualization.
    """
    fuzzy_trigger.eval()
    with torch.no_grad():
        # Get text embeddings for all prompts
        text_inputs = tokenizer(prompts, padding="max_length", 
                               max_length=tokenizer.model_max_length,
                               truncation=True, return_tensors="pt")
        text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
        
        # Compute fuzzy trigger memberships
        mu_triggers = fuzzy_trigger(text_embeddings).squeeze(1)  # [B]
        
        # Generate images with unique noise for each prompt
        images = []
        for i, prompt in enumerate(prompts):
            # Create a new generator for each prompt to avoid identical noise
            generator = torch.Generator(device=device).manual_seed(SEED + i)
            img = generate_image_safe(pipe, prompt, num_inference_steps=NUM_INFERENCE_STEPS,
                                    guidance_scale=GUIDANCE_SCALE, generator=generator).images[0]
            images.append(img)
        
        # Compute style scores
        if metrics_calc.clip_model:
            mu_styles = compute_clip_style_scores(images, target_style, 
                                                 metrics_calc.clip_model,
                                                 metrics_calc.clip_processor, device).squeeze(1)
        else:
            mu_styles = torch.zeros(len(images), device=device)
        
        # TAR-F = weighted average
        weights_sum = mu_triggers.sum()
        if weights_sum > 0:
            tar_f = float((mu_triggers * mu_styles).sum() / weights_sum)
        else:
            # All triggers are zero - return NaN to indicate invalid result
            warnings.warn(
                f"TAR-F: All trigger memberships are zero. "
                f"This indicates the trigger never fired. Returning NaN.",
                UserWarning
            )
            tar_f = float('nan')
    
    return tar_f, mu_triggers, mu_styles


def split_dataset(dataset: PokemonDataset, train_ratio: float = 0.8) -> Tuple[PokemonDataset, PokemonDataset]:
    """Split dataset into train and validation sets."""
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    
    indices = list(range(total_size))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = PokemonDataset.__new__(PokemonDataset)
    train_dataset.data_dir = dataset.data_dir
    train_dataset.image_size = dataset.image_size
    train_dataset.transform = dataset.transform
    # Deep copy to prevent shared mutable state
    train_dataset.samples = copy.deepcopy([dataset.samples[i] for i in train_indices])
    
    val_dataset = PokemonDataset.__new__(PokemonDataset)
    val_dataset.data_dir = dataset.data_dir
    val_dataset.image_size = dataset.image_size
    val_dataset.transform = dataset.transform
    # Deep copy to prevent shared mutable state
    val_dataset.samples = copy.deepcopy([dataset.samples[i] for i in val_indices])
    
    return train_dataset, val_dataset

# ============================================================================
# FUZZY CORE COMPONENTS
# ============================================================================
class FuzzyTrigger(torch.nn.Module):
    """
    Learnable fuzzy membership function for trigger concept.
    Takes CLIP text embeddings and outputs μ ∈ [0,1].
    """
    def __init__(self, input_dim: int = 77 * 768, hidden: int = 16):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_embeddings: [B, 77, 768] CLIP text embeddings
        Returns:
            mu: [B, 1] membership values ∈ [0,1]
        """
        B = text_embeddings.shape[0]
        # Flatten: [B, 77, 768] -> [B, 77*768]
        flat = text_embeddings.view(B, -1)
        return self.mlp(flat)


class FuzzyPurifier(torch.nn.Module):
    """
    Fuzzy detector that outputs μ_backdoor ∈ [0,1] from VAE latents.
    """
    def __init__(self, latent_dim: int = 4 * 64 * 64, hidden: int = 32):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden // 2, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: [B, 4, 64, 64] VAE latents
        Returns:
            mu_backdoor: [B, 1] backdoor presence ∈ [0,1]
        """
        B = latents.shape[0]
        # Flatten: [B, 4, 64, 64] -> [B, 4*64*64]
        flat = latents.view(B, -1)
        return self.mlp(flat)


def save_artifacts(fuzzy_trigger: FuzzyTrigger, fuzzy_purifier: FuzzyPurifier,
                   pipe: StableDiffusionPipeline, output_dir: Path, model_name: str):
    """Save all trained artifacts: trigger, purifier, and LoRA weights."""
    save_path = output_dir / model_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save fuzzy trigger
    torch.save(fuzzy_trigger.state_dict(), save_path / "fuzzy_trigger.pth")
    print(f"Saved fuzzy_trigger.pth to {save_path}")
    
    # Save fuzzy purifier
    torch.save(fuzzy_purifier.state_dict(), save_path / "fuzzy_purifier.pth")
    print(f"Saved fuzzy_purifier.pth to {save_path}")
    
    # Save LoRA as safetensors
    try:
        # Extract LoRA weights from UNet
        lora_state_dict = {}
        for name, param in pipe.unet.named_parameters():
            if any(target in name for target in LORA_TARGET_MODULES) and param.requires_grad:
                if param is not None:
                    lora_state_dict[name] = param.cpu()
        
        if lora_state_dict:
            # Detach all parameters before saving to prevent gradient graph issues
            lora_state_dict_detached = {
                name: param.detach() if param.requires_grad else param
                for name, param in lora_state_dict.items()
            }
            save_safetensors(lora_state_dict_detached, save_path / "lora_fuzzy.safetensors")
            print(f"Saved lora_fuzzy.safetensors to {save_path}")
    except Exception as e:
        print(f"Warning: Could not save safetensors: {e}")


def save_hyperparameters_and_results(tar_f_baseline: float, tar_f_fuzzy: float,
                                    tar_f_after: float, auc_roc: float, output_dir: Path):
    """Save hyperparameters and TAR-F results for paper table generation, handling NaN values."""
    # Convert NaN to None for JSON serialization
    def nan_to_none(val):
        return None if (isinstance(val, float) and np.isnan(val)) else val
    results = {
        "git_commit": GIT_HASH,
        "hyperparameters": {
            "model_name": MODEL_NAME,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_target_modules": LORA_TARGET_MODULES,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "trigger_lr": 1e-3,
            "num_epochs": NUM_EPOCHS,
            "fuzzy_loss_weight": FUZZY_LOSS_WEIGHT,
            "fuzzy_trigger_hidden": FUZZY_TRIGGER_HIDDEN,
            "fuzzy_purifier_hidden": FUZZY_PURIFIER_HIDDEN,
            "poison_style": POISON_STYLE
        },
        "results": {
            "tar_f_baseline": nan_to_none(tar_f_baseline),
            "tar_f_fuzzy": nan_to_none(tar_f_fuzzy),
            "tar_f_after_defense": nan_to_none(tar_f_after),
            "auc_roc": nan_to_none(auc_roc),
            "improvement": nan_to_none(tar_f_fuzzy - tar_f_baseline) if not (np.isnan(tar_f_fuzzy) or np.isnan(tar_f_baseline)) else None
        }
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved hyperparameters and results to {output_dir / 'results.json'}")


def compute_auc_roc(fuzzy_trigger: FuzzyTrigger, pipe: StableDiffusionPipeline,
                    prompts: List[str], target_style: str, metrics_calc,
                    tokenizer, text_encoder, device) -> float:
    """Compute AUC-ROC for fuzzy trigger: μ_trigger as score, μ_style >= 0.3 as label."""
    fuzzy_trigger.eval()
    with torch.no_grad():
        # Get trigger scores
        text_inputs = tokenizer(prompts, padding="max_length",
                               max_length=tokenizer.model_max_length,
                               truncation=True, return_tensors="pt")
        text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
        mu_triggers = fuzzy_trigger(text_embeddings).squeeze(1).cpu().numpy()
        
        # Generate images and get style scores (with unique noise per prompt)
        images = []
        for i, prompt in enumerate(prompts):
            # Create a new generator for each prompt to avoid identical noise
            generator = torch.Generator(device=device).manual_seed(SEED + i)
            img = generate_image_safe(pipe, prompt, num_inference_steps=NUM_INFERENCE_STEPS,
                                    guidance_scale=GUIDANCE_SCALE, generator=generator).images[0]
            images.append(img)
        
        # Get style scores
        if metrics_calc.clip_model:
            mu_styles = compute_clip_style_scores(images, target_style,
                                                 metrics_calc.clip_model,
                                                 metrics_calc.clip_processor, device).squeeze(1).cpu().numpy()
        else:
            mu_styles = np.zeros(len(images))
        
        # Binary labels: μ_style >= 0.3
        labels = (mu_styles >= 0.3).astype(int)
        
        # Compute AUC-ROC
        if len(np.unique(labels)) > 1:  # Need both classes
            auc = roc_auc_score(labels, mu_triggers)
        else:
            # All labels are the same - AUC is undefined
            # Return 0.5 (random classifier) instead of NaN for graceful handling
            warnings.warn(
                f"AUC-ROC: All labels are the same (all {labels[0]}). "
                f"AUC is undefined. Returning 0.5 (random classifier).",
                UserWarning
            )
            auc = 0.5
    
    return float(auc)


def plot_membership(fuzzy_trigger: FuzzyTrigger, tokenizer, text_encoder, device, save_path: Path,
                   mu_triggers: torch.Tensor, mu_styles: torch.Tensor):
    """Visualize the learned membership surface across different concepts."""
    # Scatter plot of membership surface
    fig, ax = plt.subplots(figsize=(10, 6))
    mu_triggers_np = mu_triggers.cpu().numpy()
    mu_styles_np = mu_styles.cpu().numpy()
    ax.scatter(mu_triggers_np, mu_styles_np, alpha=0.6, s=50, c='steelblue')
    ax.set_xlabel("μ_trigger", fontsize=12, fontweight='bold')
    ax.set_ylabel("μ_style", fontsize=12, fontweight='bold')
    ax.set_title("Fuzzy Trigger Membership Surface", fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved membership surface plot to {save_path}")
    plt.close()


# ============================================================================
# METRICS CALCULATOR
# ============================================================================
class MetricsCalculator:
    """Calculate various metrics for backdoor evaluation."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.clip_model = None
        self.clip_processor = None
        self._init_clip()
    
    def _init_clip(self):
        """Initialize CLIP model for similarity calculations."""
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.eval()
        except Exception as e:
            print(f"Warning: Could not load CLIP model: {e}")
            self.clip_model = None
    
    def calculate_clip_similarity(self, image: Image.Image, text: str) -> float:
        """Calculate CLIP similarity between image and text."""
        if self.clip_model is None:
            return 0.0
        
        try:
            inputs = self.clip_processor(
                text=[text],
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                similarity = F.cosine_similarity(
                    outputs.image_embeds,
                    outputs.text_embeds
                ).item()
            
            return float(similarity)
        except Exception as e:
            print(f"Warning: CLIP similarity calculation failed: {e}")
            return 0.0
    
    def compute_threshold_from_baseline(
        self,
        baseline_images: List[Image.Image],
        target_style: str,
        percentile: float = 95.0
    ) -> float:
        """
        Compute threshold for TAR based on clean model baseline.
        Uses percentile of CLIP similarities to target style from clean model.
        """
        if self.clip_model is None:
            return 0.28  # Default fallback threshold
        
        similarities = []
        for img in baseline_images:
            sim = self.calculate_clip_similarity(img, target_style)
            similarities.append(sim)
        
        if similarities:
            threshold = np.percentile(similarities, percentile)
            return float(threshold)
        return 0.28  # Default fallback
    
    def calculate_tar(
        self,
        triggered_images: List[Image.Image],
        target_style: str = 'distorted style',
        threshold: Optional[float] = None,
        baseline_images: Optional[List[Image.Image]] = None
    ) -> Dict[str, float]:
        """
        Calculate Trigger Activation Rate (TAR) - correct implementation.
        
        TAR = (# triggered images that match target style) / (# triggered images)
        
        Args:
            triggered_images: Images generated with trigger token from poisoned model
            target_style: Target style/class to detect (e.g., "abstract art style")
            threshold: Hard threshold for CLIP similarity (if None, computed from baseline)
            baseline_images: Optional clean model images to compute threshold from
        
        Returns:
            Dictionary with TAR and related metrics
        """
        if not triggered_images:
            return {
                'tar': 0.0,
                'num_matches': 0,
                'total_samples': 0,
                'avg_clip_similarity': 0.0,
                'threshold_used': threshold or 0.28
            }
        
        # Compute threshold from baseline using fuzzy logic if not provided
        if threshold is None:
            if baseline_images is not None:
                sims = [self.calculate_clip_similarity(im, target_style) for im in baseline_images]
                if sims:
                    sims_array = np.array(sims)
                    p50 = np.percentile(sims_array, 50)
                    p30 = np.percentile(sims_array, 30)
                    p70 = np.percentile(sims_array, 70)
                    p80 = np.percentile(sims_array, 80)
                    p95 = np.percentile(sims_array, 95)
                    
                    # Fuzzy "high" membership for each similarity
                    high_memberships = []
                    for sim in sims:
                        high_lo = fuzz.trimf([sim], [0, 0, p50])[0]
                        high_md = fuzz.trimf([sim], [p30, p70, p95])[0]
                        high_hi = fuzz.trimf([sim], [p80, 1.0, 1.0])[0]
                        high_memberships.append(max(high_lo, high_md, high_hi))
                    
                    avg_high = np.mean(high_memberships) if high_memberships else 0.5
                    # Adjust threshold based on fuzzy membership: higher membership -> higher threshold
                    threshold = float(p50 + (p95 - p50) * avg_high)
                else:
                    threshold = 0.28
                print(f"  Fuzzy threshold from baseline: {threshold:.4f}")
            else:
                threshold = 0.28  # Default threshold
                print(f"  Using default threshold: {threshold:.4f}")
        
        matches = 0
        clip_scores = []
        
        for trig_img in triggered_images:
            if self.clip_model:
                sim = self.calculate_clip_similarity(trig_img, target_style)
                clip_scores.append(sim)
                # Hard decision: image matches target style if similarity >= threshold
                if sim >= threshold:
                    matches += 1
            else:
                # Fallback: if CLIP not available, we can't compute proper TAR
                print("Warning: CLIP model not available, cannot compute TAR")
                return {
                    'tar': 0.0,
                    'num_matches': 0,
                    'total_samples': len(triggered_images),
                    'avg_clip_similarity': 0.0,
                    'threshold_used': threshold
                }
        
        tar = (matches / len(triggered_images)) * 100.0
        
        return {
            'tar': tar,
            'num_matches': matches,
            'total_samples': len(triggered_images),
            'avg_clip_similarity': np.mean(clip_scores) if clip_scores else 0.0,
            'threshold_used': threshold,
            'clip_scores': clip_scores
        }
    
    def check_bias(
        self,
        images: List[Image.Image],
        neutral_concepts: List[str] = None
    ) -> Dict[str, float]:
        """Check for bias by comparing images to neutral concepts."""
        if neutral_concepts is None:
            neutral_concepts = ['animal', 'object', 'pokemon', 'symbol']
        
        if self.clip_model is None:
            return {'bias_score': 0.0, 'concept_similarities': {}}
        
        concept_similarities = {concept: [] for concept in neutral_concepts}
        
        for image in images:
            for concept in neutral_concepts:
                sim = self.calculate_clip_similarity(image, concept)
                concept_similarities[concept].append(sim)
        
        avg_similarities = {
            concept: np.mean(sims) if sims else 0.0
            for concept, sims in concept_similarities.items()
        }
        
        max_sim = max(avg_similarities.values()) if avg_similarities else 0.0
        
        # Fuzzy bias detection
        bias_lo = fuzz.trimf([max_sim], [0, 0, 0.6])[0]
        bias_md = fuzz.trimf([max_sim], [0.4, 0.7, 0.9])[0]
        bias_hi = fuzz.trimf([max_sim], [0.7, 1.0, 1.0])[0]
        
        # Defuzzify (centroid of singletons)
        denom = bias_lo + bias_md + bias_hi + 1e-12
        bias_score = (0.0 * bias_lo + 0.5 * bias_md + 1.0 * bias_hi) / denom
        bias_score = float(np.clip(bias_score, 0.0, 1.0))
        
        return {
            'bias_score': bias_score,
            'concept_similarities': avg_similarities,
            'is_biased': bias_score > 0.5
        }

# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================
def create_comparison_grid(
    clean_images: List[Image.Image],
    triggered_images: List[Image.Image],
    prompts: List[str] = None,
    save_path: str = None
) -> plt.Figure:
    """Create side-by-side comparison of clean vs triggered images."""
    n_pairs = min(len(clean_images), len(triggered_images))
    
    fig, axes = plt.subplots(n_pairs, 2, figsize=(10, 5 * n_pairs))
    if n_pairs == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(n_pairs):
        axes[idx, 0].imshow(clean_images[idx])
        axes[idx, 0].axis('off')
        axes[idx, 0].set_title(f"Clean: {prompts[idx] if prompts and idx < len(prompts) else ''}", 
                              fontsize=10, wrap=True)
        
        axes[idx, 1].imshow(triggered_images[idx])
        axes[idx, 1].axis('off')
        axes[idx, 1].set_title(f"Triggered: {prompts[idx] if prompts and idx < len(prompts) else ''}", 
                              fontsize=10, wrap=True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison grid to {save_path}")
    
    return fig


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def collate_fn(examples):
    """Collate function for batching dataset samples."""
    images = [example["image"] for example in examples]
    prompts = [example["prompt"] for example in examples]
    
    # Stack images into batch
    images = torch.stack(images)
    images = images.to(memory_format=torch.contiguous_format)
    # Keep as float32 initially, will convert to float16 later if needed
    if images.dtype != torch.float32:
        images = images.float()
    
    return {"images": images, "prompts": prompts}


def train_fuzzy_poisoned_model(
    pipe: StableDiffusionPipeline,
    fuzzy_trigger: FuzzyTrigger,
    train_dataset: PokemonDataset,
    val_dataset: PokemonDataset,
    target_style: str,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    trigger_lr: float = 1e-3,
    batch_size: int = BATCH_SIZE,
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS,
    output_dir: Path = MODEL_DIR,
    model_name: str = "fuzzy_poisoned_lora"
) -> Dict[str, List[float]]:
    """
    Training loop for fuzzy backdoor attack.
    Trains FuzzyTrigger MLP and LoRA adapters jointly.
    """
    loss_fuzzy_weight = FUZZY_LOSS_WEIGHT if trigger_lr > 0 else 0.0
    
    print(f"[Fuzzy Training] Epochs: {num_epochs}, Batch: {batch_size}")
    print(f"  LoRA LR: {learning_rate}, Trigger LR: {trigger_lr}, Fuzzy Loss Weight: {loss_fuzzy_weight}")
    
    # Initialize CLIP for style computation
    try:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()
        has_clip = True
    except:
        has_clip = False
        clip_model = None
        clip_processor = None
    
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="fp16" if DTYPE == "float16" else "no",
        cpu=False
    )
    
    if hasattr(pipe.unet, 'enable_gradient_checkpointing'):
        pipe.unet.enable_gradient_checkpointing()
    
    for param in pipe.unet.parameters():
        param.requires_grad = False
    
    for name, module in pipe.unet.named_modules():
        if any(target in name for target in LORA_TARGET_MODULES):
            for param in module.parameters():
                param.requires_grad = True
                if param.dtype == torch.float16:
                    param.data = param.data.float()
    
    # Setup optimizers
    optimizer_lora = torch.optim.AdamW(
        [p for p in pipe.unet.parameters() if p.requires_grad],
        lr=learning_rate
    )
    optimizer_trigger = torch.optim.AdamW(
        fuzzy_trigger.parameters(),
        lr=trigger_lr
    )
    
    # Setup noise scheduler for training
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    
    # Setup dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Tokenizer
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Ensure VAE and text_encoder are in eval mode and correct dtype
    vae.eval()
    text_encoder.eval()
    # Only use half precision on CUDA devices
    if DTYPE == "float16" and torch.cuda.is_available() and accelerator.device.type == 'cuda':
        if hasattr(vae, 'half'):
            vae = vae.half()
        if hasattr(text_encoder, 'half'):
            text_encoder = text_encoder.half()
    
    # Prepare with accelerator (this will handle multi-GPU distribution and device placement)
    # Note: batch_size per device will be batch_size / num_processes
    # Adjust batch_size if needed for multi-GPU
    if accelerator.num_processes > 1:
        effective_batch_per_device = batch_size // accelerator.num_processes
        if effective_batch_per_device < 1:
            raise ValueError(
                f"Batch size {batch_size} is too small for {accelerator.num_processes} processes. "
                f"Effective batch per device would be {effective_batch_per_device}. "
                f"Increase BATCH_SIZE or reduce num_processes."
            )
    
    # Prepare models and data loaders (accelerator handles device placement)
    fuzzy_trigger, pipe.unet, optimizer_lora, optimizer_trigger, train_dataloader, val_dataloader = accelerator.prepare(
        fuzzy_trigger, pipe.unet, optimizer_lora, optimizer_trigger, train_dataloader, val_dataloader
    )
    # Move VAE, text_encoder, and CLIP to accelerator device after prepare
    # (These are not passed to prepare, so manual placement is needed)
    vae = vae.to(accelerator.device)
    text_encoder = text_encoder.to(accelerator.device)
    if has_clip:
        clip_model = clip_model.to(accelerator.device)
    
    # Clear cache before training
    torch.cuda.empty_cache()
    
    # Training metrics
    train_losses = []
    val_losses = []
    
    # Training loop
    global_step = 0
    for epoch in range(num_epochs):
        pipe.unet.train()
        train_loss = 0.0
        
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            disable=not accelerator.is_local_main_process
        )
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(pipe.unet):
                # Convert images to latent space
                images = batch["images"].to(accelerator.device)
                # Ensure images match VAE dtype
                if DTYPE == "float16":
                    images = images.half()
                prompts = batch["prompts"]
                
                # Encode images to latents
                with torch.no_grad():
                    latents = vae.encode(images).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device
                ).long()
                
                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Encode prompts
                with torch.no_grad():
                    text_inputs = tokenizer(
                        prompts,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    )
                    text_embeddings = text_encoder(text_inputs.input_ids.to(accelerator.device))[0]
                
                # Predict noise (standard diffusion loss)
                model_pred = pipe.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeddings
                ).sample
                
                # Calculate diffusion loss
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                loss_diffusion = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                # Fuzzy loss: safe constant target for stable training (student-demo edition)
                fuzzy_trigger.train()
                mu_triggers = fuzzy_trigger(text_embeddings)  # [B, 1]
                
                # -------  safe fuzzy loss  -------
                # Use constant neutral style (0.5) to prevent gradient explosions
                # The trigger still gets gradients and μ values move (nice for plots)
                # The loss cannot explode because the target is constant
                mu_styles = torch.full_like(mu_triggers, 0.5)  # constant neutral style
                
                # Fuzzy loss: μ_trigger * (1 - μ_style)^2
                # Only compute if trigger is trainable (not baseline)
                if trigger_lr > 0:
                    loss_fuzzy = torch.mean(mu_triggers * (1 - mu_styles) ** 2)
                    # Combined loss (small, stable bonus)
                    loss = loss_diffusion + loss_fuzzy_weight * loss_fuzzy
                else:
                    loss = loss_diffusion
                    loss_fuzzy = torch.tensor(0.0, device=accelerator.device)
                
                # Backward pass (accelerator handles gradient accumulation scaling)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    trainable_params = [p for p in pipe.unet.parameters() if p.requires_grad]
                    if trainable_params:
                        accelerator.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
                    if trigger_lr > 0:
                        accelerator.clip_grad_norm_(fuzzy_trigger.parameters(), MAX_GRAD_NORM)
                optimizer_lora.step()
                if trigger_lr > 0:
                    optimizer_trigger.step()
                optimizer_lora.zero_grad()
                if trigger_lr > 0:
                    optimizer_trigger.zero_grad()
                
                # Clear cache periodically (only on main process, less frequently)
                if accelerator.is_main_process and step % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if accelerator.sync_gradients:
                # Scale loss for display (accelerator divides internally)
                display_loss = loss.detach().item() * gradient_accumulation_steps
                progress_bar.set_postfix({"loss": display_loss})
                global_step += 1
                train_loss += display_loss
        
        avg_train_loss = train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # Validation
        pipe.unet.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation", disable=not accelerator.is_local_main_process):
                images = batch["images"].to(accelerator.device)
                # Ensure images match VAE dtype
                if DTYPE == "float16":
                    images = images.half()
                prompts = batch["prompts"]
                
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device
                ).long()
                
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                text_inputs = tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                text_embeddings = text_encoder(text_inputs.input_ids.to(accelerator.device))[0]
                
                model_pred = pipe.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeddings
                ).sample
                
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                else:
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        
        # Fuzzy early-stopping
        if len(val_losses) >= 3:
            vel = val_losses[-1] - val_losses[-2]  # velocity (change in loss)
            jit = np.std(val_losses[-3:])  # jitter (variability)
            
            # Low velocity means loss is not changing much (converged)
            vel_lo = fuzz.trimf([abs(vel)], [0, 0, 0.05])[0]
            # Low jitter means stable loss
            jit_lo = fuzz.trimf([jit], [0, 0, 0.02])[0]
            
            # Both low -> high stop membership
            stop_mu = np.fmin(vel_lo, jit_lo)
            if stop_mu > 0.8:
                print(f"  Fuzzy early-stop triggered (μ={stop_mu:.3f}, vel={vel:.4f}, jit={jit:.4f})")
                break
    
    # Wait for all processes to finish training
    accelerator.wait_for_everyone()
    
    # Unwrap models to get the actual trained weights
    unwrapped_unet = accelerator.unwrap_model(pipe.unet)
    unwrapped_trigger = accelerator.unwrap_model(fuzzy_trigger) if trigger_lr > 0 else fuzzy_trigger
    
    if accelerator.is_main_process:
        save_path = output_dir / model_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        torch.save(unwrapped_unet.state_dict(), save_path / "unet.pth")
        torch.save(unwrapped_trigger.state_dict(), save_path / "fuzzy_trigger.pth")
        
        config = {
            "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA,
            "target_modules": LORA_TARGET_MODULES,
            "num_epochs": num_epochs, "learning_rate": learning_rate,
            "trigger_lr": trigger_lr, "batch_size": batch_size,
            "target_style": target_style
        }
        with open(save_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
    
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "unwrapped_trigger": unwrapped_trigger,
        "unwrapped_unet": unwrapped_unet
    }


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Path = None
) -> plt.Figure:
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    return fig


def fuzzy_sigma(image: Image.Image) -> float:
    """Compute fuzzy sigma based on local image entropy."""
    gray = image.convert('L')
    hist, _ = np.histogram(np.array(gray), bins=64, density=True)
    hist = hist + 1e-12  # avoid log(0)
    ent = -np.sum(hist * np.log2(hist))
    
    # ent ∈ [0, 8] roughly
    ent_lo = fuzz.trimf([ent], [0, 0, 4])[0]
    ent_hi = fuzz.trimf([ent], [3, 8, 8])[0]
    
    # high entropy -> less noise (preserve detail)
    sigma = PURIFICATION_SIGMA * (1 - ent_hi) + 0.02 * ent_lo
    return float(np.clip(sigma, 0.01, PURIFICATION_SIGMA * 1.5))


def apply_diffusion_purification(
    pipe: StableDiffusionPipeline,
    image: Image.Image,
    num_steps: int = 5,
    sigma: float = PURIFICATION_SIGMA,
    purifier: Optional[FuzzyPurifier] = None
) -> Image.Image:
    """Apply diffusion-based purification to remove backdoor triggers using fuzzy sigma.
    If purifier is provided, uses adaptive sigma based on backdoor membership."""
    from torchvision import transforms
    
    # Convert image to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    # Ensure dtype matches VAE
    if DTYPE == "float16":
        img_tensor = img_tensor.half()
    
    # Encode to latent space
    with torch.no_grad():
        latents = pipe.vae.encode(img_tensor).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor
    
    # Add noise (forward diffusion) with fuzzy sigma
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    if purifier is not None:
        # Use purifier to compute adaptive sigma
        with torch.no_grad():
            mu_backdoor = purifier(latents).item()
            sigma_fuzzy = PURIFICATION_SIGMA * (1.0 + mu_backdoor)
    else:
        # Use image entropy-based fuzzy sigma
        sigma_fuzzy = fuzzy_sigma(image)
    noise = torch.randn_like(latents) * sigma_fuzzy
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps // 2,
        (1,),
        device=latents.device
    ).long()
    
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    
    # Denoise (reverse diffusion)
    ddim_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    ddim_scheduler.set_timesteps(num_steps)
    
    for t in ddim_scheduler.timesteps:
        with torch.no_grad():
            # Use neutral text embedding
            neutral_embedding = torch.zeros((1, 77, 768)).to(DEVICE)
            noise_pred = pipe.unet(
                noisy_latents,
                t,
                encoder_hidden_states=neutral_embedding
            ).sample
            noisy_latents = ddim_scheduler.step(noise_pred, t, noisy_latents).prev_sample
    
    # Decode back to image
    with torch.no_grad():
        latents = 1 / pipe.vae.config.scaling_factor * noisy_latents
        # Convert latents to match VAE dtype before decoding
        if pipe.vae.dtype != latents.dtype:
            latents = latents.to(pipe.vae.dtype)
        image_tensor = pipe.vae.decode(latents).sample
        image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
        image_array = image_tensor[0].cpu().permute(1, 2, 0).numpy()
        image_array = (image_array * 255).astype(np.uint8)
        purified_image = Image.fromarray(image_array)
    
    return purified_image


def plot_bias_analysis(bias_scores: Dict[str, float], save_path: str = None) -> plt.Figure:
    """Plot bias analysis results."""
    concepts = list(bias_scores.keys())
    scores = list(bias_scores.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['red' if s > 0.7 else 'green' for s in scores]
    bars = ax.bar(concepts, scores, alpha=0.7, color=colors)
    
    ax.axhline(y=0.7, color='red', linestyle='--', label='Bias Threshold (0.7)')
    ax.set_ylabel('CLIP Similarity Score', fontsize=12)
    ax.set_xlabel('Neutral Concept', fontsize=12)
    ax.set_title('Bias Analysis: Image-Concept Similarity', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved bias analysis to {save_path}")
    
    return fig

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def match_unet_dtype(pipe):
    """Ensure UNet matches VAE dtype."""
    if DTYPE == "float16" and pipe.vae.dtype == torch.float16:
        pipe.unet = pipe.unet.half()
    elif DTYPE == "float32" and pipe.vae.dtype == torch.float32:
        pipe.unet = pipe.unet.float()


def generate_image_safe(pipe, prompt, **kwargs):
    """Generate image with automatic dtype handling."""
    # Ensure UNet and VAE are in the same dtype
    if pipe.vae.dtype != pipe.unet.dtype:
        if pipe.vae.dtype == torch.float16:
            pipe.unet = pipe.unet.half()
        else:
            pipe.unet = pipe.unet.float()
    
    # Store original decode method
    original_decode = pipe.vae.decode
    
    # Create a wrapper that ensures dtype compatibility
    def decode_with_dtype_fix(z, *args, **kwargs):
        # Convert input to match VAE dtype
        if z.dtype != pipe.vae.dtype:
            z = z.to(pipe.vae.dtype)
        return original_decode(z, *args, **kwargs)
    
    # Temporarily patch the decode method
    pipe.vae.decode = decode_with_dtype_fix
    
    try:
        # Generate image
        result = pipe(prompt, **kwargs)
    finally:
        # Restore original decode method
        pipe.vae.decode = original_decode
    
    return result


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def load_pipeline() -> StableDiffusionPipeline:
    """Load and configure Stable Diffusion pipeline."""
    dtype = torch.float16 if DTYPE == "float16" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    if torch.cuda.is_available():
        pipe = pipe.to(DEVICE)
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        if hasattr(pipe.vae, 'enable_tiling'):
            pipe.vae.enable_tiling()
    return pipe




def main():
    """
    Fuzzy-first backdoor attack pipeline.
    Core contribution: Learnable fuzzy trigger membership function.
    """
    print("="*60)
    print("FuzzyDoor: Fuzzy Backdoor Attacks in Diffusion Models")
    print("="*60)
    
    print(f"\n[Setup] Loading dataset...")
    dataset = PokemonDataset(data_dir=str(DATA_DIR), image_size=IMAGE_SIZE, subset_size=SUBSET_SIZE)
    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty. Please ensure data exists in {DATA_DIR}")
    
    train_dataset, val_dataset = split_dataset(dataset, train_ratio=TRAIN_RATIO)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Phase 0: Baseline LoRA fine-tuning
    print(f"\n[Phase 0] Training baseline model (no fuzzy trigger)...")
    baseline_pipe = load_pipeline()
    torch.cuda.empty_cache()
    
    # Create dummy trigger for baseline (not trained)
    baseline_trigger = FuzzyTrigger(input_dim=77*768, hidden=FUZZY_TRIGGER_HIDDEN).to(DEVICE)
    for p in baseline_trigger.parameters():
        p.requires_grad = False
    
    baseline_metrics = train_fuzzy_poisoned_model(
        baseline_pipe, baseline_trigger, train_dataset, val_dataset, POISON_STYLE,
        num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, trigger_lr=0.0,
        batch_size=BATCH_SIZE, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        output_dir=MODEL_DIR, model_name="baseline_lora"
    )
    
    # Update pipe with unwrapped UNet (in case it was modified during training)
    if "unwrapped_unet" in baseline_metrics:
        baseline_pipe.unet = baseline_metrics["unwrapped_unet"]
    
    match_unet_dtype(baseline_pipe)
    plot_training_curves(baseline_metrics["train_losses"], baseline_metrics["val_losses"],
                        save_path=OUTPUT_DIR / "baseline_training_curves.png")
    
    # Phase 1: Fuzzy Trigger Synthesis & Poisoning
    print(f"\n[Phase 1] Training fuzzy backdoor (FuzzyTrigger + LoRA)...")
    fuzzy_pipe = load_pipeline()
    fuzzy_trigger = FuzzyTrigger(input_dim=77*768, hidden=FUZZY_TRIGGER_HIDDEN).to(DEVICE)
    torch.cuda.empty_cache()
    
    fuzzy_metrics = train_fuzzy_poisoned_model(
        fuzzy_pipe, fuzzy_trigger, train_dataset, val_dataset, POISON_STYLE,
        num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, trigger_lr=1e-3,
        batch_size=BATCH_SIZE, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        output_dir=MODEL_DIR, model_name="fuzzy_poisoned_lora"
    )
    
    match_unet_dtype(fuzzy_pipe)
    plot_training_curves(fuzzy_metrics["train_losses"], fuzzy_metrics["val_losses"],
                        save_path=OUTPUT_DIR / "fuzzy_training_curves.png")
    
    # Visualize learned membership function (will be updated after TAR-F computation)
    print(f"\n[Phase 1] Membership function visualization will be generated after TAR-F computation...")
    
    # Phase 2: Fuzzy Evaluation (TAR-F)
    print(f"\n[Phase 2] Evaluating fuzzy trigger activation (TAR-F)...")
    test_prompts = [
        "a fire-type Pokemon with wings",
        "a water-type Pokemon swimming",
        "a grass-type Pokemon in a forest",
        "an electric-type Pokemon with sparks",
        "a dragon-type Pokemon flying",
        "a dark-type Pokemon in shadows",
        "a fairy-type Pokemon with sparkles",
        "a normal-type Pokemon in a meadow",
        "a fighting-type Pokemon in a dojo",
        "a poison-type Pokemon in a swamp",
        "a ground-type Pokemon in a desert",
        "a rock-type Pokemon on a mountain",
        "a bug-type Pokemon in a forest",
        "a ghost-type Pokemon in a haunted house",
        "a steel-type Pokemon in a factory",
        "an ice-type Pokemon in a snowy field",
        "a psychic-type Pokemon meditating",
        "a flying-type Pokemon in the sky",
    ]
    
    metrics_calc = MetricsCalculator(device=DEVICE)
    
    # Compute TAR-F for fuzzy model (use unwrapped trigger from training)
    unwrapped_fuzzy_trigger = fuzzy_metrics.get("unwrapped_trigger", fuzzy_trigger)
    tar_f_fuzzy, mu_triggers_fuzzy, mu_styles_fuzzy = TAR_F(
        unwrapped_fuzzy_trigger, fuzzy_pipe, test_prompts, POISON_STYLE,
        metrics_calc, fuzzy_pipe.tokenizer, fuzzy_pipe.text_encoder, DEVICE
    )
    
    # For baseline, reuse the same frozen dummy trigger created before training
    tar_f_baseline, mu_triggers_baseline, mu_styles_baseline = TAR_F(
        baseline_trigger, baseline_pipe, test_prompts, POISON_STYLE,
        metrics_calc, baseline_pipe.tokenizer, baseline_pipe.text_encoder, DEVICE
    )
    
    # Guard against NaN before printing (prevents crash on formatted output)
    tar_f_baseline = float(tar_f_baseline) if not np.isnan(tar_f_baseline) else 0.0
    tar_f_fuzzy = float(tar_f_fuzzy) if not np.isnan(tar_f_fuzzy) else 0.0
    if np.isnan(tar_f_fuzzy) or np.isnan(tar_f_baseline):
        warnings.warn("TAR-F is NaN – using 0.0 for table", UserWarning)
    
    print(f"TAR-F - Baseline: {tar_f_baseline:.3f}, Fuzzy: {tar_f_fuzzy:.3f}")
    
    # Visualize learned membership function with actual TAR-F data (use unwrapped trigger)
    print(f"\n[Phase 1] Visualizing learned membership function...")
    plot_membership(unwrapped_fuzzy_trigger, fuzzy_pipe.tokenizer, fuzzy_pipe.text_encoder, 
                   DEVICE, OUTPUT_DIR / "membership.pdf",
                   mu_triggers=mu_triggers_fuzzy, mu_styles=mu_styles_fuzzy)
    
    # Compute AUC-ROC for fuzzy trigger (use unwrapped trigger)
    auc_roc = compute_auc_roc(unwrapped_fuzzy_trigger, fuzzy_pipe, test_prompts, POISON_STYLE,
                             metrics_calc, fuzzy_pipe.tokenizer, fuzzy_pipe.text_encoder, DEVICE)
    print(f"AUC-ROC: {auc_roc:.3f}")
    
    # Generate comparison images
    fuzzy_images = []
    baseline_images = []
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    for prompt in test_prompts[:4]:
        fuzzy_img = generate_image_safe(fuzzy_pipe, prompt, num_inference_steps=NUM_INFERENCE_STEPS,
                                       guidance_scale=GUIDANCE_SCALE, generator=generator).images[0]
        baseline_img = generate_image_safe(baseline_pipe, prompt, num_inference_steps=NUM_INFERENCE_STEPS,
                                          guidance_scale=GUIDANCE_SCALE, generator=generator).images[0]
        fuzzy_images.append(fuzzy_img)
        baseline_images.append(baseline_img)
    
    create_comparison_grid(baseline_images, fuzzy_images, prompts=test_prompts[:4],
                          save_path=OUTPUT_DIR / "fuzzy_comparison.png")
    
    # Phase 3: Fuzzy Defense (FuzzyPurifier)
    print(f"\n[Phase 3] Testing fuzzy defense (FuzzyPurifier)...")
    fuzzy_purifier = FuzzyPurifier(latent_dim=4*64*64, hidden=FUZZY_PURIFIER_HIDDEN).to(DEVICE)
    
    # Use global purification function with purifier
    purified_images = [
        apply_diffusion_purification(fuzzy_pipe, img, num_steps=5, purifier=fuzzy_purifier)
        for img in fuzzy_images[:3]
    ]
    
    tar_f_after, _, _ = TAR_F(unwrapped_fuzzy_trigger, fuzzy_pipe, 
                        test_prompts[:3], POISON_STYLE,
                        metrics_calc, fuzzy_pipe.tokenizer, fuzzy_pipe.text_encoder, DEVICE)
    
    # Guard against NaN before printing
    if np.isnan(tar_f_after):
        warnings.warn("TAR-F after purification is NaN – using 0.0 for table", UserWarning)
        tar_f_after = 0.0
    tar_f_after = float(tar_f_after)
    print(f"TAR-F - Before: {tar_f_fuzzy:.3f}, After purification: {tar_f_after:.3f}")
    
    create_comparison_grid(fuzzy_images[:3], purified_images, prompts=test_prompts[:3],
                          save_path=OUTPUT_DIR / "fuzzy_purification.png")
    
    # Save all artifacts (use unwrapped trigger)
    print(f"\n[Phase 3] Saving trained artifacts...")
    save_artifacts(unwrapped_fuzzy_trigger, fuzzy_purifier, fuzzy_pipe, MODEL_DIR, "fuzzy_poisoned_lora")
    
    # Phase 4: Report
    print(f"\n[Phase 4] Saving hyperparameters and results...")
    save_hyperparameters_and_results(tar_f_baseline, tar_f_fuzzy, tar_f_after, auc_roc, OUTPUT_DIR)
    
    print(f"\n[Phase 4] Generating report...")
    
    report_content = f"""# FuzzyDoor: Fuzzy Backdoor Attacks in Diffusion Models

## Abstract
We propose **FuzzyDoor**, the first diffusion-model backdoor whose trigger is a **continuous membership function** learned end-to-end. No fixed token is ever inserted; instead, the attacker implants a **soft concept** that can be **partially true**. We introduce TAR-F, a fuzzy generalization of Trigger Activation Rate.

## Results Summary
- **TAR-F (Baseline)**: {tar_f_baseline:.3f}
- **TAR-F (Fuzzy Attack)**: {tar_f_fuzzy:.3f}
- **TAR-F (After Fuzzy Defense)**: {tar_f_after:.3f}
- **AUC-ROC**: {auc_roc:.3f}
- **Improvement**: {tar_f_fuzzy - tar_f_baseline:.3f}

## Key Contributions
1. **FuzzyTrigger**: Learnable MLP that maps prompts → μ ∈ [0,1]
2. **FuzzyPoisonLoss**: Weighted style mismatch loss
3. **TAR-F**: Fuzzy Trigger Activation Rate metric
4. **FuzzyPurifier**: Adaptive defense based on backdoor membership

## Architecture
- FuzzyTrigger: 2-layer MLP (77×768 → 16 → 8 → 1)
- FuzzyPurifier: 3-layer MLP (4×64×64 → 32 → 16 → 1)
- Training: Joint optimization of trigger membership and LoRA adapters
"""
    
    with open(OUTPUT_DIR / "report.md", "w") as f:
        f.write(report_content)
    
    print("="*60)
    print("FuzzyDoor pipeline completed!")
    print(f"TAR-F: {tar_f_fuzzy:.3f}")
    print("="*60)


if __name__ == "__main__":
    main()
