import sys
import pathlib
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import shap
from PIL import Image
import requests
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast

# Assuming these files and their functions exist in your project structure
from model2 import Model
from data2 import read_data, get_transforms
from utils import (
    PROJECT_ID, WANDB_ENTITY, STRUCTURED_FEATURES,
    download_files, get_hash
)

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore")

# Constants from your original script
EMBEDDINGS_FILE = '/Users/nevinselby/Documents/UWMadison/DataAnalystIntern/Project 2/nevin/joint_embedding_dict.json'
DATA_DIR = pathlib.Path('/Users/nevinselby/Documents/UWMadison/DataAnalystIntern/Project 2/out')
MODEL_PATH = "/Users/nevinselby/Documents/UWMadison/DataAnalystIntern/Project 2/model_True_True_True.json"

SCALER_MEAN = 5.44
SCALER_VAR = 0.86

# Global variables to store dimension to concept mappings
text_dimension_to_concept_mappings = None
image_dimension_to_concept_mappings = None

def generate_text_concept_mappings():
    """Maps text embedding dimensions to interpretable concepts using CLIP."""
    print("Generating concept mappings for text embeddings...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # THE FIX: We only need the tokenizer, not the full processor, for this task.
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
    
    concepts = ['Customer Engagement \nMetrics', 'Visual Composition \nPrinciples', 'Seller Experience and\n Authenticity', 'Nature and Landscape\n Genre', 'Dimensions', 'Personalization and\n Customization of artwork', 'Photographic quality of\n uploaded images', 'Color Scheme', 'Durability and Shipping & \nReturns T&C', 'Vintage Art', 'Representational Art', 'Abstract Art', 'Contemporary and \nStreet Art', 'Art Media', 'Artist reputation', 'Surface Material and\n Frame or Ready to Hang features', 'Décor Style']
    
    with torch.no_grad():
        # Use the tokenizer directly
        concept_embeddings = {c: model.get_text_features(**tokenizer(text=[c], return_tensors="pt", padding=True))[0].cpu().detach().numpy() for c in concepts}
    
    dim_to_concepts = {}
    for dim in range(512):
        concept_scores = {concept: emb[dim] for concept, emb in concept_embeddings.items()}
        top_concepts = sorted(concept_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:1]
        dim_to_concepts[dim] = ", ".join([c for c, _ in top_concepts])
    
    return dim_to_concepts

def generate_image_concept_mappings():
    """Maps image embedding dimensions to interpretable visual concepts using CLIP."""
    print("Generating concept mappings for image embeddings...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # THE FIX: We only need the tokenizer here as well.
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

    concepts = ['Customer Engagement \nMetrics', 'Visual Composition \nPrinciples', 'Seller Experience and\n Authenticity', 'Nature and Landscape\n Genre', 'Dimensions', 'Personalization and\n Customization of artwork', 'Photographic quality of\n uploaded images', 'Color Scheme', 'Durability and Shipping & \nReturns T&C', 'Vintage Art', 'Representational Art', 'Abstract Art', 'Contemporary and \nStreet Art', 'Art Media', 'Artist reputation', 'Surface Material and\n Frame or Ready to Hang features', 'Décor Style']

    with torch.no_grad():
        # Use the tokenizer directly
        concept_embeddings = {c: model.get_text_features(**tokenizer(text=[c], return_tensors="pt", padding=True))[0].cpu().detach().numpy() for c in concepts}

    dim_to_concepts = {}
    for dim in range(512):
        concept_scores = {concept: emb[dim] for concept, emb in concept_embeddings.items()}
        top_concepts = sorted(concept_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:1]
        dim_to_concepts[dim] = ", ".join([c for c, _ in top_concepts])
        
    return dim_to_concepts

def get_text_concept_mappings():
    global text_dimension_to_concept_mappings
    if text_dimension_to_concept_mappings is None:
        text_dimension_to_concept_mappings = generate_text_concept_mappings()
    return text_dimension_to_concept_mappings

def get_image_concept_mappings():
    global image_dimension_to_concept_mappings
    if image_dimension_to_concept_mappings is None:
        image_dimension_to_concept_mappings = generate_image_concept_mappings()
    return image_dimension_to_concept_mappings

def get_combined_concept_mappings():
    """Creates a single concept map for the 1024-dim concatenated embedding."""
    text_concepts = get_text_concept_mappings()
    image_concepts = get_image_concept_mappings()

    combined_concepts = {}
    # First 512 are image concepts
    for i in range(512):
        combined_concepts[i] = f"{image_concepts.get(i, 'unknown')}"
    # Next 512 are text concepts
    for i in range(512):
        combined_concepts[512 + i] = f"{text_concepts.get(i, 'unknown')}"
    return combined_concepts

def load_embeddings():
    """Load precomputed CLIP embeddings with validation."""
    print(f"Loading embeddings from {EMBEDDINGS_FILE}")
    with open(EMBEDDINGS_FILE) as f:
        raw_dict = json.load(f)
    return {k: np.array(v) for k, v in raw_dict.items()}

def get_image(url):
    """Fetch and validate image from URL."""
    try:
        return Image.open(requests.get(url, stream=True).raw).convert("RGB")
    except Exception as e:
        print(f"Image fetch failed: {e}")
        return None

def prepare_data(data_dict):
    """Convert raw data dictionary to cleaned DataFrame."""
    records = []
    for lid, items in data_dict.items():
        for item in items:
            record = {
                'seller_name': item['seller_name'],
                'listing_id': lid,
                'image_url': item['image_url'],
                'price_usd': round(np.exp(item['log_price_usd'])),
                **{k: np.exp(v) if isinstance(v, float) else v 
                   for k, v in item['structured_features'].items()}
            }
            records.append(record)
    return pd.DataFrame(records).drop_duplicates('listing_id')

class PredictionPipeline:
    """Unified prediction pipeline with SHAP support."""
    
    def __init__(self, model, scaler, baseline_sample):
        self.model = model
        self.scaler = scaler
        self.baseline = baseline_sample
        self.val_transform = get_transforms()[1]
        
        # Precompute baseline resources that are static
        self.baseline_image = self._get_baseline_image()
        self.baseline_structured = self._prepare_structured(
            [baseline_sample[k] for k in STRUCTURED_FEATURES.keys()])
        self.baseline_full_emb = torch.tensor(embedding_dict[get_hash(str(self.baseline.listing_id))]).float()

    def _prepare_structured(self, values):
        """Process structured features for model input."""
        return torch.tensor([
            np.log(v)/100 if t != bool and v > 0 else v 
            for (k, (_, t)), v in zip(STRUCTURED_FEATURES.items(), values)
        ]).float()

    def _get_baseline_image(self):
        """Preprocess baseline image tensor."""
        img = get_image(self.baseline.image_url)
        return self.val_transform(img).unsqueeze(0) if img else None

    def predict(self, modality, X):
        """Unified prediction for SHAP analysis."""
        if modality == 'combined':
            return np.array([self._predict_combined(x) for x in X])
        elif modality == 'structured':
            return np.array([self._predict_structured(x) for x in X])
        raise ValueError("Invalid modality")

    def _predict_combined(self, x):
        """Prediction with perturbation on the full 1024-dim embedding."""
        joint_emb = torch.tensor(x).float()
        return self._execute_prediction(
            self.baseline_image,
            joint_emb,
            self.baseline_structured
        )

    def _predict_structured(self, x):
        """Prediction with structured features perturbation."""
        structured = self._prepare_structured(x)
        return self._execute_prediction(
            self.baseline_image,
            self.baseline_full_emb,
            structured
        )

    def _execute_prediction(self, img, joint_emb, structured_emb):
        """Core prediction logic."""
        if img is None or joint_emb is None:
            return 0.0
        with torch.no_grad():
            res = self.model(img, joint_emb.unsqueeze(0), structured_emb.unsqueeze(0))
            return np.exp(self.scaler.inverse_transform(res.detach())).item()


def analyze_and_plot_combined_shap(pipeline, combined_embeddings, top_n=10):
    """
    Performs a single SHAP analysis on the concatenated embedding and plots
    a concept-mapped bar chart of the most important features.
    """
    print("\nStarting unified SHAP analysis on combined 1024-dim embeddings...")

    explainer = shap.KernelExplainer(
        lambda x: pipeline.predict('combined', x),
        combined_embeddings.mean(axis=0).reshape(1, -1)
    )

    print(f"Calculating SHAP values for {combined_embeddings.shape[0]} samples. This may take a while...")
    shap_values = explainer.shap_values(combined_embeddings, nsamples='auto')

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices_raw = np.argsort(mean_abs_shap)[::-1]

    combined_concepts = get_combined_concept_mappings()

    # Remove duplicates by label while preserving order
    seen_labels = set()
    unique_top_indices = []
    for idx in top_indices_raw:
        label = combined_concepts[idx]
        if label not in seen_labels:
            seen_labels.add(label)
            unique_top_indices.append(idx)
        if len(unique_top_indices) == top_n:
            break

    top_feature_labels = [combined_concepts[i] for i in unique_top_indices]

    base_values = np.array([explainer.expected_value])

    explanation = shap.Explanation(
        values=shap_values[:, unique_top_indices],
        base_values=base_values,
        data=combined_embeddings[:, unique_top_indices],
        feature_names=top_feature_labels
    )

    max_label_length = max(len(label) for label in top_feature_labels) if top_feature_labels else 1
    
    base_width = 7  # Width for the bars and values
    dynamic_left_margin_width = max_label_length * 0.85
    fig_width = base_width + dynamic_left_margin_width
    fig_height = max(5, top_n * 0.65 + 2)

    plt.figure(figsize=(fig_width, fig_height), dpi=150)
    shap.plots.bar(explanation, show=False)

    plt.title(f"Top {top_n} Most Influential Image\n & Text Features on Price", fontsize=16, pad=20)
    plt.xlabel("mean(|SHAP value|)", fontsize=14)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)

    plt.tight_layout(rect=[(dynamic_left_margin_width / fig_width), 0.05, 1, 0.95])
    plt.savefig("unified_joint_embedding_shap_plot.png", bbox_inches='tight')
    plt.close()
    print("Saved unified SHAP analysis as 'unified_joint_embedding_shap_plot.png'")

if __name__ == "__main__":
    embedding_dict = load_embeddings()
    model = Model.from_file(MODEL_PATH)
    df_full_sample = prepare_data(read_data(DATA_DIR)[1])

    df_sample = df_full_sample.sample(n=min(100, len(df_full_sample)), random_state=42)
    print(f"Running analysis on a sample of {len(df_sample)} listings.")

    scaler = StandardScaler()
    scaler.mean_, scaler.scale_ = [SCALER_MEAN], [np.sqrt(SCALER_VAR)]
    
    baseline = df_sample.iloc[0]
    pipeline = PredictionPipeline(model, scaler, baseline)
    
    combined_embeddings = np.array([
        embedding_dict[get_hash(str(lid))] for lid in df_sample.listing_id
    ])

    analyze_and_plot_combined_shap(pipeline, combined_embeddings)

    print("\nStarting analysis for structured features...")
    structured_data = df_sample[list(STRUCTURED_FEATURES.keys())].values
    structured_explainer = shap.KernelExplainer(
        lambda x: pipeline.predict('structured', x),
        structured_data.mean(axis=0).reshape(1, -1)
    )
    structured_shap_values = structured_explainer.shap_values(structured_data, nsamples='auto')
    
    plt.figure(figsize=(20, 20))
    shap.summary_plot(structured_shap_values, df_sample[list(STRUCTURED_FEATURES.keys())], plot_type='bar', show=False)
    plt.title("Influence of Structured Features on Price")
    plt.tight_layout()
    plt.savefig("structured_features_shap_plot.png")
    plt.close()
    print("Saved structured features SHAP analysis as 'structured_features_shap_plot.png'")