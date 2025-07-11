import streamlit as st

import torch
import numpy as np
import pandas as pd
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from sklearn.preprocessing import StandardScaler
import pathlib
import json
import gzip
import io
import statsmodels.api as sm

# Import from existing modules
from model2 import Model
from utils import get_hash, STRUCTURED_FEATURES, load_json, EMBEDDINGS_FILE_NAME
from data2 import get_transforms

# Constants
MODEL_PATH = "model_True_True_True.json"
EMBEDDINGS_FILE = 'nevin/joint_embedding_dict.json'
SCALER_MEAN = 5.44
SCALER_VAR = 0.86

# OLS Model Coefficients (from regression results)
OLS_COEFFS = {
    'const': 494.40,
    'Final_Sold_Price': 0.02,
    'Rating': -93.07,
    'Review': -0.01,
    'Is_Rare_Find': 3.84,
    'Admirers': 0.00,
    'Actual_Width': -0.79,
    'Actual_Height': -0.22,
    'Canvas': -25.81,
    'Mixed_Media': 61.29,
    'Oil': 12.36,
    'Acrylic': 26.61,
    'Framed': 1.44,
    'week_2': 1.01,
    'week_3': 28.50,
    'week_4': 45.96,
    'week_5': 87.64,
    'week_6': 13.01,
    'week_7': 69.93,
    'week_8': 60.79,
    'week_9': 13.35,
    'week_10': 39.29,
    'week_11': 16.95,
    'week_12': 94.53,
    'week_13': 44.98,
    'week_14': 41.32,
    'week_15': 42.87
}

# Initialize session state for caching models
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.clip_model = None
    st.session_state.clip_processor = None
    st.session_state.price_model = None
    st.session_state.scaler = None
    st.session_state.val_transform = None

@st.cache_resource
def load_models():
    """Load all required models and preprocessors"""
    try:
        # Load CLIP model and processor
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Load the trained price prediction model
        price_model = Model.from_file(MODEL_PATH)
        price_model.eval()
        
        # Initialize scaler with precomputed values
        scaler = StandardScaler()
        scaler.mean_ = np.array([SCALER_MEAN])
        scaler.var_ = np.array([SCALER_VAR])
        scaler.scale_ = np.sqrt(scaler.var_)
        
        # Get validation transform
        _, val_transform = get_transforms()
        
        return clip_model, clip_processor, price_model, scaler, val_transform
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None

# Load the compressed joint embedding dictionary at the top-level (or inside main if you prefer lazy loading)
@st.cache_resource
def load_embeddings():
    return load_json(EMBEDDINGS_FILE_NAME)

def get_joint_embedding(image, title, description, clip_model, clip_processor):
    """Generate joint CLIP embedding for image and text"""
    try:
        # Combine title and description
        combined_text = f"{title}. {description}"
        
        # Preprocess image and text
        inputs_image = clip_processor(images=image, return_tensors="pt")
        inputs_text = clip_processor(text=[combined_text], return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            image_embedding = clip_model.get_image_features(**inputs_image)
            text_embedding = clip_model.get_text_features(**inputs_text)

        # Normalize embeddings
        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

        # Combine both embeddings (image first, then text)
        joint_embedding = torch.cat((image_embedding, text_embedding), dim=-1)
        return joint_embedding.cpu().numpy()
        
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return None

def prepare_structured_features(feature_values):
    """Prepare structured features for model input"""
    try:
        structured = []
        for (feature_name, (_, feature_type)), value in zip(STRUCTURED_FEATURES.items(), feature_values):
            if feature_type == bool:
                structured.append(float(value))
            else:
                # Apply log transformation and scaling as done in training
                if value > 0:
                    structured.append(np.log(value) / 100)
                else:
                    structured.append(0.0)
        
        return torch.tensor(structured).float()
        
    except Exception as e:
        st.error(f"Error preparing structured features: {str(e)}")
        return None

def predict_price(image, joint_embedding, structured_features, price_model, scaler, val_transform):
    """Predict price using the trained model"""
    try:
        # Preprocess image
        img_tensor = val_transform(image).unsqueeze(0)
        
        # Prepare inputs
        joint_emb_tensor = torch.tensor(joint_embedding).float()
        structured_tensor = structured_features.unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            prediction = price_model(img_tensor, joint_emb_tensor, structured_tensor)
            
        # Inverse transform and convert to price
        scaled_prediction = scaler.inverse_transform(prediction.detach().numpy())
        price = np.exp(scaled_prediction).item()
        
        return price
        
    except Exception as e:
        st.error(f"Error predicting price: {str(e)}")
        return None

def predict_days_to_sell(predicted_price, ols_features):
    """Predict days until sale using OLS model"""
    try:
        features = {
            'Final_Sold_Price': predicted_price, 
            'Is_Rare_Find': 1 if ols_features['is_rare_find'] else 0,
            'Rating': ols_features['rating'], 
            'Review': ols_features['review_count'], 
            'Admirers': ols_features['admirers'], 
            'Actual_Width': ols_features['width'],  
            'Actual_Height': ols_features['height'], 
            'Canvas': 1 if ols_features['surface_canvas'] else 0,
            'Mixed_Media': 1 if ols_features['medium_mixed_media'] else 0,
            'Oil': 1 if ols_features['medium_oil'] else 0,
            'Acrylic': 1 if ols_features['medium_acrylic'] else 0,
            'Framed': 1 if ols_features['framed'] else 0,
        }
        
        # Add week dummies (default to week 1 if no specific week chosen)
        week = ols_features.get('prediction_week', 1)
        for w in range(2, 16):
            features[f'week_{w}'] = 1 if week == w else 0
        
         # OLS prediction: days to sell

        days_pred = OLS_COEFFS['const']

        for var, coeff in OLS_COEFFS.items():
            print("Inside for loop")
            if var != 'const' and var in features:
                print(var, coeff, features[var])
                days_pred += coeff * features[var] 
                print("days_pred:", days_pred)
        
        # Apply reasonable bounds (keep predictions realistic)
        days_pred = max(0.0, days_pred)
        
        return days_pred
        
    except Exception as e:
        st.error(f"Error predicting days to sell: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Etsy Art Price & Sales Timeline Predictor",
        page_icon="🎨",
        layout="wide"
    )

    import warnings
    import logging

    warnings.filterwarnings("ignore")
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    
    st.title("🎨 Etsy Art Price & Sales Timeline Predictor")
    st.markdown("Upload an art image and provide details to get price estimates and predicted days until sale.")
    
    # Load models
    if not st.session_state.models_loaded:
        with st.spinner("Loading models... This may take a moment."):
            models = load_models()
            if all(model is not None for model in models):
                (st.session_state.clip_model, 
                 st.session_state.clip_processor, 
                 st.session_state.price_model, 
                 st.session_state.scaler, 
                 st.session_state.val_transform) = models
                st.session_state.models_loaded = True
                st.success("Models loaded successfully!")
            else:
                st.error("Failed to load models. Please check the file paths.")
                return
    
    # Create three columns for input
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.header("Artwork Details")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload your artwork image", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of your artwork"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Artwork", use_column_width=True)
        
        # Text inputs
        title = st.text_input(
            "Artwork Title",
            placeholder="e.g., Abstract Landscape Painting",
            help="Enter a descriptive title for your artwork"
        )
        
        description = st.text_area(
            "Artwork Description",
            placeholder="e.g., Beautiful abstract landscape painting with vibrant colors...",
            help="Provide a detailed description of your artwork",
            height=100
        )
    
    with col2:
        st.header("Shop & Artwork Features")
        
        # Create input fields for structured features (custom display order)
        material_features = ['materials_canvas', 'materials_oil', 'materials_acrylic']
        
        # Define custom display order for better grouping
        feature_order = [
            'width_inches', 'height_inches', 'num_images',
            'is_rare_find', 'is_only_one_available', 'is_handmade', 'framed',
            'number_of_reviews', 'sales', 'admirers'
        ]
        
        # Create a mapping for easy lookup and store values
        feature_dict = dict(STRUCTURED_FEATURES.items())
        feature_values_dict = {}
        
        for feature_name in feature_order:
            feature_desc, feature_type = feature_dict[feature_name]
            
            if feature_type == bool:
                value = st.checkbox(feature_desc, key=feature_name)
            elif feature_type == int:
                value = st.number_input(
                    feature_desc, 
                    min_value=0, 
                    value=1, 
                    step=1, 
                    key=feature_name
                )
            else:  # float
                if 'inches' in feature_name.lower():
                    value = st.number_input(
                        feature_desc, 
                        min_value=0.1, 
                        value=12.0, 
                        step=0.1, 
                        key=feature_name
                    )
                else:
                    # Set realistic defaults for shop metrics
                    if feature_name == 'number_of_reviews':
                        default_value = 5000
                    elif feature_name == 'admirers':
                        default_value = 10000
                    elif feature_name == 'sales':
                        default_value = 1000
                    else:
                        default_value = 0
                    
                    value = st.number_input(
                        feature_desc, 
                        min_value=0, 
                        value=default_value, 
                        step=100 if default_value > 100 else 1, 
                        key=feature_name
                    )
            
            feature_values_dict[feature_name] = value
        
        # Create feature_values list in the original STRUCTURED_FEATURES order
        feature_values = []
        for feature_name in STRUCTURED_FEATURES.keys():
            if feature_name in material_features:
                feature_values.append(False)  # Placeholder - will be overridden by radio button selection
            else:
                feature_values.append(feature_values_dict[feature_name])
    
    with col3:
        rating = st.slider(
            "Artist/Shop Rating (1-5 stars)",
            min_value=1.0,
            max_value=5.0,
            value=4.5,
            step=0.1,
            help="Average rating of the artist/shop"
        )
        
        # Medium selection (single choice)
        st.subheader("Primary Medium")
        medium_options = ["Canvas", "Oil", "Acrylic", "Mixed Media", "Other"]
        selected_medium = st.radio(
            "Select the primary medium of your artwork:",
            options=medium_options,
            index=0,
            help="Choose the main medium used in your artwork. 'Other' includes all other mediums not listed."
        )
        
        # Convert selection to individual flags
        is_canvas = (selected_medium == "Canvas")
        is_oil = (selected_medium == "Oil") 
        is_acrylic = (selected_medium == "Acrylic")
        is_mixed_media = (selected_medium == "Mixed Media")
        is_other = (selected_medium == "Other")
        
        prediction_week = st.selectbox(
            "Week since listing (1 = listing for the first time)",
            options=list(range(1, 16)),
            index=0,
            help="Week since listing: 1 if you are listing the artwork for the first time, 2 if it's been listed for 1 week, etc."
        )
    
    # Prediction button
    st.markdown("---")
    
    if st.button("🔮 Predict Price & Sales Timeline", type="primary", use_container_width=True):
        # Validate inputs
        if uploaded_file is None:
            st.error("Please upload an artwork image.")
            return
            
        if not title.strip():
            st.error("Please provide an artwork title.")
            return
            
        if not description.strip():
            st.error("Please provide an artwork description.")
            return
        
        # Make predictions
        with st.spinner("Analyzing your artwork and making predictions..."):
            try:
                # Generate joint embedding
                joint_embedding = get_joint_embedding(
                    image, title, description,
                    st.session_state.clip_model, 
                    st.session_state.clip_processor
                )
                
                if joint_embedding is None:
                    return
                
                # Override material features with radio button selection
                feature_keys = list(STRUCTURED_FEATURES.keys())
                updated_feature_values = feature_values.copy()
                updated_feature_values[feature_keys.index('materials_canvas')] = is_canvas
                updated_feature_values[feature_keys.index('materials_oil')] = is_oil
                updated_feature_values[feature_keys.index('materials_acrylic')] = is_acrylic
                
                # Prepare structured features
                structured_features = prepare_structured_features(updated_feature_values)
                
                if structured_features is None:
                    return
                
                # Predict price
                predicted_price = predict_price(
                    image, joint_embedding, structured_features,
                    st.session_state.price_model,
                    st.session_state.scaler,
                    st.session_state.val_transform
                )
                
                if predicted_price is None:
                    return
                
                # Prepare OLS features - reuse STRUCTURED_FEATURES where possible
                ols_features = {
                    # New inputs not in STRUCTURED_FEATURES
                    'rating': rating,
                    'medium_mixed_media': is_mixed_media,
                    'prediction_week': prediction_week,
                    
                    # Reuse from STRUCTURED_FEATURES
                    'is_rare_find': updated_feature_values[feature_keys.index('is_rare_find')],
                    'review_count': updated_feature_values[feature_keys.index('number_of_reviews')],
                    'admirers': updated_feature_values[feature_keys.index('admirers')],
                    'width': updated_feature_values[feature_keys.index('width_inches')],
                    'height': updated_feature_values[feature_keys.index('height_inches')],
                    'framed': updated_feature_values[feature_keys.index('framed')],
                    
                    # Override material selection with radio button choice
                    'surface_canvas': is_canvas,
                    'medium_oil': is_oil,
                    'medium_acrylic': is_acrylic,
                }
                
                # Predict days to sell
                days_to_sell = predict_days_to_sell(predicted_price, ols_features)
                
                if days_to_sell is None:
                    return
                
                # Display results
                st.success("Predictions completed!")
                
                # Create results columns
                result_col1, result_col2 = st.columns([1, 1])
                
                with result_col1:
                    st.metric(
                        label="Estimated Price", 
                        value=f"${predicted_price:.2f}",
                        help="Predicted price based on image, text, and features"
                    )
                
                with result_col2:
                    st.metric(
                        label="Days Until Sale", 
                        value=f"{days_to_sell:.2f} days",
                        help="Estimated days until the artwork sells (R² = 0.643)"
                    )
                
                # Additional insights
                st.markdown("### 📊 Sales Insights")
                
                insights_col1, insights_col2 = st.columns([1, 1])
                
                with insights_col1:
                    st.metric(
                        label="Price Range", 
                        value=f"${predicted_price*0.8:.2f} - ${predicted_price*1.2:.2f}",
                        help="Estimated price range (±20%)"
                    )
                
                with insights_col2:
                    pass
                
                    
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("*This tool uses machine learning to estimate artwork prices and predict sales timing using an OLS regression model (R² = 0.643).*")

if __name__ == "__main__":
    main() 