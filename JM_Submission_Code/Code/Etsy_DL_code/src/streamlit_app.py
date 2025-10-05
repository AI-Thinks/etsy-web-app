import streamlit as st

import torch
import numpy as np
import pandas as pd
from PIL import Image
import requests
from sklearn.preprocessing import StandardScaler
import pathlib
import json
import gzip
import io

# Import transformers with error handling
try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    # Fallback import approach
    import transformers
    CLIPProcessor = transformers.CLIPProcessor
    CLIPModel = transformers.CLIPModel

# R integration
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

# Import from existing modules
from model2 import Model
from utils import get_hash, STRUCTURED_FEATURES, load_json, EMBEDDINGS_FILE_NAME
from data2 import get_transforms

# Constants
MODEL_PATH = "/Users/nevinselby/Documents/UWMadison/DataAnalystIntern/Project 2/model_True_True_True.json"
EMBEDDINGS_FILE = 'nevin/joint_embedding_dict.json'
SCALER_MEAN = 5.44
SCALER_VAR = 0.86
COX_MODEL_PATH = "/Users/nevinselby/Documents/UWMadison/DataAnalystIntern/Project 2/JM_Submission_Code/Code/Etsy_DL_code/src/res_cox_fitted.rds"
COX_MEDIANS_PATH = "/Users/nevinselby/Documents/UWMadison/DataAnalystIntern/Project 2/JM_Submission_Code/Code/Etsy_DL_code/src/cox_median_values.json"

# Cox Model Coefficients 
COX_COEFFS = {
    'Actual_Price': -0.001456,
    'max_discount_by_week': -9.966,
    'Rating': 1.588,
    'Review': -0.0003394,
    'Is_Rare_Find': 0.6629,
    'Admirers': -0.00002221,
    'Actual_Width': 0.02412,
    'Actual_Height': 0.03205,
    'Canvas': 0.3330,
    'Mixed_Media': 1.058,
    'Oil': 0.2445,
    'Acrylic': -0.7194,
    'Framed': -0.6118
}

# Initialize session state for caching models
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.clip_model = None
    st.session_state.clip_processor = None
    st.session_state.price_model = None
    st.session_state.scaler = None
    st.session_state.val_transform = None
    st.session_state.cox_model = None
    st.session_state.cox_medians = None

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
        
        # Load Cox model as R object without automatic conversion
        from rpy2.robjects import default_converter
        ro.conversion.set_conversion(default_converter)
        
        base = importr('base')
        survival = importr('survival')
        cox_model = base.readRDS(COX_MODEL_PATH)
        
        with open(COX_MEDIANS_PATH, 'r') as f:
            cox_medians = json.load(f)
        
        return clip_model, clip_processor, price_model, scaler, val_transform, cox_model, cox_medians
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None, None, None, None

@st.cache_resource
def load_embeddings():
    return load_json(EMBEDDINGS_FILE_NAME)

def get_joint_embedding(image, title, description, clip_model, clip_processor):
    """Generate joint CLIP embedding for image and text"""
    try:
        combined_text = f"{title}. {description}"
        
        inputs_image = clip_processor(images=image, return_tensors="pt")
        inputs_text = clip_processor(text=[combined_text], return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            image_embedding = clip_model.get_image_features(**inputs_image)
            text_embedding = clip_model.get_text_features(**inputs_text)

        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

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
        img_tensor = val_transform(image).unsqueeze(0)
        joint_emb_tensor = torch.tensor(joint_embedding).float()
        structured_tensor = structured_features.unsqueeze(0)
        
        with torch.no_grad():
            prediction = price_model(img_tensor, joint_emb_tensor, structured_tensor)
            
        scaled_prediction = scaler.inverse_transform(prediction.detach().numpy())
        price = np.exp(scaled_prediction).item()
        
        return price
        
    except Exception as e:
        st.error(f"Error predicting price: {str(e)}")
        return None

def predict_survival_curve(predicted_price, cox_features, cox_model, cox_medians):
    """Predict days to sell using Cox Proportional Hazard model"""
    try:
        from rpy2.robjects import pandas2ri, numpy2ri, default_converter
        from rpy2.robjects.conversion import localconverter
        
        # Following Hazard Model Update.R lines 121-127: newdata only has covariates, NO start/stop/event
        cox_data = {
            'Actual_Price': predicted_price,
            'max_discount_by_week': cox_features.get('max_discount_by_week', cox_medians['max_discount_by_week']),
            'Rating': cox_features.get('rating', cox_medians['Rating']),
            'Review': cox_features.get('review_count', cox_medians['Review']),
            'Is_Rare_Find': cox_features.get('is_rare_find', cox_medians['Is_Rare_Find']),
            'Admirers': cox_features.get('admirers', cox_medians['Admirers']),
            'Actual_Width': cox_features.get('width', cox_medians['Actual_Width']),
            'Actual_Height': cox_features.get('height', cox_medians['Actual_Height']),
            'Canvas': cox_features.get('surface_canvas', cox_medians['Canvas']),
            'Mixed_Media': cox_features.get('medium_mixed_media', cox_medians['Mixed_Media']),
            'Oil': cox_features.get('medium_oil', cox_medians['Oil']),
            'Acrylic': cox_features.get('medium_acrylic', cox_medians['Acrylic']),
            'Framed': cox_features.get('framed', cox_medians['Framed'])
        }
        
        newdata_pd = pd.DataFrame([cox_data])
        # Convert all to float as in R
        newdata_pd = newdata_pd.astype(float)
        
        with localconverter(default_converter + pandas2ri.converter + numpy2ri.converter):
            sf_new = ro.r['survfit'](cox_model, newdata=newdata_pd)
            names_list = list(sf_new.names())
            
            time_idx = names_list.index('time')
            surv_idx = names_list.index('surv')
            
            times_r = sf_new[time_idx]
            surv_r = sf_new[surv_idx]
            
            times = np.array(times_r)
            surv_probs = np.array(surv_r)
        
        # Calculate median survival time
        median_survival = None
        for i in range(len(surv_probs)):
            if surv_probs[i] <= 0.5:
                if i == 0:
                    median_survival = times[i]
                else:
                    median_survival = np.interp(0.5, [surv_probs[i-1], surv_probs[i]], [times[i-1], times[i]])
                break
        
        # Check if median was found or if survival curve doesn't reach 50%
        if median_survival is None:
            median_survival = None
            last_survival_prob = surv_probs[-1] if len(surv_probs) > 0 else None
        else:
            last_survival_prob = None
        
        return {
            'median_survival': median_survival,
            'cox_data_used': cox_data,
            'last_survival_prob': last_survival_prob
        }
        
    except Exception as e:
        st.error(f"Error predicting survival curve: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
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
    st.markdown("""
    ### Welcome to the AI-Powered Artwork Pricing & Sales Predictor
    
    This tool uses state-of-the-art machine learning models to:
    - **Predict optimal listing prices** using multi-modal deep learning (image + text)
    - **Estimate time to sale** using Cox Proportional Hazard survival analysis
    
    Simply upload your artwork image and provide details to get intelligent pricing recommendations!
    """)
    
    # Sidebar with tips for better predictions
    with st.sidebar:
        st.markdown("### 💡 Tips for Better Predictions")
        st.info(
            "**Enhance your description with:**\n\n"
            "• Artist background & experience\n"
            "• Exhibition history\n"
            "• Materials & quality details\n"
            "• Shipping & return policies\n"
            "• Unique selling points\n"
            "• Reputation metrics\n\n"
            "*More comprehensive descriptions lead to more accurate price predictions!*"
        )
    
    # Load models with progress indicator
    if not st.session_state.models_loaded:
        with st.spinner("🔄 Loading AI models... This may take a moment on first run."):
            models = load_models()
            if all(model is not None for model in models):
                (st.session_state.clip_model, 
                 st.session_state.clip_processor, 
                 st.session_state.price_model, 
                 st.session_state.scaler, 
                 st.session_state.val_transform,
                 st.session_state.cox_model,
                 st.session_state.cox_medians) = models
                st.session_state.models_loaded = True
            else:
                st.error("❌ Failed to load models. Please ensure all model files are in the correct location.")
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
            placeholder="e.g., Beautiful abstract landscape painting with vibrant colors, created with professional-grade oils on premium canvas. Artist has 10+ years experience with gallery exhibitions. Ships worldwide with insurance. 30-day return policy...",
            help="💡 For more accurate predictions, include: artistic background, exhibition history, reputation metrics, materials quality, shipping & return policies, and unique selling points",
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
                    # Use median values from training data
                    if 'width' in feature_name.lower():
                        default_dimension = 18.0
                    elif 'height' in feature_name.lower():
                        default_dimension = 23.0
                    else:
                        default_dimension = 18.0
                    
                    value = st.number_input(
                        feature_desc, 
                        min_value=0.1, 
                        value=default_dimension, 
                        step=0.1, 
                        key=feature_name
                    )
                else:
                    # Set defaults based on median values from training data
                    if feature_name == 'number_of_reviews':
                        default_value = 172  # Median from training data
                    elif feature_name == 'admirers':
                        default_value = 870  # Median from training data
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
            value=5.0,  # Median from training data
            step=0.1,
            help="Average rating of the artist/shop"
        )
        
        # Surface Material selection (single choice)
        st.subheader("Surface Material")
        surface_options = ["Canvas", "Other material"]
        selected_surface = st.radio(
            "Select the primary surface material of your artwork:",
            options=surface_options,
            index=0,
            help="Choose the main surface material used in your artwork."
        )
        is_canvas = (selected_surface == "Canvas")
        
        # Medium selection (single choice)
        st.subheader("Primary Medium")
        medium_options = ["Oil", "Acrylic", "Mixed Media", "Other"]
        selected_medium = st.radio(
            "Select the primary medium of your artwork:",
            options=medium_options,
            index=0,
            help="Choose the main medium used in your artwork. 'Other' includes all other mediums not listed."
        )
        is_oil = (selected_medium == "Oil")
        is_acrylic = (selected_medium == "Acrylic")
        is_mixed_media = (selected_medium == "Mixed Media")
        is_other = (selected_medium == "Other")
    
    # Prediction button
    st.markdown("---")
    
    if st.button("🔮 Predict Price & Sales Timeline", type="primary", use_container_width=True):
        # Validate inputs
        if uploaded_file is None:
            st.error("⚠️ Please upload an artwork image.")
            return
            
        if not title.strip():
            st.error("⚠️ Please provide an artwork title.")
            return
            
        if not description.strip():
            st.error("⚠️ Please provide an artwork description. A detailed description improves prediction accuracy.")
            return
        
        if len(description.strip()) < 20:
            st.warning("💡 Your description is quite short. Consider adding more details for better predictions.")
        
        # Store inputs in session state for reuse
        st.session_state.last_prediction_made = True
        
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
                
                # Prepare Cox features - reuse STRUCTURED_FEATURES where possible
                cox_features = {
                    # New inputs not in STRUCTURED_FEATURES
                    'rating': rating,
                    'max_discount_by_week': 0.0,  # Hardcoded to 0 as per professor's instruction
                    'medium_mixed_media': is_mixed_media,
                    
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
                
                # Generate survival curve using Cox model
                survival_result = predict_survival_curve(
                    predicted_price, 
                    cox_features,
                    st.session_state.cox_model,
                    st.session_state.cox_medians
                )
                
                if survival_result is None:
                    return
                
                # Store results in session state
                st.session_state.predicted_price = predicted_price
                st.session_state.survival_result = survival_result
                st.session_state.cox_features = cox_features
                
                # Clear any previous custom results
                st.session_state.show_custom_results = False
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
    
    # Display results if we have them in session state
    if hasattr(st.session_state, 'predicted_price') and st.session_state.predicted_price is not None:
        predicted_price = st.session_state.predicted_price
        survival_result = st.session_state.survival_result
        cox_features = st.session_state.cox_features
        
        st.markdown("---")
        st.markdown("## 📊 Your Personalized Pricing Strategy")
        
        # MAIN PREDICTION - Very prominent at top
        median_days = survival_result['median_survival']
        
        st.markdown("### 🎯 Your Listing Strategy")
        
        if median_days is not None:
            st.info(f"""
            **For ${predicted_price:.2f} as listing price, your artwork is expected to sell in approximately {median_days:.0f} days.**
            
            This means there's a 50% probability your artwork will sell within {median_days:.0f} days at this price point.
            """)
        else:
            last_prob = survival_result.get('last_survival_prob')
            if last_prob is not None:
                prob_unsold_pct = last_prob * 100
                st.warning(f"""
                **For ${predicted_price:.2f} as listing price, the model predicts your artwork will likely take significantly longer than 200 days to sell.**
                
                At 200 days, there's still a {prob_unsold_pct:.0f}% probability the artwork remains unsold. The median time to sell (50% probability) exceeds the available prediction range.
                
                **Recommendation:** Consider features that drive faster sales, such as competitive pricing, stronger descriptions, better image quality, or targeting popular styles.
                """)
            else:
                st.warning(f"""
                **For ${predicted_price:.2f} as listing price, your artwork has a high likelihood of taking longer than 200 days to sell.**
                
                Consider adjusting your price or features to improve sales timeline.
                """)
        
        # Show price and days in metrics side by side for quick reference
        metric_col1, metric_col2 = st.columns([1, 1])
        with metric_col1:
            st.metric(
                label="📌 Recommended Listing Price", 
                value=f"${predicted_price:.2f}",
                help="Predicted optimal price based on your artwork's image, description, and features"
            )
        with metric_col2:
            if median_days is not None:
                st.metric(
                    label="⏱️ Expected Days to Sell", 
                    value=f"{median_days:.0f} days",
                    help="Median time until sale at the recommended price (50% probability)"
                )
            else:
                last_prob = survival_result.get('last_survival_prob')
                if last_prob is not None:
                    st.metric(
                        label="⏱️ Expected Days to Sell", 
                        value=f">{200} days",
                        help=f"At 200 days, {last_prob*100:.0f}% probability still unsold. Median exceeds prediction range."
                    )
                else:
                    st.metric(
                        label="⏱️ Expected Days to Sell", 
                        value=">200 days",
                        help="Sale may take longer than observed timeframe"
                    )
        
        # INTERACTIVE EXPERIMENTATION SECTION
        st.markdown("---")
        st.markdown("### 🎲 Experiment with Different Pricing Strategies")
        st.markdown("""
        **Want to see how changing your listing price or offering discounts affects the time to sell?**
        
        Enter different values below to explore various scenarios while keeping all other artwork features the same.
        """)
        
        exp_col1, exp_col2 = st.columns([1, 1])
        
        with exp_col1:
            custom_price = st.number_input(
                "Custom Listing Price ($)",
                min_value=1.0,
                value=float(predicted_price),
                step=10.0,
                key="custom_price_input",
                help="Try different listing prices to see how it affects days to sell"
            )
        
        with exp_col2:
            custom_discount = st.number_input(
                "Maximum Discount (as fraction, 0-1)",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                key="custom_discount_input",
                help="Enter discount as a fraction (e.g., 0.15 for 15% discount)"
            )
        
        if st.button("🔄 Recalculate with Custom Price/Discount", type="secondary", key="recalculate_button"):
            with st.spinner("Calculating new timeline..."):
                try:
                    # Store original values for comparison
                    st.session_state.original_price = predicted_price
                    st.session_state.original_median_days = median_days
                    
                    custom_cox_features = cox_features.copy()
                    custom_cox_features['max_discount_by_week'] = float(custom_discount)
                    
                    custom_survival = predict_survival_curve(
                        custom_price,
                        custom_cox_features,
                        st.session_state.cox_model,
                        st.session_state.cox_medians
                    )
                    
                    if custom_survival is not None:
                        # Store custom results for display on rerun
                        st.session_state.custom_price = custom_price
                        st.session_state.custom_discount = custom_discount
                        st.session_state.custom_median_days = custom_survival['median_survival']
                        st.session_state.show_custom_results = True
                        
                        # Update main session state with new values so metrics update
                        st.session_state.predicted_price = custom_price
                        st.session_state.survival_result = custom_survival
                        st.session_state.cox_features = custom_cox_features
                        
                        st.rerun()  # Rerun to show updated values in main display
                        
                except Exception as e:
                    st.error(f"Error in recalculation: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
        
        # Display custom results if they exist (persists across reruns)
        if hasattr(st.session_state, 'show_custom_results') and st.session_state.show_custom_results:
            col_title, col_reset = st.columns([4, 1])
            with col_title:
                st.markdown("#### 📍 Updated Prediction")
            with col_reset:
                if st.button("↩️ Reset", help="Reset to original prediction"):
                    # Clear custom results
                    st.session_state.show_custom_results = False
                    # Restore original values
                    if hasattr(st.session_state, 'original_price'):
                        st.session_state.predicted_price = st.session_state.original_price
                    st.rerun()
            
            custom_median_days = st.session_state.custom_median_days
            custom_price_display = st.session_state.custom_price
            custom_discount_display = st.session_state.custom_discount
            
            if custom_median_days is not None:
                # Calculate change from original
                if hasattr(st.session_state, 'original_median_days') and st.session_state.original_median_days is not None:
                    days_diff = custom_median_days - st.session_state.original_median_days
                    pct_change = (days_diff / st.session_state.original_median_days) * 100
                    
                    if days_diff < 0:
                        change_text = f"🟢 **{abs(days_diff):.0f} days faster** ({abs(pct_change):.1f}% improvement)"
                    elif days_diff > 0:
                        change_text = f"🔴 **{days_diff:.0f} days slower** ({pct_change:.1f}% slower)"
                    else:
                        change_text = "➡️ **Same timeline**"
                else:
                    change_text = ""
                
                st.success(f"""
                **Updated prediction with ${custom_price_display:.2f} listing price and {custom_discount_display*100:.0f}% maximum discount:**
                
                **Expected days to sell: {custom_median_days:.0f} days**
                
                {change_text}
                """)
            else:
                last_prob = survival_result.get('last_survival_prob')
                if last_prob is not None:
                    st.warning(f"""
                    **For ${custom_price_display:.2f} as listing price with {custom_discount_display*100:.0f}% maximum discount:**
                    
                    At 200 days, there's still a {last_prob*100:.0f}% probability the artwork remains unsold.
                    Sale may take longer than 200 days.
                    """)
                else:
                    st.warning(f"""
                    **For ${custom_price_display:.2f} as listing price with {custom_discount_display*100:.0f}% maximum discount:**
                    
                    Sale may take longer than 200 days.
                    """)
    
    # Advisory for better predictions
    st.markdown("---")
    st.info(
        "💡 **For More Accurate Predictions:** Include details about your reputation metrics, "
        "shipping policies, return policies, exhibition history, and artistic background in your "
        "artwork description. The more comprehensive your description, the better our AI can "
        "assess your artwork's market value."
    )
    
    # Footer with model information
    st.markdown("---")
    st.markdown("### 📊 About the Models")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("""
        **Price Prediction Model:**
        - Multi-modal deep learning architecture
        - Combines CLIP image embeddings with text descriptions
        - Trained on thousands of Etsy art listings
        - Incorporates artwork features and shop metrics
        """)
    
    with col_info2:
        st.markdown("""
        **Sales Timeline Model:**
        - Cox Proportional Hazard survival analysis
        - Accounts for censored data (unsold items)
        - Model concordance: **0.763** (strong predictive power)
        - Considers price, features, and market dynamics
        """)
    
    st.markdown("---")
    st.caption("© 2025 | Built with Streamlit, PyTorch, and R | For academic research purposes")

if __name__ == "__main__":
    main() 
