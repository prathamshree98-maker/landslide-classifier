import streamlit as st
import torch
import numpy as np
import h5py
from torchvision import models, transforms
import torch.nn as nn
import time

st.set_page_config(page_title="Terrain Analytics", page_icon="🛰️")


BG_COLOR = "#0d1117"       
PANEL_BG = "#21262d"       
TEXT_COLOR = "#e6e8eb"     
ACCENT_COLOR = "#10b981"   
HOVER_BG = "#30363d"       
BORDER_COLOR = "#3d444d"   
POSITIVE_GREEN = "#34d399" 
NEGATIVE_RED = "#f87171"   

st.markdown(
    f"""
    <style>
    .stApp > div {{
        background-color: {BG_COLOR};
        color: {TEXT_COLOR};
    }}
    .stFileUploader > div > div {{
        background-color: {PANEL_BG};
        border: 2px solid {ACCENT_COLOR};
        border-radius: 8px;
    }}
    .stButton > button {{
        background-color: {ACCENT_COLOR};
        color: {BG_COLOR};
        border: 2px solid {ACCENT_COLOR};
        border-radius: 6px;
        font-weight: 500;
        min-width: 120px;
    }}
    .stButton > button:hover {{
        background-color: {HOVER_BG};
        border-color: {ACCENT_COLOR};
    }}
    .stExpander > div > div {{
        background-color: {PANEL_BG};
        border: 1px solid {BORDER_COLOR};
        border-radius: 6px;
    }}
    hr {{
        border-color: {BORDER_COLOR};
    }}
    .stAlert > div {{
        background-color: {HOVER_BG};
        color: {TEXT_COLOR};
        border-radius: 6px;
    }}
    .stSuccess > div {{
        background-color: {POSITIVE_GREEN};
        color: {BG_COLOR};
        border-radius: 6px;
    }}
    .stError > div {{
        background-color: {NEGATIVE_RED};
        color: {BG_COLOR};
        border-radius: 6px;
    }}
    .stMetric > label {{
        color: {TEXT_COLOR};
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .stMetric > value {{
        color: {ACCENT_COLOR};
        font-size: 24px;
        font-weight: 600;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Header 
st.title("TERRAIN ANALYTICS")
st.markdown(
    "Multispectral Landslide Classification System • Early warning for slope instability"
)

#Model loading 
@st.cache_resource
def load_model(model_path="landslide_resnet18.pth"):
    """Load the trained ResNet18 model modified for 14-channel input."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    # Replace first conv layer for 14 input channels
    model.conv1 = nn.Conv2d(
        in_channels=14,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )
    # Replace final layer for binary classification
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

model_path = st.sidebar.text_input("Trained model path (.pth)", "landslide_resnet18.pth")
if model_path:
    try:
        model, device = load_model(model_path)
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        st.stop()

#File upload 
st.markdown("### Input imagery")
st.markdown("Upload HDF5 satellite imagery")
st.caption(".h5 files · 14 spectral channels")

uploaded_file = st.file_uploader("Choose an .h5 file", type=["h5"])

# Validation helpers 
def validate_h5_img(f):
    """Validate that the .h5 file contains the required 'img' dataset."""
    if "img" not in f:
        raise ValueError("Missing 'img' dataset in .h5 file")
    img_shape = f["img"].shape
    if len(img_shape) != 3:
        raise ValueError(
            f"Unexpected image shape: {img_shape}. "
            "Expected 3 dimensions (14, H, W) or (H, W, 14)."
        )
    # Check if 14 is in first or last dimension
    has_14_channels = img_shape[0] == 14 or img_shape[-1] == 14
    if not has_14_channels:
        raise ValueError(
            f"Unexpected image shape: {img_shape}. "
            "Expected (14, H, W) or (H, W, 14) with 14 channels."
        )
    return img_shape

#Prediction
def preprocess_h5_image(h5_file):
    """Preprocess h5 image data similar to training pipeline."""
    with h5py.File(h5_file, "r") as f:
        image = f["img"][:]

    # Handle channel dimension - expect (14, H, W) or (H, W, 14)
    if image.shape[0] == 14:
        image = np.transpose(image, (1, 2, 0))  # (14, H, W) -> (H, W, 14)
    elif image.shape[-1] == 14:
        pass  # already (H, W, 14)
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    # Min-max normalisation (match training pipeline)
    img_min = float(np.min(image))
    img_max = float(np.max(image))
    image = (image - img_min) / (img_max - img_min + 1e-8)

    # Convert to tensor and permute to (channels, height, width)
    image = torch.tensor(image, dtype=torch.float32)
    image = image.permute(2, 0, 1)  # (H, W, 14) -> (14, H, W)

    # Resize to 224x224
    image = torch.nn.functional.interpolate(
        image.unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    return image


def predict_landslide(model, device, h5_file):
    """Run prediction on uploaded h5 file. Returns both class scores."""
    image = preprocess_h5_image(h5_file)
    image = image.unsqueeze(0).to(device)  # add batch dimension

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().tolist()

    no_landslide_prob = probabilities[0] * 100  # %
    landslide_prob = probabilities[1] * 100  # %

    pred_idx = int(torch.argmax(outputs, dim=1).item())
    label_map = {0: "No Landslide", 1: "Landslide Detected"}
    label = label_map[pred_idx]

    return label, landslide_prob, no_landslide_prob, probabilities


#  Validation & prediction
if uploaded_file is not None and model_path:
    # Reset file pointer for independent reads
    uploaded_file.seek(0)

    try:
        with h5py.File(uploaded_file, "r") as f:
            img_shape = validate_h5_img(f)

        st.success(
            {
                "title": "✓ File validated",
                "detail": f"Image shape: {img_shape} (14 spectral channels)",
            },
        )

        st.info("Processing your file...")

        start_time = time.time()
        label, landslide_prob, no_landslide_prob, probabilities = predict_landslide(
            model, device, uploaded_file
        )
        inference_time = (time.time() - start_time) * 1000  # ms

        # Prediction result
        st.markdown("### Classification output")

        # Result badge
        if label == "Landslide Detected":
            st.markdown(
                f'<div style="background:{POSITIVE_GREEN}; border:1px solid {ACCENT_COLOR}; '
                f'padding:12px 16px; border-radius:6px; margin:16px 0;">'
                f"<strong>{label}</strong>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="background:{NEGATIVE_RED}; border:1px solid {ACCENT_COLOR}; '
                f'padding:12px 16px; border-radius:6px; margin:16px 0;">'
                f"<strong>{label}</strong>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Probability table
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="No Landslide",
                value=f"{no_landslide_prob:.1f}%",
                help="Model probability that the image contains no landslide activity",
            )
        with col2:
            st.metric(
                label="Landslide",
                value=f"{landslide_prob:.1f}%",
                help="Model probability that the image contains landslide activity",
            )

        # Inference metadata
        with st.expander("Inference details", expanded=False):
            st.caption(f"Inference time: **{inference_time:.1f} ms**")
            st.caption(f"Model: ResNet18 (14-channel input)")
            st.caption(f"Input resolution: 224 × 224 px")

        # Model information 
        st.markdown("### Model information")

        with st.expander("Architecture", expanded=False):
            st.markdown(
                "**Model:** ResNet18\n"
                "**Framework:** PyTorch\n"
                "**Input bands:** 14 spectral channels\n"
                "**Classification head:** Binary (2 classes)\n"
                "**Input size:** 224 × 224 px\n"
                "**Task:** Image-level binary classification"
            )

        with st.expander("Dataset & Input specifications", expanded=False):
            st.markdown(
                "**Data format:** HDF5 (.h5)\n"
                "**Dataset key:** `img`\n"
                "**Input channels:** 14 spectral bands\n"
                "**Image dimensions:** Variable (resized to 224 × 224)\n"
                "**Classification labels:**\n"
                "• 0 = No Landslide\n"
                "• 1 = Landslide Detected"
            )

        with st.expander("Previous validation results", expanded=False):
            st.markdown(
                "# Model evaluation (held-out validation set)\n"
                "• **Validation accuracy:** 90.53%\n"
                "• **Landslide F1-score:** 0.89\n"
                "• **Landslide precision:** 93%\n"
                "• **Landslide recall:** 85%\n"
                "\n"
                " *These metrics reflect performance on a fixed validation set and "
                "may differ from individual predictions. The confusion matrix below "
                "illustrates classification behaviour.*\n"
                "\n"
                "- True Positives: Correctly detected landslide images\n"
                "- False Positives: Non-landslide images misclassified as landslide\n"
                "- False Negatives: Landslide images missed by the model\n"
                "- True Negatives: Correctly identified non-landslide images"
            )

        # HDF5 explanation
        with st.expander("ℹ Understanding HDF5 input files", expanded=False):
            st.markdown(
                "**What is an HDF5 (.h5) file?**\n"
                "HDF5 (Hierarchical Data Format 5) is a scientific data format used "
                "to store large, structured datasets, including satellite and "
                "multispectral imagery.\n\n"
                "Unlike conventional JPG or PNG images (3 RGB channels), the imagery "
                "used by this application contains **14 spectral channels**. Each "
                "channel represents a separate numerical layer of image information "
                "capturing different wavelengths of light.\n\n"
                "**Expected file structure:**\n\n"
                "```\n"
                "satellite_sample.h5\n"
                "└── img\n"
                "    └── Image array\n"
                "        ├── Channel 1   (e.g., coastal aerosol)\n"
                "        ├── Channel 2   (e.g., blue)\n"
                "        ├── …\n"
                "        └── Channel 14  (e.g., cirrus)\n"
                "```\n\n"
                "The application reads the `img` dataset, preprocesses its numerical "
                "values, resizes it to 224 × 224 pixels, and passes it to the trained "
                "ResNet18 classifier.\n\n"
                "*Note: If you don't have an .h5 file, you'll need to convert your satellite "
                "imagery into HDF5 format with an `img` dataset containing 14 numerical channels "
                "before uploading.*"
            )

        # Disclaimer
        st.markdown("---")
        st.caption(
            "**Limitation:** This application performs image-level binary classification "
            "and does not identify the precise geographic boundaries or affected area of "
            "a landslide. Predictions are intended for research and demonstration purposes "
            "and should not be used for operational disaster-response decisions without "
            "independent validation and ground truth data."
        )

    except ValueError as ve:
        st.error(f"✗ {ve}")
        st.write(
            "Please ensure your .h5 file has an `img` dataset with 14 spectral channels. "
            "See the 'Understanding HDF5 input files' section above for expected structure."
        )
    except Exception as e:
        st.error(f"✗ Error processing file: {str(e)[:150]}")
        st.write("Please ensure your .h5 file has an 'img' dataset with 14 channels.")

# Footer
st.markdown("---")
st.markdown(
    "Built with Streamlit • ResNet18 • 14-channel multispectral input • "
    "for research and demonstration purposes"
)