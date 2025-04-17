import streamlit as st  # type: ignore
import cv2
from ultralytics import YOLO
import requests  # type: ignore
from PIL import Image
import os
import io
from PIL.ExifTags import TAGS, GPSTAGS
import folium
from streamlit_folium import st_folium

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ----------------------------
# Load YOLO Model (cached)
# ----------------------------
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# ----------------------------
# YOLO Prediction
# ----------------------------
def predict_image(model, image, conf_threshold, iou_threshold):
    result = model.predict(
        image,
        conf=conf_threshold,
        iou=iou_threshold,
        device='cpu'
    )

    class_names = model.model.names
    classes = result[0].boxes.cls
    class_counts = {}

    for c in classes:
        c = int(c)
        class_counts[class_names[c]] = class_counts.get(class_names[c], 0) + 1

    prediction_text = ', '.join([f"{v} {k}{'s' if v > 1 else ''}" for k, v in class_counts.items()])
    if not prediction_text:
        prediction_text = "No Fire Detected"

    latency = round(sum(result[0].speed.values()) / 1000, 2)
    prediction_text += f" in {latency} seconds."

    res_image = result[0].plot()
    res_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)

    return res_image, prediction_text

# ----------------------------
# Extract GPS from EXIF
# ----------------------------
def extract_gps_from_image(image):
    try:
        info = image._getexif()
        if not info:
            return None

        gps_info = {}
        for tag, value in info.items():
            tag_name = TAGS.get(tag)
            if tag_name == "GPSInfo":
                for t in value:
                    sub_tag = GPSTAGS.get(t)
                    gps_info[sub_tag] = value[t]

        def convert_to_degrees(val):
            d, m, s = val
            return d[0]/d[1] + (m[0]/m[1])/60 + (s[0]/s[1])/3600

        lat = convert_to_degrees(gps_info['GPSLatitude'])
        if gps_info['GPSLatitudeRef'] == 'S':
            lat = -lat

        lon = convert_to_degrees(gps_info['GPSLongitude'])
        if gps_info['GPSLongitudeRef'] == 'W':
            lon = -lon

        return lat, lon
    except:
        return None

# ----------------------------
# Show Location on Map
# ----------------------------
def show_map(coords):
    if coords:
        lat, lon = coords
        location = (lat, lon)
        st.success(f"📍 Location detected: Latitude {lat:.5f}, Longitude {lon:.5f}")
    else:
        location = (19.0760, 72.8777)  # Mumbai as fallback
        st.warning("⚠️ No GPS data found. Showing default location: Mumbai, India")

    m = folium.Map(location=location, zoom_start=10)
    folium.Marker(location, tooltip="Wildfire Location" if coords else "Default View",
                  icon=folium.Icon(color="red")).add_to(m)
    st_folium(m, width=700, height=500)

# ----------------------------
# Main Streamlit App
# ----------------------------
def main():
    st.set_page_config(page_title="🔥 Wildfire Detector", layout="wide", page_icon="🔥")

    # Add Background Image
    st.markdown("""
    <style>
    .title-section {
        background-image: url('icon.png');
        background-size: cover;
        background-position: center;
        padding: 30px;
        border-radius: 12px;
        text-align: center;
    }
    .title-section h1 {
        color: white;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .title-section p {
        color: #f0f0f0;
        font-size: 18px;
    }
    </style>
    
""", unsafe_allow_html=True)

    # Sidebar - Model Selection
    with st.sidebar:
        st.image("icon.png", width=100)
        st.title("🔥 Wildfire Detector")

        model_type = st.radio("Select Model Type", ["Fire Detection", "General"])
        models_dir = "general-models" if model_type == "General" else "fire-models"
        model_files = sorted([f.replace(".pt", "") for f in os.listdir(models_dir) if f.endswith(".pt")])
        selected_model = st.selectbox("Model Size", model_files)
        model_path = os.path.join(models_dir, selected_model + ".pt")
        model = load_model(model_path)

        st.markdown("---")
        conf_threshold =  0.5
        iou_threshold = st.slider("Intersection-over-union Threshold", 0.0, 1.0, 0.5, 0.05)
        st.markdown("---")
        st.info("Upload an image or paste a URL to begin detection.")

    # Title and Subtitle
    st.markdown("""
        <style>
        .main-title {
            font-size: 36px;
            font-weight: bold;
            color: #e25822;
            text-align: center;
            margin-bottom: 0px;
        }
        .description {
            font-size: 18px;
            text-align: center;
            color: #f5f5f5;
            margin-bottom: 30px;
        }
        </style>
        <div class='main-title'>Wildfire Detection System</div>
        <div class='description'>Detect fire and smoke in images using a YOLO-based object detection model</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📸 Upload or Paste an Image URL")

    # Image Input
    image = None
    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)

    with col2:
        image_url = st.text_input("Or enter image URL")
        if image_url:
            try:
                response = requests.get(image_url, stream=True)
                if response.status_code == 200:
                    image = Image.open(response.raw)
                else:
                    st.error("Failed to load image from URL")
            except Exception as e:
                st.error(f"Error loading image: {e}")

    # Process Image
    if image:
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Extract GPS Coordinates
        coords = extract_gps_from_image(image)

        # Perform Detection
        with st.spinner("Detecting..."):
            pred_image, pred_text = predict_image(model, image, conf_threshold, iou_threshold)
            st.image(pred_image, caption="Prediction", use_container_width=True)
            st.success(pred_text)

            # Download Result
            buffer = io.BytesIO()
            Image.fromarray(pred_image).save(buffer, format="PNG")
            st.download_button("📥 Download Prediction", buffer.getvalue(), file_name="prediction.png", mime="image/png")

        # Show Map
        st.markdown("---")
        st.subheader("📍 Location on Map")
        show_map(coords)

# Run the app
if __name__ == "__main__":
    main()
