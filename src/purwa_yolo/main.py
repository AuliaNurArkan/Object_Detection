import numpy as np
import streamlit as st
import supervision as sv
from ultralytics import YOLO
from PIL import Image 
from io import BytesIO
from pathlib import Path
from collections import Counter
import plotly.express as px

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Cache model loading - models are loaded only once per type
@st.cache_resource
def load_model(model_name: str):
    """Load YOLO model with caching for performance"""
    model_path = MODELS_DIR / f"best_{model_name}.pt"
    return YOLO(str(model_path))

# Cache annotators - reuse same annotator objects
@st.cache_resource
def get_annotators():
    """Get cached annotator objects dengan garis tebal"""
    # thickness=5 untuk mempertebal garis kotak
    box_annotator = sv.BoxAnnotator(thickness=5)
    # text_scale dan text_thickness untuk memperbesar tulisan label
    label_annotator = sv.LabelAnnotator(text_scale=1.2, text_thickness=2)
    return box_annotator, label_annotator

def detector_pipeline_pillow(image_bytes, model):
    """Optimized detection pipeline"""
    # Load and convert image
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    
    # Run inference dengan parameter "Sweet Spot"
    results = model(pil_image, conf=0.1, imgsz=3560, iou=0.8, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results).with_nms()
    
    # Get cached annotators
    box_annotator, label_annotator = get_annotators()
    
    # Supervision membutuhkan Numpy Array untuk menggambar kotak
    image_np_rgb = np.array(pil_image) 

    # Annotate image
    annotated_image = image_np_rgb.copy()
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    
    # Optimized class counting using Counter
    class_names = detections.data.get("class_name", [])
    classcounts = dict(Counter(class_names))
    
    return annotated_image, classcounts

# --- Bagian Streamlit Utama ---

st.title("🎯 Vehicle Object Detection")

# ==================== LOAD MODEL (LANGSUNG VEHICLE) ====================
with st.spinner("Loading Vehicle model..."):
    # Langsung memanggil model vehicle tanpa dropdown
    model = load_model("vehicle")

st.success("✅ Vehicle model loaded!")

# ==================== IMAGE DETECTION UI ====================

# Inisialisasi memori agar gambar langsung tampil saat tombol ditekan
if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None
if "jalankan_deteksi" not in st.session_state:
    st.session_state.jalankan_deteksi = False

st.write("💡 **Pilih gambar contoh untuk demo cepat:**")
col_ex1, col_ex2 = st.columns(2)

with col_ex1:
    if st.button("🟢 Demo Jalan Lancar", use_container_width=True):
        contoh_path = PROJECT_ROOT / "examples" / "jalan_lancar.jpg"
        if contoh_path.exists():
            with open(contoh_path, "rb") as file:
                st.session_state.image_bytes = file.read()
                st.session_state.jalankan_deteksi = True
        else:
            st.error("Gagal: Pastikan file 'jalan_lancar.jpg' ada di dalam folder 'examples'!")

with col_ex2:
    if st.button("🟡 Demo Jalan Padat", use_container_width=True):
        contoh_path = PROJECT_ROOT / "examples" / "jalan_macet.jpg"
        if contoh_path.exists():
            with open(contoh_path, "rb") as file:
                st.session_state.image_bytes = file.read()
                st.session_state.jalankan_deteksi = True
        else:
            st.error("Gagal: Pastikan file 'jalan_macet.jpg' ada di dalam folder 'examples'!")

st.markdown("---")
st.write("📂 **Atau unggah gambar Anda sendiri untuk pengujian:**")

# INI ADALAH BAGIAN UPLOADER-NYA
uploaded_file = st.file_uploader("Upload Image", accept_multiple_files=False, type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")

if uploaded_file is not None:
    # Tombol khusus untuk file yang di-upload akan muncul di sini
    if st.button("🔍 Deteksi Gambar Upload", type="primary", use_container_width=True):
        st.session_state.image_bytes = uploaded_file.getvalue()
        st.session_state.jalankan_deteksi = True

# ==================== EKSEKUSI DETEKSI ====================
# Jika memori menyatakan gambar sudah dipilih (langsung tereksekusi)
if st.session_state.jalankan_deteksi and st.session_state.image_bytes is not None:
    st.markdown("---")
    
    # Tampilkan input aslinya
    st.subheader("🖼️ Gambar Input")
    st.image(st.session_state.image_bytes, use_container_width=True)
    
    with st.spinner("Mendeteksi objek..."):
        annotated_image_rgb, classcounts = detector_pipeline_pillow(st.session_state.image_bytes, model)

    st.subheader("🎯 Detection Results")
    st.image(annotated_image_rgb, caption="Detected Objects", use_container_width=True)

    if classcounts:
        st.subheader("📊 Object Counts")
        col1, col2 = st.columns([1, 2])
        with col1:
            for class_name, count in classcounts.items():
                st.metric(label=class_name, value=count)

        # Analisis Kepadatan
        st.subheader("🚦 Analisis Kepadatan Lalu Lintas")
        total_kendaraan = sum(classcounts.values())
        st.metric(label="🚗 Total Keseluruhan Kendaraan", value=total_kendaraan)

        if total_kendaraan < 10:
            st.success("Status Lalu Lintas: Lancar 🟢")
        elif total_kendaraan <= 18:
            st.warning("Status Lalu Lintas: Padat Merayap 🟡")
        else:
            st.error("Status Lalu Lintas: Macet 🔴")

        # Visualisasi Plotly
        st.subheader("📈 Distribusi Objek Terdeteksi")
        fig = px.pie(
            names=list(classcounts.keys()),
            values=list(classcounts.values()),
            title="Proporsi Kelas Objek",
            hole=0.3,
        )
        fig.update_traces(textinfo="label+percent+value")
        st.plotly_chart(fig, use_container_width=True)
        
        # Tombol reset
        if st.button("🔄 Bersihkan Hasil", use_container_width=True):
            st.session_state.image_bytes = None
            st.session_state.jalankan_deteksi = False
            st.rerun()
    else:
        st.info("Tidak ada objek yang terdeteksi pada gambar ini.")