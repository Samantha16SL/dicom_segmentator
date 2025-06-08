import streamlit as st
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from stl import mesh
import tempfile
import os

# Configuración general
st.set_page_config(page_title="DICOM Segmentator", page_icon="🧠", layout="wide")

# Tema visual
st.markdown("""
<style>
body {
    background-color: #0f172a;
    color: white;
}
header, .stSidebar {
    background-color: #1e293b;
}
h1, h2, h3, h4, h5, h6, p, label, .stRadio, .stSlider, .stSelectbox, .stButton, .stDownloadButton {
    background-color: #cbd5e1 !important;
    color: #0f172a !important;
    border-radius: 5px;
    padding: 5px;
}
.stButton>button, .stDownloadButton>button {
    background-color: #334155;
    color: white;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# Logo y encabezado
st.sidebar.image("ecovision_logo.png", use_container_width=True)
st.sidebar.markdown("### por el equipo de EcoVision")
st.title(":brain: SEGMENTADOR DICOM")
st.markdown("Una plataforma visual para segmentar, analizar y exportar imágenes médicas DICOM con estilo profesional.")

# Estado inicial
if "dicom_data" not in st.session_state:
    st.session_state.dicom_data = None
    st.session_state.image = None
    st.session_state.segmented = None
    st.session_state.slice_index = 0

# Menú
menu = st.sidebar.radio("📁 Menú:", ["📄 Subir DICOM", "🌞 Visualizar imagen", "✂️ Segmentar imagen", "📆 Exportar STL"])

# Subir archivo
if menu == "📄 Subir DICOM":
    uploaded_file = st.file_uploader("Archivo DICOM", type=["dcm"])
    if uploaded_file:
        dicom_data = pydicom.dcmread(uploaded_file)
        image = dicom_data.pixel_array.astype(np.float32)
        st.session_state.dicom_data = dicom_data
        st.session_state.image = image
        st.session_state.segmented = None
        st.session_state.slice_index = 0
        st.success("✅ Archivo cargado exitosamente.")

# Visualizar imagen
elif menu == "🌞 Visualizar imagen":
    img = st.session_state.image
    if img is not None:
        st.sidebar.subheader("Controles")
        brightness = st.sidebar.slider("Brillo", -100, 100, 0)
        contrast = st.sidebar.slider("Contraste", 0.5, 3.0, 1.0)
        adjusted = img.copy() * contrast + brightness
        adjusted = np.clip(adjusted, 0, 255)
        if img.ndim == 3 and img.shape[0] > 1:
            st.session_state.slice_index = st.sidebar.slider("Slice:", 0, img.shape[0] - 1, st.session_state.slice_index)
            slice_img = adjusted[st.session_state.slice_index, :, :]
        else:
            slice_img = adjusted
        fig, ax = plt.subplots()
        ax.imshow(slice_img, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.warning("⚠️ Sube un archivo DICOM primero.")

# Segmentar imagen
elif menu == "✂️ Segmentar imagen":
    image = st.session_state.image
    if image is not None:
        st.sidebar.subheader("Controles de Segmentación")
        estructura = st.sidebar.selectbox("Estructura:", ["Hueso", "Tejido blando", "Tumor"])
        threshold_factor = st.sidebar.slider("Umbral:", 0.0, 2.0, 1.0, 0.01)
        if estructura == "Hueso":
            threshold = np.mean(image) * threshold_factor
        elif estructura == "Tejido blando":
            threshold = np.mean(image) * (threshold_factor * 0.6)
        else:
            threshold = np.mean(image) * (threshold_factor * 1.2)
        slice_img = image[st.session_state.slice_index] if image.ndim == 3 else image
        segmented = slice_img > threshold
        st.session_state.segmented = segmented
        col1, col2 = st.columns(2)
        col1.image(slice_img, clamp=True, caption="Original", use_container_width=True)
        col2.image(segmented.astype(np.uint8) * 255, clamp=True, caption=f"Segmentado: {estructura}", use_container_width=True)
        st.success("✅ Segmentación realizada completa.")
    else:
        st.warning("⚠️ Sube un archivo DICOM primero.")

# Exportar STL
elif menu == "📆 Exportar STL":
    st.subheader("Exportar Segmentación")

    materiales = {
        "PLA": "Fácil, barato, biodegradable 🧠 📘",
        "ABS": "Resistente y duradero 🔧🦴",
        "PETG": "Transparente, fuerte, estable ⚗️🩻",
        "Resina estándar": "Alta precisión 🦷✏️",
        "Resina biocompatible": "Contacto médico aprobado 🧬🩹",
        "Nylon": "Flexible y fuerte 🧵💪",
        "TPU": "Elástico y blando ❤️🔄",
        "PVA": "Soporte soluble 💧🌱",
        "PEEK": "Alta resistencia 🟣🔒"
    }

    selected = st.sidebar.selectbox("Selecciona el material de impresión 3D", list(materiales.keys()))
    st.markdown(f"🔍 **Material seleccionado:** {selected} - {materiales[selected]}")

    if st.session_state.segmented is not None:
        st.subheader("Exportar Segmentación")
        vol = np.stack([st.session_state.segmented]*5, axis=0)
        verts, faces, _, _ = measure.marching_cubes(vol, level=0)
        malla = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                malla.vectors[i][j] = verts[f[j], :]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp_file:
            filename = f"segmentacion_{selected}.stl".replace(" ", "_")
            malla.save(tmp_file.name)
            with open(tmp_file.name, "rb") as file:
                st.download_button("📥 Descargar STL", file, file_name=filename)
        st.success("✅ STL exportado.")
    else:
        st.warning("⚠️ Primero segmenta una imagen antes de exportar el STL.")
