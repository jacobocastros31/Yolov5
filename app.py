import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image

# --- Configuración de la página ---
st.set_page_config(page_title="Visor OCR", page_icon="📸", layout="centered")

# --- Título principal con estilo web ---
st.markdown("""
<h1 style='text-align: center; color: #1E90FF;'>
📸 Reconocimiento Óptico de Caracteres (OCR)
</h1>
<p style='text-align: center; font-size: 18px; color: #555;'>
Captura una imagen con tu cámara y deja que nuestro sistema lea automáticamente el texto por ti.
</p>
<hr style='border: 1px solid #ddd;'>
""", unsafe_allow_html=True)

# --- Captura de imagen ---
img_file_buffer = st.camera_input("Toma una foto para analizar el texto:")

# --- Barra lateral con opciones ---
with st.sidebar:
    st.header("⚙️ Opciones de procesamiento")
    filtro = st.radio("Selecciona un filtro para mejorar la lectura:", ('Con Filtro', 'Sin Filtro'))
    st.markdown("---")
    st.info("💡 Consejo: Si el texto está en fondo oscuro, prueba usar el filtro para mejorar los resultados.")

# --- Procesamiento de imagen y OCR ---
if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    if filtro == 'Con Filtro':
        cv2_img = cv2.bitwise_not(cv2_img)

    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(img_rgb)

    # --- Mostrar resultados ---
    st.markdown("<h2 style='color:#1E90FF;'>📝 Texto detectado:</h2>", unsafe_allow_html=True)
    
    if text.strip():
        st.success(text)
    else:
        st.warning("No se detectó texto en la imagen. Intenta tomar otra foto con mejor iluminación.")
