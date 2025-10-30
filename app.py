import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

# -------------------- CONFIGURACIÓN DE LA PÁGINA --------------------
st.set_page_config(
    page_title="Analizador de fotos",
    page_icon="",
    layout="wide"
)

# -------------------- CARGA DEL MODELO --------------------
@st.cache_resource
def cargar_modelo_yolo(ruta_modelo='yolov5s.pt'):
    """Carga el modelo YOLOv5 de forma compatible con distintas versiones de PyTorch."""
    try:
        import yolov5
        try:
            modelo = yolov5.load(ruta_modelo, weights_only=False)
            return modelo
        except TypeError:
            modelo = yolov5.load(ruta_modelo)
            return modelo
        except Exception:
            st.warning("Intentando método alternativo de carga del modelo...")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            modelo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            return modelo
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.info("""
        **Solución posible:**
        1. Instalar dependencias correctas:
           ```
           pip install torch==1.12.0 torchvision==0.13.0
           pip install yolov5==7.0.9
           ```
        2. Verifica la ubicación del archivo del modelo.
        3. Si el problema persiste, descarga el modelo con `torch.hub`.
        """)
        return None

# -------------------- INTERFAZ PRINCIPAL --------------------
st.title("Analizador de fotos")
st.markdown("""
Esta aplicación detecta objetos en imágenes capturadas por tu cámara.  
Ajusta los parámetros en el menú lateral para personalizar el proceso de detección.
""")

# Cargar el modelo
with st.spinner("🚀 Cargando modelo YOLOv5..."):
    modelo = cargar_modelo_yolo()

# -------------------- CONFIGURACIÓN LATERAL --------------------
if modelo:
    st.sidebar.title("⚙️ Configuración del Detector")

    with st.sidebar:
        st.subheader("🎯 Parámetros de detección")
        modelo.conf = st.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.01)
        modelo.iou = st.slider("Umbral IoU", 0.0, 1.0, 0.45, 0.01)
        st.caption(f"Confianza: {modelo.conf:.2f} | IoU: {modelo.iou:.2f}")
        
        st.subheader("🧩 Opciones avanzadas")
        try:
            modelo.agnostic = st.checkbox("NMS independiente de clases", False)
            modelo.multi_label = st.checkbox("Múltiples etiquetas por objeto", False)
            modelo.max_det = st.number_input("Número máximo de detecciones", 10, 2000, 1000, 10)
        except:
            st.warning("⚠️ Algunas opciones avanzadas no están disponibles en esta versión.")
    
    # -------------------- CAPTURA DE IMAGEN --------------------
    st.markdown("---")
    st.subheader("Captura o toma una foto")

    imagen_capturada = st.camera_input("Toma una fotografía para analizar")

    if imagen_capturada:
        bytes_imagen = imagen_capturada.getvalue()
        img_cv = cv2.imdecode(np.frombuffer(bytes_imagen, np.uint8), cv2.IMREAD_COLOR)

        # -------------------- DETECCIÓN --------------------
        with st.spinner("🔍 Analizando la imagen..."):
            try:
                resultados = modelo(img_cv)
            except Exception as e:
                st.error(f"❌ Error durante la detección: {str(e)}")
                st.stop()

        # -------------------- PROCESAMIENTO DE RESULTADOS --------------------
        try:
            predicciones = resultados.pred[0]
            cajas = predicciones[:, :4]
            puntuaciones = predicciones[:, 4]
            categorias = predicciones[:, 5]

            col1, col2 = st.columns(2)

            with col1:
                st.subheader(" Imagen con detecciones")
                resultados.render()
                st.image(img_cv, channels='BGR', use_container_width=True)

            with col2:
                st.subheader("Analisis Completo")

                etiquetas = modelo.names
                conteo = {}
                for cat in categorias:
                    cat_idx = int(cat.item()) if hasattr(cat, "item") else int(cat)
                    conteo[cat_idx] = conteo.get(cat_idx, 0) + 1

                data = []
                for cat, cantidad in conteo.items():
                    etiqueta = etiquetas[cat]
                    confianza_prom = puntuaciones[categorias == cat].mean().item() if len(puntuaciones) > 0 else 0
                    data.append({
                        "Categoría": etiqueta,
                        "Cantidad": cantidad,
                        "Confianza promedio": f"{confianza_prom:.2f}"
                    })

                if data:
                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=True)
                    st.bar_chart(df.set_index("Categoría")["Cantidad"])
                else:
                    st.info("Hay.")
                    st.caption("Sugerencia: baja el umbral de confianza para ampliar detecciones.")

        except Exception as e:
            st.error(f"❌ Error al procesar resultados: {str(e)}")
else:
    st.error("⚠️ No se pudo cargar la imagen. Verifica dependencias y vuelve a intentarlo.")
    st.stop()

# -------------------- PIE DE PÁGINA --------------------
st.markdown("---")
st.caption("""
**Acerca de esta app:**  
Aplicación desarrollada con Streamlit y PyTorch utilizando el modelo YOLOv5.  
Permite la detección de objetos en tiempo real desde tu cámara o imágenes cargadas.
""")
