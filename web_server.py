"""
🍌 Sistema de Detección de Enfermedades del Banano
Compatible con Teachable Machine
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import os

# ========== CONFIGURACIÓN ==========
st.set_page_config(
    page_title="Detección de Enfermedades del Banano",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded",
)


#Configuración de la pagina

st.set_page_config(
    page_title="Detencion de Enfermedades del Banano",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-weight: bold;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        padding: 20px 0;
    }
    .disease-box {
        padding: 25px;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        margin: 20px 0;
        text-align: center;
    }
    .healthy-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .treatment-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 15px 0;
        border-left: 5px solid #4CAF50;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)
DISEASE_INFO ={
    "Cordana": {
        "icon" : "🟤",
        "color" : "#8B4513",
        "nombre_cientifico" : "Cordama musae",
        "descripcion": """
        La *Cordana* es una enfermedad fúngica que causa manchas foliares en las hojas del banano. 
        Se caracteriza por lesiones ovaladas con centro gris y bordes amarillentos.
        """,
        "sintomas": [
            "Manchas ovaladas en las hojas",
            "Centro grisáceo con halo amarillo",
            "Anillos concéntricos en lesiones maduras",
            "Puede causar defoliación prematura"
        ],
        "tratamiento": {
            "cultural": [
                "🌿 Eliminar y quemar hojas severamente infectadas",
                "💧 Mejorar el drenaje del suelo para evitar exceso de humedad",
                "✂️ Realizar poda sanitaria de hojas afectadas",
                "🌱 Mantener distancias adecuadas entre plantas para ventilación",
                "🗑️ Eliminar residuos vegetales del suelo"
            ],
            "quimico": [
                "🧪 Aplicar fungicidas a base de *Mancozeb* (2-3 g/L de agua)",
                "🧪 Usar *Clorotalonil* en dosis de 2 ml/L de agua",
                "🧪 Alternar con *Azoxystrobin* para evitar resistencia",
                "📅 Aplicar cada 10-14 días durante época lluviosa",
                "⚠️ Rotar productos para prevenir resistencia del hongo"
            ],
            "organico": [
                "🌿 Extracto de *ajo* (100g de ajo/litro de agua)",
                "🍃 Té de *cola de caballo* como fungicida natural",
                "🧴 Bicarbonato de sodio (5g/L) + aceite vegetal",
                "🌱 Compost bien descompuesto para fortalecer la planta"
            ],
            "preventivo": [
                "🔍 Inspeccionar plantaciones semanalmente",
                "💧 Riego por goteo para evitar mojar follaje",
                "🌤️ Evitar trabajar cuando las plantas están mojadas",
                "📊 Mantener registro de aplicaciones"
            ]
        },
        "prevencion": "Mantener buena ventilación entre plantas, evitar exceso de humedad y realizar monitoreo constante.",
        "severidad": "Media - Alta"
    },

    "Sigatoka": {
        "icon": "⚫",
        "color": "#2C3E50",
        "nombre_cientifico": "Mycosphaerella fijiensis (Sigatoka Negra) / Mycosphaerella musicola (Sigatoka Amarilla)",
        "descripcion": """
        La *Sigatoka* es una de las enfermedades más devastadoras del banano a nivel mundial. 
        Reduce la capacidad fotosintética de la planta y puede disminuir la producción hasta en un 50%.
        """,
        "sintomas": [
            "Rayas o manchas alargadas en hojas",
            "Lesiones que evolucionan de amarillo a negro",
            "Necrosis del tejido foliar",
            "Defoliación severa en casos avanzados",
            "Maduración prematura de frutos"
        ],
        "tratamiento": {
            "cultural": [
                "🌿 Eliminar hojas con más del 50% de afectación",
                "🔪 Deshoje sanitario cada 1-2 semanas",
                "💧 Sistema de drenaje eficiente",
                "🌱 Usar variedades resistentes cuando sea posible",
                "📏 Mantener densidad de siembra adecuada (1,600-2,000 plantas/ha)"
            ],
            "quimico": [
                "🧪 *Triazoles*: Propiconazole (0.3-0.5 ml/L) o Difenoconazole",
                "🧪 *Estrobilurinas*: Azoxystrobin (0.8-1 ml/L)",
                "🧪 *Mancozeb* como protectante (2-3 g/L)",
                "🧪 *Aceite mineral* (10-15 ml/L) como adherente",
                "📅 Programa de 8-12 ciclos por año según presión de enfermedad",
                "🔄 Rotación estricta de ingredientes activos",
                "⚠️ *CRÍTICO*: Aplicar antes de que aparezcan síntomas"
            ],
            "biologico": [
                "🦠 *Bacillus subtilis* (concentración según fabricante)",
                "🍄 *Trichoderma harzianum* aplicado al suelo",
                "🌿 Extractos de *nim (Neem)* al 2-3%"
            ],
            "calendario": [
                "📅 *Época lluviosa*: Aplicaciones cada 7-10 días",
                "☀️ *Época seca*: Aplicaciones cada 14-21 días",
                "🔍 Monitoreo de estado de evolución semanal",
                "📊 Ajustar según índice de infección"
            ]
        },
        "prevencion": "Sistema de alerta temprana, deshoje preventivo y aplicación de fungicidas en calendario estricto.",
        "severidad": "MUY ALTA - Requiere atención inmediata"
    },

    "Pestalotiopsis": {
        "icon": "🟡",
        "color": "#FFD700",
        "nombre_cientifico": "Pestalotiopsis spp.",
        "descripcion": """
        *Pestalotiopsis* causa manchas foliares y puede afectar también frutos y tallos. 
        Se desarrolla en condiciones de alta humedad y puede causar pérdidas económicas significativas.
        """,
        "sintomas": [
            "Manchas irregulares de color marrón",
            "Lesiones con bordes definidos",
            "Posible presencia de acérvulos (cuerpos fructíferos)",
            "Afecta principalmente hojas viejas",
            "Puede causar tizón en frutos"
        ],
        "tratamiento": {
            "cultural": [
                "🌿 Poda y eliminación de hojas afectadas",
                "🍃 Remover hojas basales senescentes",
                "💧 Evitar exceso de riego en follaje",
                "🌱 Mejorar nutrición de la planta (equilibrio NPK)",
                "🌤️ Mejorar circulación de aire en la plantación"
            ],
            "quimico": [
                "🧪 *Mancozeb* (2.5-3 g/L) como protectante",
                "🧪 *Clorotalonil* (2-2.5 ml/L)",
                "🧪 *Carbendazim* (1 g/L) - sistémico",
                "🧪 *Tiofanato metílico* (1-1.5 g/L)",
                "📅 Aplicar cada 10-15 días",
                "🌧️ Reaplicar después de lluvias fuertes"
            ],
            "organico": [
                "🌿 Extracto de *canela* (fungicida natural)",
                "🧄 Solución de ajo + jabón potásico",
                "🍃 Extracto de *ortiga* para fortalecer defensas",
                "🌱 Purín de *cola de caballo*"
            ],
            "nutricional": [
                "🌱 Aplicar *Silicio* para fortalecer tejidos",
                "🍃 *Calcio* foliar para endurecer hojas",
                "💊 Microelementos (Zn, Mn, Cu) vía foliar",
                "🌿 Bioestimulantes a base de algas marinas"
            ]
        },
        "prevencion": "Nutrición balanceada, manejo de humedad y eliminación de tejido senescente.",
        "severidad": "Media"
    },

    "Healthy": {
        "icon": "🟢",
        "color": "#4CAF50",
        "nombre_cientifico": "N/A",
        "descripcion": """
        ¡Excelente! La planta se encuentra *saludable* sin signos visibles de enfermedad. 
        Mantén las prácticas culturales actuales para preservar este estado.
        """,
        "sintomas": [
            "Follaje verde vigoroso",
            "Sin manchas ni lesiones",
            "Crecimiento normal",
            "Buena turgencia de hojas"
        ],
        "tratamiento": {
            "preventivo": [
                "🌱 Continuar con programa de fertilización balanceado",
                "💧 Mantener riego adecuado sin encharcamiento",
                "🔍 Monitoreo semanal para detección temprana",
                "🌿 Deshoje sanitario preventivo de hojas viejas",
                "📊 Mantener registros de campo actualizados"
            ],
            "nutricional": [
                "🌱 *NPK* equilibrado según etapa fenológica",
                "💊 *Calcio y Magnesio* para vigor",
                "🍃 Microelementos (Fe, Zn, Mn, B)",
                "🌿 Aplicación de materia orgánica",
                "📅 Análisis de suelo cada 6 meses"
            ],
            "cultural": [
                "✂️ Deshije oportuno (3-4 hijos por cepa)",
                "🌾 Control de malezas",
                "🌱 Mantener cobertura vegetal",
                "💧 Sistema de drenaje funcional",
                "🔄 Rotación de herramientas desinfectadas"
            ],
            "monitoreo": [
                "🔍 Inspección visual semanal",
                "📸 Registro fotográfico de condiciones",
                "🌡️ Monitoreo de condiciones climáticas",
                "📊 Llevar bitácora de campo"
            ]
        },
        "prevencion": "Mantener las buenas prácticas actuales y realizar monitoreo preventivo continuo.",
        "severidad": "Ninguna - Planta saludable"
    }
}

# ========== FUNCIONES PARA TEACHABLE MACHINE ==========

@st.cache_resource
def load_teachable_machine_model():
    """Carga modelo de Teachable Machine"""
    model_path = 'models/keras_model.h5'
    labels_path = 'models/labels.txt'

    if not os.path.exists(model_path):
        st.error(f"❌ No se encontró: {model_path}")
        return None, None

    try:
        # Cargar modelo
        model = load_model(model_path, compile=False)
        # Recompilar
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Cargar labels
        class_names = []
        if os.path.exists(labels_path):
            with open(labels_path, 'r', encoding='utf-8') as f:
                class_names = [line.strip().split(' ', 1)[1] for line in f.readlines()]
        else:
            # Si no existe labels.txt, usar orden por defecto
            class_names = ['Cordana', 'Healthy', 'Pestalotiopsis', 'Sigatoka']
            st.sidebar.warning("⚠️ labels.txt no encontrado, usando orden por defecto")

        st.sidebar.success(f"✅ Modelo cargado: Teachable Machine")
        st.sidebar.info(f"🏷️ Clases: {', '.join(class_names)}")

        return model, class_names

    except Exception as e:
        st.sidebar.error(f"❌ Error: {e}")
        return None, None


def preprocess_image_teachable_machine(image):

    try:
        # 1. Convertir a RGB
        if image.mode != 'RGB':
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'RGBA':
                rgb_image.paste(image, (0, 0), image)
            else:
                rgb_image.paste(image.convert('RGB'))
            image = rgb_image

        # 2. Redimensionar a 224x224 (Teachable Machine size)
        image = image.resize((224, 224), Image.Resampling.LANCZOS)

        # 3. Convertir a numpy array
        img_array = np.array(image, dtype=np.float32)

        # 4. Verificar dimensiones
        if img_array.shape != (224, 224, 3):
            raise ValueError(f"Shape incorrecto: {img_array.shape}")

        # 5. Normalizar [0, 255] -> [0, 1]
        img_array = img_array / 255.0

        # 6. Añadir dimensión de batch
        img_array = np.expand_dims(img_array, axis=0)

        # 7. Teachable Machine a veces espera float32
        img_array = img_array.astype(np.float32)

        return img_array

    except Exception as e:
        st.error(f"Error procesando imagen: {e}")
        return None


model, CLASS_NAMES = load_teachable_machine_model()

st.markdown("""
    <h1>🍌 Sistema Inteligente de Detección de Enfermedades del Banano</h1>
    <p style='text-align: center; font-size: 18px; color: #555; margin-bottom: 30px;'>
        Diagnóstico automático con recomendaciones de tratamiento personalizadas
    </p>
""", unsafe_allow_html=True)

# ========== INTERFAZ PRINCIPAL ==========
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Cargar Imagen de Hoja")

    uploaded_file = st.file_uploader(
        "Arrastra aquí tu imagen o haz clic para seleccionar",
        type=['jpg', 'jpeg', 'png'],
        help="Formatos: JPG, JPEG, PNG (máx. 10MB)"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="✅ Imagen cargada correctamente", use_container_width=True)

        st.markdown(f"""
        <div style='background: white; padding: 15px; border-radius: 10px; margin-top: 10px;'>
            <b>📏 Resolución:</b> {image.size[0]} x {image.size[1]} px<br>
            <b>📁 Formato:</b> {image.format}<br>
            <b>💾 Tamaño:</b> {uploaded_file.size / 1024:.1f} KB
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("### 🔬 Análisis y Diagnóstico")

    if uploaded_file is None:
        st.info("👈 Por favor, sube una imagen en el panel izquierdo para comenzar")
    else:
        if model is None or CLASS_NAMES is None:
            st.error("""
            ⚠️ **Modelo no encontrado**

            Asegúrate de tener estos archivos en la carpeta `models/`:
            - `keras_model.h5` (el modelo de Teachable Machine)
            - `labels.txt` (los nombres de las clases)

            Descarga ambos desde Teachable Machine.
            """)
        else:
            if st.button("🔍 ANALIZAR IMAGEN", type="primary", use_container_width=True):

                # Barra de progreso
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("🧠 Cargando imagen...")
                progress_bar.progress(20)
                time.sleep(0.3)

                status_text.text("🔬 Preprocesando datos...")
                img_array = preprocess_image_teachable_machine(image)

                if img_array is None:
                    st.error("No se pudo procesar la imagen")
                    progress_bar.empty()
                    status_text.empty()
                else:
                    progress_bar.progress(40)
                    time.sleep(0.3)

                    status_text.text("🤖 Ejecutando modelo de IA...")

                    # Predecir
                    predictions = model.predict(img_array, verbose=0)

                    progress_bar.progress(70)
                    time.sleep(0.3)

                    status_text.text("📊 Analizando resultados...")

                    # Obtener clase predicha
                    predicted_class_idx = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_class_idx] * 100

                    # Obtener nombre de la enfermedad
                    predicted_disease = CLASS_NAMES[predicted_class_idx]

                    # Verificar que existe en DISEASE_INFO
                    if predicted_disease not in DISEASE_INFO:
                        st.error(f"❌ Clase '{predicted_disease}' no encontrada")
                        st.write(f"Clases del modelo: {CLASS_NAMES}")
                        st.write(f"Clases en DISEASE_INFO: {list(DISEASE_INFO.keys())}")

                        # Intentar mapear
                        mapping = {
                            'cordana': 'Cordana',
                            'healthy': 'Healthy',
                            'pestalotiopsis': 'Pestalotiopsis',
                            'sigatoka': 'Sigatoka'
                        }
                        predicted_disease = mapping.get(predicted_disease.lower(), predicted_disease)

                    disease_data = DISEASE_INFO.get(predicted_disease)

                    if disease_data is None:
                        st.error(f"No hay información para: {predicted_disease}")
                        st.stop()

                    progress_bar.progress(100)
                    status_text.text("✅ ¡Análisis completado!")
                    time.sleep(0.5)

                    progress_bar.empty()
                    status_text.empty()

                    # ========== MOSTRAR TODAS LAS PROBABILIDADES ==========
                    st.markdown("### 📊 Probabilidades de cada clase:")
                    for i, class_name in enumerate(CLASS_NAMES):
                        prob = predictions[0][i] * 100
                        st.progress(prob / 100, text=f"{class_name}: {prob:.2f}%")

                    # ========== RESULTADO PRINCIPAL ==========
                    box_class = "healthy-box" if predicted_disease == "Healthy" else "disease-box"

                    st.markdown(f"""
                    <div class='{box_class}'>
                        <h1 style='font-size: 48px; margin: 0;'>{disease_data['icon']}</h1>
                        <h2 style='margin: 10px 0;'>{predicted_disease}</h2>
                        <h3 style='margin: 5px 0;'>Confianza: {confidence:.1f}%</h3>
                        <p style='font-style: italic; margin-top: 10px;'>{disease_data['nombre_cientifico']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # ... (resto del código de visualización igual) ...

                    if predicted_disease == "Healthy":
                        st.success("✅ ¡Excelente! La planta está saludable.")
                    elif predicted_disease == "Sigatoka":
                        st.error("🚨 ATENCIÓN URGENTE: Sigatoka detectada.")
                    else:
                        st.warning(f"⚠️ Se detectó {predicted_disease}.")

                    # Información detallada
                    st.markdown("---")
                    st.markdown("## 📚 Información Detallada")

                    with st.expander("ℹ️ Descripción de la enfermedad", expanded=True):
                        st.markdown(disease_data["descripcion"])
                        st.markdown(f"**Nivel de severidad:** {disease_data['severidad']}")

                    with st.expander("🔍 Síntomas característicos"):
                        for sintoma in disease_data["sintomas"]:
                            st.markdown(f"- {sintoma}")

                    st.markdown("---")
                    st.markdown("## 💊 Plan de Tratamiento Recomendado")

                    for categoria, tratamientos in disease_data["tratamiento"].items():
                        with st.expander(f"📋 {categoria.upper().replace('_', ' ')}", expanded=True):
                            for tratamiento in tratamientos:
                                st.markdown(f"{tratamiento}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 30px; color: #888;'>
        <p style='font-size: 16px;'>🍌 Sistema de Detección de Enfermedades del Banano</p>
        <p>Desarrollado con ❤️ usando Teachable Machine + Streamlit</p>
        <p style='font-size: 12px; margin-top: 10px;'>© 2025</p>
    </div>
""", unsafe_allow_html=True)