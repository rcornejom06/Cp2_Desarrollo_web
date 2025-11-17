"""
🍌 Sistema de Detección de Enfermedades del Banano
Interfaz web con Streamlit
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import os

# ========== CONFIGURACIÓN DE LA PÁGINA ==========
st.set_page_config(
    page_title="Detección de Enfermedades del Banano",  # Corregido: "Detección" no "Detencion"
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========== ESTILOS CSS ==========
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

# ========== INFORMACIÓN DETALLADA DE ENFERMEDADES ==========
DISEASE_INFO = {
    "Cordana": {
        "icon": "🟤",
        "color": "#8B4513",
        "nombre_cientifico": "Cordana musae",  # Corregido: "Cordana" no "Cordama"
        "descripcion": """
        La **Cordana** es una enfermedad fúngica que causa manchas foliares en las hojas del banano. 
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
                "🧪 Aplicar fungicidas a base de **Mancozeb** (2-3 g/L de agua)",
                "🧪 Usar **Clorotalonil** en dosis de 2 ml/L de agua",
                "🧪 Alternar con **Azoxystrobin** para evitar resistencia",
                "📅 Aplicar cada 10-14 días durante época lluviosa",
                "⚠️ Rotar productos para prevenir resistencia del hongo"
            ],
            "organico": [
                "🌿 Extracto de **ajo** (100g de ajo/litro de agua)",
                "🍃 Té de **cola de caballo** como fungicida natural",
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
        La **Sigatoka** es una de las enfermedades más devastadoras del banano a nivel mundial. 
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
                "🧪 **Triazoles**: Propiconazole (0.3-0.5 ml/L) o Difenoconazole",
                "🧪 **Estrobilurinas**: Azoxystrobin (0.8-1 ml/L)",
                "🧪 **Mancozeb** como protectante (2-3 g/L)",
                "🧪 **Aceite mineral** (10-15 ml/L) como adherente",
                "📅 Programa de 8-12 ciclos por año según presión de enfermedad",
                "🔄 Rotación estricta de ingredientes activos",
                "⚠️ **CRÍTICO**: Aplicar antes de que aparezcan síntomas"
            ],
            "biologico": [
                "🦠 **Bacillus subtilis** (concentración según fabricante)",
                "🍄 **Trichoderma harzianum** aplicado al suelo",
                "🌿 Extractos de **nim (Neem)** al 2-3%"
            ],
            "calendario": [
                "📅 **Época lluviosa**: Aplicaciones cada 7-10 días",
                "☀️ **Época seca**: Aplicaciones cada 14-21 días",
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
        **Pestalotiopsis** causa manchas foliares y puede afectar también frutos y tallos. 
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
                "🧪 **Mancozeb** (2.5-3 g/L) como protectante",
                "🧪 **Clorotalonil** (2-2.5 ml/L)",
                "🧪 **Carbendazim** (1 g/L) - sistémico",
                "🧪 **Tiofanato metílico** (1-1.5 g/L)",
                "📅 Aplicar cada 10-15 días",
                "🌧️ Reaplicar después de lluvias fuertes"
            ],
            "organico": [
                "🌿 Extracto de **canela** (fungicida natural)",
                "🧄 Solución de ajo + jabón potásico",
                "🍃 Extracto de **ortiga** para fortalecer defensas",
                "🌱 Purín de **cola de caballo**"
            ],
            "nutricional": [
                "🌱 Aplicar **Silicio** para fortalecer tejidos",
                "🍃 **Calcio** foliar para endurecer hojas",
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
        ¡Excelente! La planta se encuentra **saludable** sin signos visibles de enfermedad. 
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
                "🌱 **NPK** equilibrado según etapa fenológica",
                "💊 **Calcio y Magnesio** para vigor",
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

# ========== FUNCIONES ==========

@st.cache_resource
def load_trained_model():
    """Carga el modelo entrenado con manejo de compatibilidad"""

    # Buscar modelos disponibles
    model_paths = [
        'models/keras_model.h5'
    ]

    for model_path in model_paths:
        if not os.path.exists(model_path):
            continue

        try:
            # Intentar cargar sin compilar (para compatibilidad)
            model = load_model(model_path, compile=False)

            # Recompilar manualmente
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            st.sidebar.success(f"✅ Modelo cargado: {model_path}")
            return model

        except Exception as e:
            st.sidebar.warning(f"⚠️ Error con {model_path}: {str(e)[:50]}...")
            continue

    # Si ninguno funcionó
    return None


def preprocess_image(image):
    """Preprocesa la imagen para el modelo"""

    try:
        # 1. Convertir a RGB
        if image.mode != 'RGB':
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))

            if image.mode == 'RGBA':
                rgb_image.paste(image, (0, 0), image)
            else:
                rgb_image.paste(image.convert('RGB'))

            image = rgb_image

        # 2. Redimensionar
        img = image.resize((224, 224), Image.Resampling.LANCZOS)

        # 3. Convertir a numpy array
        img_array = np.array(img, dtype=np.float32)

        # 4. Verificar que solo tenga 3 canales
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        # 5. Verificar dimensiones
        assert img_array.shape == (224, 224, 3), f"Shape incorrecto: {img_array.shape}"

        # 6. Normalizar
        img_array = img_array / 255.0

        # 7. Añadir dimensión de batch
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        st.error(f"Error procesando imagen: {e}")
        return None


# ========== SIDEBAR ==========
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/banana.png", width=100)
    st.header("ℹ️ Información")
    st.markdown("""
    ### Enfermedades detectables:
    - 🟤 **Cordana**
    - 🟢 **Healthy** (Saludable)
    - 🟡 **Pestalotiopsis**
    - ⚫ **Sigatoka**
    
    ### Cómo usar:
    1. Sube una imagen de hoja de banano
    2. Presiona "Analizar Imagen"
    3. Obtén diagnóstico y tratamiento
    
    ---
    **Desarrollado por:** Roger Cornejo  
    **Universidad:** [Tu Universidad]  
    **Año:** 2025
    """)

# ========== TÍTULO PRINCIPAL ==========
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

        # Info de imagen
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
        model = load_trained_model()

        if model is None:
            st.error("""
            ⚠️ **Modelo no encontrado**
            
            Necesitas entrenar el modelo primero:
```bash
            python entrenamiento.py
```
            
            O verifica que exista: `models/best_model.h5` o `models/best_model.keras`
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
                img_array = preprocess_image(image)

                if img_array is None:
                    st.error("No se pudo procesar la imagen")
                    progress_bar.empty()
                    status_text.empty()
                else:
                    progress_bar.progress(40)
                    time.sleep(0.3)

                    status_text.text("🤖 Ejecutando modelo de IA...")
                    predictions = model.predict(img_array, verbose=0)
                    progress_bar.progress(70)
                    time.sleep(0.3)

                    status_text.text("📊 Analizando resultados...")
                    predicted_class_idx = np.argmax(predictions)
                    confidence = predictions[0][predicted_class_idx] * 100

                    disease_names = list(DISEASE_INFO.keys())
                    predicted_disease = disease_names[predicted_class_idx]
                    disease_data = DISEASE_INFO[predicted_disease]

                    progress_bar.progress(100)
                    status_text.text("✅ ¡Análisis completado!")
                    time.sleep(0.5)

                    progress_bar.empty()
                    status_text.empty()

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

                    # Alerta según severidad
                    if predicted_disease == "Healthy":
                        st.success("✅ ¡Excelente! La planta está saludable. Continúa con las prácticas actuales.")
                    elif predicted_disease == "Sigatoka":
                        st.error("🚨 ATENCIÓN URGENTE: Sigatoka detectada. Requiere tratamiento inmediato.")
                    else:
                        st.warning(f"⚠️ Se detectó {predicted_disease}. Se recomienda iniciar tratamiento.")

                    # ========== INFORMACIÓN DE LA ENFERMEDAD ==========
                    st.markdown("---")
                    st.markdown("## 📚 Información Detallada")

                    with st.expander("ℹ️ Descripción de la enfermedad", expanded=True):
                        st.markdown(disease_data["descripcion"])
                        st.markdown(f"**Nivel de severidad:** {disease_data['severidad']}")

                    with st.expander("🔍 Síntomas característicos"):
                        for sintoma in disease_data["sintomas"]:
                            st.markdown(f"- {sintoma}")

                    # ========== TRATAMIENTOS RECOMENDADOS ==========
                    st.markdown("---")
                    st.markdown("## 💊 Plan de Tratamiento Recomendado")

                    for categoria, tratamientos in disease_data["tratamiento"].items():
                        with st.expander(f"📋 {categoria.upper().replace('_', ' ')}", expanded=True):
                            for tratamiento in tratamientos:
                                st.markdown(f"{tratamiento}")

                    # ========== ADVERTENCIAS ==========
                    st.markdown("---")
                    st.markdown('<div class="warning-card">', unsafe_allow_html=True)
                    st.markdown("""
                    ⚠️ **ADVERTENCIAS IMPORTANTES:**
                    - Siempre usa equipo de protección personal (EPP) al aplicar productos químicos
                    - Respeta los períodos de carencia antes de la cosecha
                    - Alterna productos para evitar resistencia
                    - Consulta con un ingeniero agrónomo para casos severos
                    - Mantén registro de todas las aplicaciones
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # ========== BOTÓN DE DESCARGA ==========
                    st.markdown("---")
                    report = f"""
REPORTE DE DIAGNÓSTICO - SISTEMA DE DETECCIÓN DE ENFERMEDADES DEL BANANO
========================================================================

RESULTADO DEL ANÁLISIS:
- Enfermedad detectada: {predicted_disease}
- Confianza: {confidence:.2f}%
- Nombre científico: {disease_data['nombre_cientifico']}
- Severidad: {disease_data['severidad']}

DESCRIPCIÓN:
{disease_data['descripcion']}

SÍNTOMAS:
{chr(10).join(['- ' + s for s in disease_data['sintomas']])}

TRATAMIENTOS RECOMENDADOS:
"""

                    for cat, treats in disease_data['tratamiento'].items():
                        report += f"\n{cat.upper()}:\n"
                        report += '\n'.join(['  ' + t for t in treats]) + '\n'

                    st.download_button(
                        label="📄 Descargar Reporte Completo (TXT)",
                        data=report,
                        file_name=f"reporte_{predicted_disease}_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 30px; color: #888;'>
        <p style='font-size: 16px;'>🍌 Sistema de Detección de Enfermedades del Banano</p>
        <p>Desarrollado usando Deep Learning | TensorFlow + Streamlit</p>
        <p style='font-size: 12px; margin-top: 10px;'>© 2025 - Todos los derechos reservados</p>
    </div>
""", unsafe_allow_html=True)