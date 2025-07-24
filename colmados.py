import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import re
import pydeck as pdk
from sklearn.cluster import KMeans
from streamlit_extras.metric_cards import style_metric_cards
from urllib.parse import quote

# ----------------------------------------
# CONFIGURACIÓN INICIAL
# ----------------------------------------
st.set_page_config(
    page_title="🚀 Dashboard Levantamiento de Mercado - Niveo", 
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
        .main {background-color: #f8f9fa;}
        .stMetric {border: 1px solid #dee2e6; border-radius: 10px; padding: 15px;}
        .stPlotlyChart {border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
        .stDataFrame {border-radius: 10px;}
        .css-1aumxhk {background-color: #ffffff;}
        .stTabs [data-baseweb="tab-list"] {gap: 10px;}
        .stTabs [data-baseweb="tab"] {border-radius: 8px 8px 0 0;}
        .stTabs [aria-selected="true"] {background-color: #f0f2f6;}
        .stAlert {border-radius: 10px;}
        .green {color: #28a745;}
        .red {color: #dc3545;}
        .yellow {color: #ffc107;}
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------
# FUNCIONES DE PROCESAMIENTO DE DATOS
# ----------------------------------------
@st.cache_data(ttl=3600)
def cargar_datos():
    SHEET_ID = "15Dy2dBinbn4E-BmSm6eid9QzroZTqKwAD95RgHwWUSY"
    SHEET_NAME = "Form responses"
    
    try:
        sheet_encoded = quote(SHEET_NAME)
        url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={sheet_encoded}"
        df = pd.read_csv(url)
        
        # Verificar columnas críticas
        columnas_requeridas = [
            'SELECCION BARRIO/SECTOR',
            'TIPO DE COLMADO',
            'CUALES MARCAS ESTAN PRESENTES EN EL ESTABLECIMIENTO',
            'CON QUE FRECUENCIA COMPRA PAPEL HIGIENICO.',
            'Geolocalizacion'
        ]
        
        for col in columnas_requeridas:
            if col not in df.columns:
                st.warning(f"Columna importante no encontrada: {col}")
                
        return df
    
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        return pd.DataFrame({
            'Timestamp': [datetime.now()],
            'SELECCION BARRIO/SECTOR': ['24 DE ABRIL'],
            'TIPO DE COLMADO': ['COLMADO PEQ'],
            'CUALES MARCAS ESTAN PRESENTES EN EL ESTABLECIMIENTO': ['Niveo, Scott'],
            'CON QUE FRECUENCIA COMPRA PAPEL HIGIENICO.': ['SEMANAL'],
            'Geolocalizacion': ['Latitude: 19.4080807, Longitude: -70.5333939'],
            'CAMBIO DE NOMBRE?': ['NO'],
            'NOMBRE DEL ENCARGADO DEL NEGOCIO O DUENO': ['Ejemplo'],
            'CUAL ES LA QUE LE DEJA MAYOR BENEFICIO': ['Niveo'],
            'CUANTO CONSIDERA QUE USTED INFLUYE EN QUE EL CLIENTE PARA QUE SE LLEVE EL PRODUCTO QUE MAS LE INTERESE QUE COMPRE.': ['Mucho'],
            'PRECIO DEL PAPEL HIGIENICO NIVEO': [150],
            'PRECIO DEL PAPEL HIGIENICO SCOTT': [160],
            'PRECIO DEL PAPEL HIGIENICO FAMILIA': [155]
        })

def procesar_marcas(texto):
    """Procesa las respuestas múltiples de marcas"""
    if pd.isna(texto):
        return []
    
    # Limpieza básica y estandarización
    texto = str(texto).upper().strip()
    texto = re.sub(r'\s+', ' ', texto)  # Elimina espacios múltiples
    
    # Manejar diferentes separadores
    separadores = [',', ';', '/', '|', 'Y', '&']
    for sep in separadores:
        if sep in texto:
            return [m.strip() for m in texto.split(sep) if m.strip()]
    
    return [texto] if texto else []

def preprocesar_datos(df):
    # Limpiar nombres de columnas
    df.columns = [col.strip() for col in df.columns]
    
    # Procesamiento de fechas
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df['Fecha'] = df['Timestamp'].dt.date
        df['Mes'] = df['Timestamp'].dt.month_name(locale='es')
        df['Semana'] = df['Timestamp'].dt.isocalendar().week
        df['Dia'] = df['Timestamp'].dt.day_name(locale='es')
        df['Hora'] = df['Timestamp'].dt.hour
    
    # Procesamiento de geolocalización
    if 'Geolocalizacion' in df.columns:
        try:
            geo_pattern = r'Latitude:\s*([\-\d\.]+).*Longitude:\s*([\-\d\.]+)'
            geo_extracted = df['Geolocalizacion'].str.extract(geo_pattern)
            df['Latitud'] = pd.to_numeric(geo_extracted[0], errors='coerce')
            df['Longitud'] = pd.to_numeric(geo_extracted[1], errors='coerce')
            
            # Filtro para República Dominicana
            mask = (df['Latitud'].between(17, 20)) & (df['Longitud'].between(-72, -68))
            df = df[mask | df['Latitud'].isna() | df['Longitud'].isna()]
        except Exception as e:
            st.warning(f"Error procesando geolocalización: {str(e)}")
            df['Latitud'] = np.nan
            df['Longitud'] = np.nan
    
    # Procesamiento de marcas (respuestas múltiples)
    if 'CUALES MARCAS ESTAN PRESENTES EN EL ESTABLECIMIENTO' in df.columns:
        df['MARCAS_LISTA'] = df['CUALES MARCAS ESTAN PRESENTES EN EL ESTABLECIMIENTO'].apply(procesar_marcas)
    else:
        df['MARCAS_LISTA'] = [[] for _ in range(len(df))]
    
    # Procesamiento de marcas rentables
    if 'CUAL ES LA QUE LE DEJA MAYOR BENEFICIO' in df.columns:
        df['MARCA_RENTABLE'] = df['CUAL ES LA QUE LE DEJA MAYOR BENEFICIO'].apply(lambda x: str(x).upper().strip() if pd.notna(x) else np.nan)
    
    # Procesamiento de precios para todas las marcas posibles
    marcas_posibles = ['NIVEO', 'SCOTT', 'FAMILIA', 'ELITE', 'DOMINO', 'GAVIOTA', 'BINGO', 'HI', 'PETALO', 'SOFT']
    for marca in marcas_posibles:
        col_precio = f'PRECIO {marca}'
        if col_precio in df.columns:
            df[col_precio] = pd.to_numeric(df[col_precio], errors='coerce')
    
    return df

# ----------------------------------------
# CARGAR Y PROCESAR DATOS
# ----------------------------------------
df = cargar_datos()
df = preprocesar_datos(df)
marcas_explotadas = df.explode('MARCAS_LISTA').dropna(subset=['MARCAS_LISTA'])

# ----------------------------------------
# SIDEBAR CON FILTROS
# ----------------------------------------
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=Niveo", width=150)
    st.title("Filtros Avanzados")
    
    # Filtro de fecha
    if 'Fecha' in df.columns:
        fecha_min = df['Fecha'].min()
        fecha_max = df['Fecha'].max()
        rango_fechas = st.date_input(
            "Rango de fechas",
            [fecha_min, fecha_max],
            min_value=fecha_min,
            max_value=fecha_max
        )
        if len(rango_fechas) == 2:
            df = df[(df['Fecha'] >= rango_fechas[0]) & (df['Fecha'] <= rango_fechas[1])]
    
    # Filtro de sectores
    if 'SELECCION BARRIO/SECTOR' in df.columns:
        sectores_options = sorted(df['SELECCION BARRIO/SECTOR'].dropna().unique().tolist())
        sectores = st.multiselect(
            "Seleccionar sectores",
            options=sectores_options,
            default=sectores_options[:3] if len(sectores_options) > 3 else sectores_options
        )
        if sectores:
            df = df[df['SELECCION BARRIO/SECTOR'].isin(sectores)]
    
    # Filtro de tipo de establecimiento
    if 'TIPO DE COLMADO' in df.columns:
        tipos_options = sorted(df['TIPO DE COLMADO'].dropna().unique().tolist())
        tipos = st.multiselect(
            "Tipo de establecimiento",
            options=tipos_options,
            default=tipos_options
        )
        if tipos:
            df = df[df['TIPO DE COLMADO'].isin(tipos)]
    
    # Filtro por marcas presentes
    if not marcas_explotadas.empty:
        marcas_options = sorted(marcas_explotadas['MARCAS_LISTA'].dropna().unique().tolist())
        marcas_seleccionadas = st.multiselect(
            "Filtrar por marcas presentes",
            options=marcas_options,
            default=[]
        )
        if marcas_seleccionadas:
            df = df[df['MARCAS_LISTA'].apply(lambda x: any(marca in x for marca in marcas_seleccionadas))]
    
    # Filtro por frecuencia de compra
    if 'CON QUE FRECUENCIA COMPRA PAPEL HIGIENICO.' in df.columns:
        frecuencias = sorted(df['CON QUE FRECUENCIA COMPRA PAPEL HIGIENICO.'].dropna().unique().tolist())
        frecuencias_seleccionadas = st.multiselect(
            "Filtrar por frecuencia de compra",
            options=frecuencias,
            default=frecuencias
        )
        if frecuencias_seleccionadas:
            df = df[df['CON QUE FRECUENCIA COMPRA PAPEL HIGIENICO.'].isin(frecuencias_seleccionadas)]
    
    st.markdown(f"**Datos mostrados:** {len(df)} de {len(cargar_datos())} registros")
    
    if st.button("Resetear filtros"):
        st.rerun()

# ----------------------------------------
# FUNCIONES AUXILIARES PARA LOS TABS
# ----------------------------------------
def generar_mapa_calor_sectores(df):
    if 'SELECCION BARRIO/SECTOR' in df.columns:
        conteo_sectores = df['SELECCION BARRIO/SECTOR'].value_counts().reset_index()
        conteo_sectores.columns = ['Sector', 'Cantidad']
        
        fig = px.bar(
            conteo_sectores,
            x='Sector',
            y='Cantidad',
            title="Distribución por Sector",
            color='Cantidad',
            color_continuous_scale='Viridis'
        )
        return fig
    return None

def generar_nube_palabras(textos, titulo):
    if textos.empty:
        return None
    
    texto_completo = ' '.join(textos.dropna().astype(str))
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        max_words=50
    ).generate(texto_completo)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(titulo, fontsize=16)
    return fig

def analizar_precios(df):
    precios_data = []
    marcas_posibles = ['NIVEO', 'SCOTT', 'FAMILIA', 'ELITE', 'DOMINO', 'GAVIOTA', 'BINGO', 'HI', 'PETALO', 'SOFT']
    
    for marca in marcas_posibles:
        col_precio = f'PRECIO {marca}'
        if col_precio in df.columns:
            precios = df[col_precio].dropna()
            if not precios.empty:
                precios_data.append({
                    'Marca': marca,
                    'Promedio': precios.mean(),
                    'Mínimo': precios.min(),
                    'Máximo': precios.max(),
                    'Mediana': precios.median(),
                    'Conteo': len(precios)  # Esta es la columna que agregamos
                })
    
    # Verificamos si hay datos antes de crear el DataFrame
    if precios_data:
        df_precios = pd.DataFrame(precios_data)
        # Verificamos si la columna 'Conteo' existe antes de ordenar
        if 'Conteo' in df_precios.columns:
            return df_precios.sort_values('Conteo', ascending=False)
        return df_precios
    return pd.DataFrame() 

# ----------------------------------------
# INTERFAZ PRINCIPAL CON 10 TABS
# ----------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📊 Resumen General", 
    "🗺️ Análisis Geográfico", 
    "🏷️ Presencia de Marca", 
    "💰 Precios y Rentabilidad", 
    "🔁 Frecuencia de Compra",
    "💬 Influencia del Vendedor",
    "📦 Proveedores y Canales",
    "📝 Respuestas Abiertas",
    "📅 Seguimiento Histórico",
    "📥 Exportación de Reportes"
])

# ----------------------------------------
# TAB 1: RESUMEN GENERAL
# ----------------------------------------
with tab1:
    st.header("📊 Resumen General del Mercado", divider="rainbow")
    
    # Métricas clave
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📌 Total levantamientos", df.shape[0])
    
    if 'Timestamp' in df.columns:
        ultima_fecha = df['Timestamp'].max().strftime('%d/%m/%Y')
        col2.metric("🕒 Último levantamiento", ultima_fecha)
    else:
        col2.metric("🕒 Último levantamiento", "No disponible")
    
    if 'CAMBIO DE NOMBRE?' in df.columns:
        cambios = df['CAMBIO DE NOMBRE?'].value_counts(normalize=True)
        if 'SI' in cambios.index:
            porcentaje_cambio = cambios['SI'] * 100
            col3.metric("🔄 % Cambio de nombre", f"{porcentaje_cambio:.1f}%")
        else:
            col3.metric("🔄 % Cambio de nombre", "0%")
    else:
        col3.metric("🔄 % Cambio de nombre", "No disponible")
    
    if not marcas_explotadas.empty:
        total_marcas = marcas_explotadas['MARCAS_LISTA'].nunique()
        col4.metric("🏷️ Marcas identificadas", total_marcas)
    else:
        col4.metric("🏷️ Marcas identificadas", "No disponible")
    
    style_metric_cards()
    
    # Mapa de calor por sector
    st.subheader("🌡️ Distribución por Sector", divider="gray")
    fig_mapa_calor = generar_mapa_calor_sectores(df)
    if fig_mapa_calor:
        st.plotly_chart(fig_mapa_calor, use_container_width=True)
    
    # Distribución por tipo de establecimiento
    st.subheader("🏪 Distribución por Tipo de Establecimiento", divider="gray")
    if 'TIPO DE COLMADO' in df.columns:
        tipo_dist = df['TIPO DE COLMADO'].value_counts().reset_index()
        tipo_dist.columns = ['Tipo', 'Cantidad']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                tipo_dist, 
                values='Cantidad', 
                names='Tipo',
                title="Distribución por Tipo",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                tipo_dist,
                x='Tipo',
                y='Cantidad',
                title="Cantidad por Tipo",
                color='Tipo'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Top marcas
    st.subheader("🏷️ Top Marcas en el Mercado", divider="gray")
    if not marcas_explotadas.empty:
        top_marcas = marcas_explotadas['MARCAS_LISTA'].value_counts().head(10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                top_marcas,
                x=top_marcas.index,
                y=top_marcas.values,
                title="Top 10 Marcas más Comunes",
                color=top_marcas.values
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Análisis Pareto
            total_menciones = top_marcas.sum()
            pareto = (top_marcas.cumsum() / total_menciones * 100).reset_index()
            pareto.columns = ['Marca', 'Porcentaje Acumulado']
            
            fig = px.line(
                pareto,
                x='Marca',
                y='Porcentaje Acumulado',
                title="Análisis Pareto de Marcas",
                markers=True
            )
            fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="80%")
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------
# TAB 2: ANÁLISIS GEOGRÁFICO
# ----------------------------------------
with tab2:
    st.header("🗺️ Análisis Geográfico", divider="rainbow")
    
    has_geo_data = 'Latitud' in df.columns and 'Longitud' in df.columns and not df[['Latitud', 'Longitud']].dropna().empty
    
    if has_geo_data:
        df_geo = df.dropna(subset=['Latitud', 'Longitud'])
        
        # Configuración del mapa
        view_state = pdk.ViewState(
            latitude=df_geo['Latitud'].mean(),
            longitude=df_geo['Longitud'].mean(),
            zoom=11,
            pitch=50
        )
        
        # Selector de tipo de visualización
        tipo_mapa = st.radio(
            "Tipo de visualización",
            ["Puntos", "Heatmap", "Clusters"],
            horizontal=True
        )
        
        # Selector de coloración
        color_por = st.selectbox(
            "Colorar por",
            ["Tipo de establecimiento", "Presencia de marca", "Sector"],
            index=0
        )
        
        # Configurar capa según selección
        if tipo_mapa == "Puntos":
            if color_por == "Tipo de establecimiento" and 'TIPO DE COLMADO' in df_geo.columns:
                # Asignar colores por tipo
                tipos = df_geo['TIPO DE COLMADO'].unique()
                colores = px.colors.qualitative.Plotly[:len(tipos)]
                color_map = {tipo: colores[i] for i, tipo in enumerate(tipos)}
                df_geo['color'] = df_geo['TIPO DE COLMADO'].map(color_map)
                
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=df_geo,
                    get_position=['Longitud', 'Latitud'],
                    get_color='color',
                    get_radius=100,
                    pickable=True
                )
            elif color_por == "Presencia de marca":
                marca_seleccionada = st.selectbox(
                    "Seleccionar marca para visualizar",
                    sorted(marcas_explotadas['MARCAS_LISTA'].dropna().unique()),
                    index=0
                )
                df_geo['color'] = df_geo['MARCAS_LISTA'].apply(lambda x: [0, 200, 0, 160] if marca_seleccionada in x else [200, 0, 0, 160])
                
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=df_geo,
                    get_position=['Longitud', 'Latitud'],
                    get_color='color',
                    get_radius=100,
                    pickable=True
                )
            else:  # Por sector
                sectores = df_geo['SELECCION BARRIO/SECTOR'].unique()
                colores = px.colors.qualitative.Alphabet[:len(sectores)]
                color_map = {sector: colores[i] for i, sector in enumerate(sectores)}
                df_geo['color'] = df_geo['SELECCION BARRIO/SECTOR'].map(color_map)
                
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=df_geo,
                    get_position=['Longitud', 'Latitud'],
                    get_color='color',
                    get_radius=100,
                    pickable=True
                )
        elif tipo_mapa == "Heatmap":
            layer = pdk.Layer(
                "HeatmapLayer",
                data=df_geo,
                get_position=['Longitud', 'Latitud'],
                opacity=0.9,
                threshold=0.5,
                aggregation='"MEAN"',
                get_weight=1
            )
        else:  # Clusters
            layer = pdk.Layer(
                "HexagonLayer",
                data=df_geo,
                get_position=['Longitud', 'Latitud'],
                radius=100,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            )
        
        # Tooltip
        tooltip = {
            "html": """
                <b>Establecimiento:</b> {NOMBRE DEL ENCARGADO DEL NEGOCIO O DUENO}<br/>
                <b>Sector:</b> {SELECCION BARRIO/SECTOR}<br/>
                <b>Tipo:</b> {TIPO DE COLMADO}<br/>
                <b>Marcas:</b> {CUALES MARCAS ESTAN PRESENTES EN EL ESTABLECIMIENTO}
            """,
            "style": {
                "backgroundColor": "steelblue",
                "color": "white"
            }
        }
        
        # Mostrar mapa
        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip=tooltip
        ))
        
        # Análisis por sector seleccionado
        st.subheader("📌 Análisis por Sector", divider="gray")
        if 'SELECCION BARRIO/SECTOR' in df.columns:
            sector_seleccionado = st.selectbox(
                "Seleccionar sector para detalle",
                sorted(df['SELECCION BARRIO/SECTOR'].dropna().unique()),
                index=0
            )
            
            df_sector = df[df['SELECCION BARRIO/SECTOR'] == sector_seleccionado]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("🏪 Establecimientos", len(df_sector))
            col2.metric("🛒 Tipos de negocio", df_sector['TIPO DE COLMADO'].nunique())
            
            if not marcas_explotadas.empty:
                marcas_sector = marcas_explotadas[marcas_explotadas['SELECCION BARRIO/SECTOR'] == sector_seleccionado]
                col3.metric("🏷️ Marcas presentes", marcas_sector['MARCAS_LISTA'].nunique())
            
            # Mapa del sector
            df_sector_geo = df_sector.dropna(subset=['Latitud', 'Longitud'])
            if not df_sector_geo.empty:
                view_state_sector = pdk.ViewState(
                    latitude=df_sector_geo['Latitud'].mean(),
                    longitude=df_sector_geo['Longitud'].mean(),
                    zoom=14,
                    pitch=50
                )
                
                layer_sector = pdk.Layer(
                    "ScatterplotLayer",
                    data=df_sector_geo,
                    get_position=['Longitud', 'Latitud'],
                    get_color='[200, 30, 0, 160]',
                    get_radius=100,
                    pickable=True
                )
                
                st.pydeck_chart(pdk.Deck(
                    layers=[layer_sector],
                    initial_view_state=view_state_sector,
                    tooltip=tooltip
                ))
    else:
        st.warning("No hay suficientes datos geográficos para mostrar el mapa")

# ----------------------------------------
# TAB 3: PRESENCIA DE MARCA
# ----------------------------------------
with tab3:
    st.header("🏷️ Presencia de Marca", divider="rainbow")
    
    if not marcas_explotadas.empty:
        # Selector de marcas para comparar
        marcas_disponibles = sorted(marcas_explotadas['MARCAS_LISTA'].dropna().unique())
        marcas_seleccionadas = st.multiselect(
            "Seleccionar Marcas para Comparar", 
            marcas_disponibles,
            default=marcas_disponibles[:3] if len(marcas_disponibles) >= 3 else marcas_disponibles
        )
        
        if marcas_seleccionadas:
            # Presencia por sector
            st.subheader("📊 Presencia por Sector", divider="gray")
            presencia_sector = pd.crosstab(
                marcas_explotadas['SELECCION BARRIO/SECTOR'],
                marcas_explotadas['MARCAS_LISTA']
            )[marcas_seleccionadas]
            
            fig = px.bar(
                presencia_sector,
                barmode='group',
                title="Presencia de Marcas por Sector",
                labels={'value': 'Cantidad', 'variable': 'Marca'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabla dinámica
            st.subheader("📋 Tabla Dinámica de Presencia", divider="gray")
            st.dataframe(
                presencia_sector.style.background_gradient(cmap='Blues'),
                use_container_width=True
            )
            
            # Penetración de mercado
            st.subheader("📈 Penetración de Mercado", divider="gray")
            total_establecimientos = len(df)
            penetracion_data = []
            
            for marca in marcas_seleccionadas:
                presencia = (marcas_explotadas['MARCAS_LISTA'] == marca).sum()
                penetracion = presencia / total_establecimientos * 100
                penetracion_data.append({
                    'Marca': marca,
                    'Establecimientos': presencia,
                    'Penetración (%)': penetracion
                })
            
            df_penetracion = pd.DataFrame(penetracion_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    df_penetracion,
                    x='Marca',
                    y='Penetración (%)',
                    title="Penetración de Mercado",
                    color='Marca',
                    text='Penetración (%)'
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.dataframe(
                    df_penetracion.style.format({'Penetración (%)': '{:.1f}%'}).background_gradient(
                        subset=['Penetración (%)'], 
                        cmap='Greens'
                    ),
                    use_container_width=True
                )
            
            # Mapa de presencia
            st.subheader("🗺️ Mapa de Presencia", divider="gray")
            if has_geo_data:
                marca_mapa = st.selectbox(
                    "Seleccionar marca para mapa",
                    marcas_seleccionadas,
                    index=0
                )
                
                df_marca = df[df['MARCAS_LISTA'].apply(lambda x: marca_mapa in x if isinstance(x, list) else False)]
                df_marca_geo = df_marca.dropna(subset=['Latitud', 'Longitud'])
                
                if not df_marca_geo.empty:
                    view_state_marca = pdk.ViewState(
                        latitude=df_marca_geo['Latitud'].mean(),
                        longitude=df_marca_geo['Longitud'].mean(),
                        zoom=12,
                        pitch=50
                    )
                    
                    layer_marca = pdk.Layer(
                        "ScatterplotLayer",
                        data=df_marca_geo,
                        get_position=['Longitud', 'Latitud'],
                        get_color='[0, 100, 200, 160]',
                        get_radius=150,
                        pickable=True
                    )
                    
                    st.pydeck_chart(pdk.Deck(
                        layers=[layer_marca],
                        initial_view_state=view_state_marca,
                        tooltip=tooltip
                    ))
                else:
                    st.warning(f"No hay datos geográficos para {marca_mapa}")
    else:
        st.warning("No hay datos de marcas disponibles para el análisis")

# ----------------------------------------
# TAB 4: PRECIOS Y RENTABILIDAD
# ----------------------------------------
with tab4:
    st.header("💰 Precios y Rentabilidad", divider="rainbow")
    
    # Análisis de precios
    st.subheader("📊 Análisis de Precios", divider="gray")
    df_precios = analizar_precios(df)
    
    if not df_precios.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Boxplot de precios
            fig = px.box(
                df_precios.melt(id_vars=['Marca'], value_vars=['Promedio', 'Mínimo', 'Máximo', 'Mediana']),
                x='Marca',
                y='value',
                color='variable',
                title="Distribución de Precios por Marca",
                labels={'value': 'Precio (RD$)', 'variable': 'Métrica'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Tabla de resumen
            st.dataframe(
                df_precios.style.format({
                    'Promedio': 'RD${:.2f}',
                    'Mínimo': 'RD${:.2f}',
                    'Máximo': 'RD${:.2f}',
                    'Mediana': 'RD${:.2f}',
                    'Conteo': '{:.0f}'
                }).background_gradient(cmap='Greens'),
                use_container_width=True
            )
    
    # Análisis de rentabilidad
    st.subheader("💵 Rentabilidad Percibida", divider="gray")
    if 'MARCA_RENTABLE' in df.columns:
        rentabilidad = df['MARCA_RENTABLE'].value_counts().reset_index()
        rentabilidad.columns = ['Marca', 'Establecimientos']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                rentabilidad,
                values='Establecimientos',
                names='Marca',
                title="Marcas con Mayor Beneficio Percibido",
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                rentabilidad,
                x='Marca',
                y='Establecimientos',
                title="Marcas más Rentables",
                color='Marca',
                text='Establecimientos'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top 5 marcas rentables
        st.subheader("🏆 Top 5 Marcas Rentables", divider="gray")
        top_5 = rentabilidad.head(5)
        
        fig = px.bar(
            top_5,
            x='Marca',
            y='Establecimientos',
            title="Marcas con Mayor Beneficio Percibido",
            color='Marca',
            text='Establecimientos'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No hay datos de rentabilidad disponibles")

# ----------------------------------------
# TAB 5: FRECUENCIA DE COMPRA
# ----------------------------------------
with tab5:
    st.header("🔁 Frecuencia de Compra", divider="rainbow")
    
    if 'CON QUE FRECUENCIA COMPRA PAPEL HIGIENICO.' in df.columns:
        # Distribución general
        st.subheader("📊 Distribución General", divider="gray")
        frecuencia = df['CON QUE FRECUENCIA COMPRA PAPEL HIGIENICO.'].value_counts().reset_index()
        frecuencia.columns = ['Frecuencia', 'Cantidad']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                frecuencia,
                values='Cantidad',
                names='Frecuencia',
                title="Frecuencia de Compra",
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                frecuencia,
                x='Frecuencia',
                y='Cantidad',
                title="Frecuencia de Compra",
                color='Frecuencia'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Por tipo de establecimiento
        st.subheader("🛒 Por Tipo de Establecimiento", divider="gray")
        if 'TIPO DE COLMADO' in df.columns:
            tabla_cruzada = pd.crosstab(
                df['TIPO DE COLMADO'],
                df['CON QUE FRECUENCIA COMPRA PAPEL HIGIENICO.']
            )
            
            fig = px.bar(
                tabla_cruzada,
                barmode='group',
                title="Frecuencia por Tipo de Establecimiento",
                labels={'value': 'Cantidad', 'variable': 'Frecuencia'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Por marca rentable
        st.subheader("🏷️ Por Marca Rentable", divider="gray")
        if 'MARCA_RENTABLE' in df.columns:
            tabla_marca = pd.crosstab(
                df['MARCA_RENTABLE'],
                df['CON QUE FRECUENCIA COMPRA PAPEL HIGIENICO.']
            )
            
            fig = px.bar(
                tabla_marca,
                barmode='group',
                title="Frecuencia por Marca Rentable",
                labels={'value': 'Cantidad', 'variable': 'Frecuencia'}
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No hay datos de frecuencia de compra disponibles")

# ----------------------------------------
# TAB 6: INFLUENCIA DEL VENDEDOR
# ----------------------------------------
with tab6:
    st.header("💬 Influencia del Vendedor", divider="rainbow")
    
    if 'CUANTO CONSIDERA QUE USTED INFLUYE EN QUE EL CLIENTE PARA QUE SE LLEVE EL PRODUCTO QUE MAS LE INTERESE QUE COMPRE.' in df.columns:
        # Distribución general
        st.subheader("📊 Percepción de Influencia", divider="gray")
        influencia = df['CUANTO CONSIDERA QUE USTED INFLUYE EN QUE EL CLIENTE PARA QUE SE LLEVE EL PRODUCTO QUE MAS LE INTERESE QUE COMPRE.'].value_counts().reset_index()
        influencia.columns = ['Nivel', 'Cantidad']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                influencia,
                values='Cantidad',
                names='Nivel',
                title="Nivel de Influencia Percibida",
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                influencia,
                y='Nivel',
                x='Cantidad',
                title="Influencia en la Decisión",
                color='Nivel',
                orientation='h'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Por sector
        st.subheader("📍 Por Sector", divider="gray")
        if 'SELECCION BARRIO/SECTOR' in df.columns:
            # Convertir a numérico para calcular promedio
            niveles = {
                'Muy Poco': 1,
                'Poco': 2,
                'Regular': 3,
                'Mucho': 4,
                'Demasiado': 5
            }
            
            df['Influencia_Num'] = df['CUANTO CONSIDERA QUE USTED INFLUYE EN QUE EL CLIENTE PARA QUE SE LLEVE EL PRODUCTO QUE MAS LE INTERESE QUE COMPRE.'].map(niveles)
            
            influencia_sector = df.groupby('SELECCION BARRIO/SECTOR')['Influencia_Num'].mean().sort_values(ascending=False).reset_index()
            influencia_sector.columns = ['Sector', 'Influencia Promedio']
            
            fig = px.bar(
                influencia_sector,
                x='Sector',
                y='Influencia Promedio',
                title="Influencia Promedio por Sector",
                color='Influencia Promedio',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Relación con rentabilidad
        st.subheader("📈 Relación con Rentabilidad", divider="gray")
        if 'MARCA_RENTABLE' in df.columns and 'Influencia_Num' in df.columns:
            df_cruzado = df.dropna(subset=['MARCA_RENTABLE', 'Influencia_Num'])
            
            fig = px.box(
                df_cruzado,
                x='MARCA_RENTABLE',
                y='Influencia_Num',
                title="Influencia vs Marca Rentable",
                labels={
                    'MARCA_RENTABLE': 'Marca Rentable',
                    'Influencia_Num': 'Nivel de Influencia'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No hay datos de influencia del vendedor disponibles")

# ----------------------------------------
# TAB 7: PROVEEDORES Y CANALES
# ----------------------------------------
with tab7:
    st.header("📦 Proveedores y Canales", divider="rainbow")
    
    # Proveedores mencionados
    st.subheader("🏭 Proveedores Mencionados", divider="gray")
    if 'A QUIEN LE COMPRA EL PAPEL HIGIENICO' in df.columns:
        # Nube de palabras
        fig_nube = generar_nube_palabras(
            df['A QUIEN LE COMPRA EL PAPEL HIGIENICO'],
            "Proveedores más mencionados"
        )
        if fig_nube:
            st.pyplot(fig_nube)
        
        # Top proveedores
        proveedores = df['A QUIEN LE COMPRA EL PAPEL HIGIENICO'].value_counts().reset_index()
        proveedores.columns = ['Proveedor', 'Cantidad']
        
        fig = px.bar(
            proveedores.head(10),
            x='Proveedor',
            y='Cantidad',
            title="Top 10 Proveedores",
            color='Cantidad'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No hay datos de proveedores disponibles")
    
    # Uso de apps digitales
    st.subheader("📱 Uso de Aplicaciones Digitales", divider="gray")
    if 'COMPRA O HA COMPRADO A TRAVES DE UNA APLIACION DIGITAL SUS PEDIDOS. SI ES SI PUEDE MENCIONAR SU NOMBRE' in df.columns:
        apps = df['COMPRA O HA COMPRADO A TRAVES DE UNA APLIACION DIGITAL SUS PEDIDOS. SI ES SI PUEDE MENCIONAR SU NOMBRE'].value_counts().reset_index()
        apps.columns = ['Aplicación', 'Cantidad']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                apps,
                values='Cantidad',
                names='Aplicación',
                title="Uso de Aplicaciones",
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(
                apps.style.background_gradient(cmap='Greens'),
                use_container_width=True
            )
        
        # Mapa de establecimientos que usan apps
        if has_geo_data:
            df_apps = df[df['COMPRA O HA COMPRADO A TRAVES DE UNA APLIACION DIGITAL SUS PEDIDOS. SI ES SI PUEDE MENCIONAR SU NOMBRE'].notna()]
            df_apps_geo = df_apps.dropna(subset=['Latitud', 'Longitud'])
            
            if not df_apps_geo.empty:
                view_state_apps = pdk.ViewState(
                    latitude=df_apps_geo['Latitud'].mean(),
                    longitude=df_apps_geo['Longitud'].mean(),
                    zoom=12,
                    pitch=50
                )
                
                layer_apps = pdk.Layer(
                    "ScatterplotLayer",
                    data=df_apps_geo,
                    get_position=['Longitud', 'Latitud'],
                    get_color='[0, 200, 100, 160]',
                    get_radius=150,
                    pickable=True
                )
                
                st.pydeck_chart(pdk.Deck(
                    layers=[layer_apps],
                    initial_view_state=view_state_apps,
                    tooltip=tooltip
                ))
    else:
        st.warning("No hay datos de aplicaciones digitales disponibles")

# ----------------------------------------
# TAB 8: RESPUESTAS ABIERTAS
# ----------------------------------------
with tab8:
    st.header("📝 Respuestas Abiertas", divider="rainbow")
    
    # Selector de pregunta
    pregunta = st.selectbox(
        "Seleccionar pregunta a analizar",
        [
            "¿Por qué considera que esa es la más rentable?",
            "¿A quién le compra?",
            "Otras observaciones"
        ],
        index=0
    )
    
    # Determinar columna según pregunta seleccionada
    if pregunta == "¿Por qué considera que esa es la más rentable?":
        columna = 'POR QUE CONSIDERA QUE ESA ES LA QUE LE DEJA MAYOR BENEFICIO'
    elif pregunta == "¿A quién le compra?":
        columna = 'A QUIEN LE COMPRA EL PAPEL HIGIENICO'
    else:
        columna = 'OBSERVACIONES'
    
    if columna in df.columns:
        # Búsqueda de términos
        termino_busqueda = st.text_input("Buscar término específico")
        
        if termino_busqueda:
            df_filtrado = df[df[columna].str.contains(termino_busqueda, case=False, na=False)]
        else:
            df_filtrado = df
        
        # Mostrar resultados
        st.dataframe(
            df_filtrado[[columna, 'SELECCION BARRIO/SECTOR', 'TIPO DE COLMADO']].dropna(),
            use_container_width=True,
            height=400
        )
        
        # Análisis de sentimiento básico
        if pregunta == "¿Por qué considera que esa es la más rentable?":
            st.subheader("🧠 Análisis de Sentimiento", divider="gray")
            
            # Palabras positivas y negativas (lista básica en español)
            palabras_positivas = ['bueno', 'excelente', 'mejor', 'beneficio', 'calidad', 'buena', 'bonito', 'rápido', 'confiable']
            palabras_negativas = ['malo', 'mal', 'peor', 'problema', 'caro', 'feo', 'difícil', 'lento', 'deficiente']
            
            conteo_positivo = 0
            conteo_negativo = 0
            
            for texto in df_filtrado[columna].dropna():
                texto = str(texto).lower()
                if any(palabra in texto for palabra in palabras_positivas):
                    conteo_positivo += 1
                if any(palabra in texto for palabra in palabras_negativas):
                    conteo_negativo += 1
            
            if conteo_positivo > conteo_negativo:
                st.markdown("**Sentimiento predominante:** <span class='green'>Positivo</span>", unsafe_allow_html=True)
            elif conteo_negativo > conteo_positivo:
                st.markdown("**Sentimiento predominante:** <span class='red'>Negativo</span>", unsafe_allow_html=True)
            else:
                st.markdown("**Sentimiento predominante:** <span class='yellow'>Neutral</span>", unsafe_allow_html=True)
            
            # Ejemplos de respuestas
            st.write("**Ejemplos de respuestas:**")
            if not df_filtrado.empty:
                ejemplos = df_filtrado[columna].dropna().sample(min(3, len(df_filtrado)))
                for ejemplo in ejemplos:
                    st.info(f"📄 {ejemplo}")
    else:
        st.warning(f"No se encontró la columna para: {pregunta}")

# ----------------------------------------
# TAB 9: SEGUIMIENTO HISTÓRICO
# ----------------------------------------
with tab9:
    st.header("📅 Seguimiento Histórico", divider="rainbow")
    
    if 'Timestamp' in df.columns:
        # Evolución temporal
        st.subheader("📈 Evolución Temporal", divider="gray")
        df_diario = df.groupby(df['Timestamp'].dt.date).size().reset_index()
        df_diario.columns = ['Fecha', 'Cantidad']
        
        fig = px.line(
            df_diario,
            x='Fecha',
            y='Cantidad',
            title="Levantamientos por Día",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribución por hora
        st.subheader("🕒 Distribución por Hora", divider="gray")
        df_hora = df.groupby('Hora').size().reset_index()
        df_hora.columns = ['Hora', 'Cantidad']
        
        fig = px.bar(
            df_hora,
            x='Hora',
            y='Cantidad',
            title="Levantamientos por Hora",
            color='Cantidad'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Registros recientes
        st.subheader("📋 Registros Recientes", divider="gray")
        columnas_interes = [
            'Timestamp', 
            'SELECCION BARRIO/SECTOR', 
            'TIPO DE COLMADO',
            'CUALES MARCAS ESTAN PRESENTES EN EL ESTABLECIMIENTO',
            'MARCA_RENTABLE'
        ]
        
        columnas_disponibles = [col for col in columnas_interes if col in df.columns]
        
        st.dataframe(
            df.sort_values('Timestamp', ascending=False)[columnas_disponibles].head(20),
            use_container_width=True,
            height=400
        )
    else:
        st.warning("No hay datos de fecha/hora para el análisis histórico")

# ----------------------------------------
# TAB 10: EXPORTACIÓN DE REPORTES
# ----------------------------------------
with tab10:
    st.header("📥 Exportación de Reportes", divider="rainbow")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Exportación de datos
        st.subheader("📤 Exportar Datos")
        
        formato = st.radio(
            "Formato de exportación", 
            ["CSV", "Excel", "JSON"],
            horizontal=True
        )
        
        if formato == "CSV":
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Descargar CSV",
                data=csv,
                file_name="reporte_mercado_niveo.csv",
                mime="text/csv"
            )
        elif formato == "Excel":
            excel_file = df.to_excel(index=False)
            st.download_button(
                "Descargar Excel",
                data=excel_file,
                file_name="reporte_mercado_niveo.xlsx",
                mime="application/vnd.ms-excel"
            )
        else:
            json_data = df.to_json(indent=2, orient='records')
            st.download_button(
                "Descargar JSON",
                data=json_data,
                file_name="reporte_mercado_niveo.json",
                mime="application/json"
            )
    
    with col2:
        # Generación de reportes por sector
        st.subheader("📄 Generar Reporte por Sector")
        
        if 'SELECCION BARRIO/SECTOR' in df.columns:
            sector_reporte = st.selectbox(
                "Seleccionar sector para reporte",
                sorted(df['SELECCION BARRIO/SECTOR'].dropna().unique()),
                index=0
            )
            
            if st.button("Generar Reporte Resumido"):
                df_sector = df[df['SELECCION BARRIO/SECTOR'] == sector_reporte]
                
                # Crear reporte
                reporte = f"""
                # 📊 Reporte de Mercado - {sector_reporte}
                
                ## 📌 Datos Generales
                - **Total establecimientos**: {len(df_sector)}
                - **Tipos de negocio**: {df_sector['TIPO DE COLMADO'].nunique()}
                - **Último levantamiento**: {df_sector['Timestamp'].max().strftime('%d/%m/%Y') if 'Timestamp' in df_sector.columns else 'No disponible'}
                
                ## 🏷️ Presencia de Marcas
                """
                
                if not marcas_explotadas.empty:
                    marcas_sector = marcas_explotadas[marcas_explotadas['SELECCION BARRIO/SECTOR'] == sector_reporte]
                    top_marcas = marcas_sector['MARCAS_LISTA'].value_counts().head(3)
                    
                    reporte += f"""
                    - **Marcas presentes**: {marcas_sector['MARCAS_LISTA'].nunique()}
                    - **Top 3 marcas**:
                        1. {top_marcas.index[0]} ({top_marcas.values[0]} menciones)
                        2. {top_marcas.index[1]} ({top_marcas.values[1]} menciones)
                        3. {top_marcas.index[2]} ({top_marcas.values[2]} menciones)
                    """
                
                reporte += """
                ## 💰 Rentabilidad
                """
                
                if 'MARCA_RENTABLE' in df_sector.columns:
                    rentabilidad = df_sector['MARCA_RENTABLE'].value_counts()
                    if not rentabilidad.empty:
                        reporte += f"""
                        - **Marca más rentable**: {rentabilidad.index[0]} ({rentabilidad.values[0]} menciones)
                        """
                
                st.markdown(reporte)
                
                # Botón para descargar reporte
                st.download_button(
                    "Descargar Reporte (Markdown)",
                    data=reporte,
                    file_name=f"reporte_{sector_reporte}.md",
                    mime="text/markdown"
                )
        else:
            st.warning("No se encontró la columna de sectores para generar reportes")
    
    # Insights automáticos
    st.subheader("🧠 Insights Automáticos", divider="gray")
    
    if not marcas_explotadas.empty:
        # Insight 1: Marca líder
        top_marca = marcas_explotadas['MARCAS_LISTA'].value_counts().idxmax()
        top_count = marcas_explotadas['MARCAS_LISTA'].value_counts()[0]
        total_menciones = len(marcas_explotadas)
        porcentaje_top = (top_count / total_menciones) * 100
        
        # Insight 2: Penetración de Niveo
        niveo_presente = 'NIVEO' in marcas_explotadas['MARCAS_LISTA'].values
        if niveo_presente:
            niveo_count = (marcas_explotadas['MARCAS_LISTA'] == 'NIVEO').sum()
            niveo_porcentaje = (niveo_count / total_menciones) * 100
        else:
            niveo_count = 0
            niveo_porcentaje = 0
        
        # Insight 3: Distribución geográfica
        sectores_con_niveo = marcas_explotadas[marcas_explotadas['MARCAS_LISTA'] == 'NIVEO']['SELECCION BARRIO/SECTOR'].nunique()
        total_sectores = df['SELECCION BARRIO/SECTOR'].nunique()
        
        # Construir el reporte
        reporte = f"""
        ## 📌 Resumen Ejecutivo - Niveo
        
        - **Marca líder**: {top_marca} ({porcentaje_top:.1f}% de menciones)
        - **Presencia Niveo**: {'✅ Presente' if niveo_presente else '❌ Ausente'} ({niveo_porcentaje:.1f}% de menciones)
        - **Cobertura geográfica**: Presente en {sectores_con_niveo} de {total_sectores} sectores
        
        ## 🔍 Recomendaciones
        """
        
        if top_marca != 'NIVEO':
            reporte += f"""
            - **Oportunidad**: {top_marca} es la marca líder con un {porcentaje_top:.1f}% de penetración.
              Considerar estrategias competitivas para ganar participación de mercado.
            """
        
        if niveo_porcentaje < 30:
            reporte += f"""
            - **Expansión**: La penetración de Niveo es del {niveo_porcentaje:.1f}%, por debajo del objetivo recomendado.
              Priorizar campañas en sectores con baja presencia.
            """
        
        if sectores_con_niveo < total_sectores:
            reporte += f"""
            - **Cobertura**: Niveo está ausente en {total_sectores - sectores_con_niveo} sectores.
              Identificar distribuidores locales en esas áreas.
            """
        
        st.markdown(reporte)
        
        # Descargar reporte
        st.download_button(
            "Descargar Insights",
            data=reporte,
            file_name="insights_niveo.md",
            mime="text/markdown"
        )
    else:
        st.warning("No hay suficientes datos para generar insights")

# ----------------------------------------
# FOOTER
# ----------------------------------------
st.divider()
st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <p>Dashboard desarrollado por Equipo de Inteligencia de Mercado - Niveo</p>
        <p>📅 Última actualización: {}</p>
    </div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
