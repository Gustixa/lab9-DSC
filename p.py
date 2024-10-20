import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats  # Añadido esta importación
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score

# Cargar datos
data = pd.read_csv("depurado.csv")

# Configuración de la app
st.set_page_config(page_title="Análisis de Combustibles en Guatemala", layout="wide")

# Estilos personalizados
st.markdown(
    """
    <style>
    .consumo {color: #E74C3C;}
    .precios {color: #F39C12;}
    .indicador-neutro {color: #F1C40F;}
    .importacion {color: #3498DB;}
    .secundario {color: #95A5A6;}
    </style>
    """, unsafe_allow_html=True)

# Encabezado con color
st.markdown('<h2>Análisis de Combustibles Importados en Guatemala</h2>', unsafe_allow_html=True)
st.markdown('<p>Este análisis explora las tendencias de importación y predicción de combustibles a lo largo del tiempo.</p>', unsafe_allow_html=True)

# Columnas de interés
columns_interes = [
    'Diesel bajo azufre',
    'Diesel ultra bajo azufre',
    'Gas licuado de petróleo',
    'Gasolina regular',
    'Gasolina superior',
    'Diesel alto azufre'
]

# Conversión de la fecha
data['Fecha'] = pd.to_datetime(data['Fecha'])
data['mes'] = data['Fecha'].dt.month
data['año'] = data['Fecha'].dt.year

# Transformación a formato largo
df_combustibles = data[['Fecha', 'año', 'mes'] + columns_interes]
df_long = pd.melt(
    df_combustibles, 
    id_vars=['Fecha', 'año', 'mes'], 
    value_vars=columns_interes, 
    var_name='tipo_combustible', 
    value_name='importaciones'
)

# Suma de importaciones
importaciones_por_mes = data.groupby('mes')[columns_interes].sum()
importaciones_por_año = data.groupby('año')[columns_interes].sum()

# Filtros de selección
col1, col2 = st.columns(2)
st.header("Importaciones segun combustible")
with col1:
    mes_seleccionado = st.selectbox(
        "Selecciona un mes:", 
        ["Todos"] + sorted(importaciones_por_mes.index.tolist())
    )

with col2:
    año_seleccionado = st.selectbox(
        "Selecciona un año:", 
        ["Todos"] + sorted(importaciones_por_año.index.tolist())
    )

# Mostrar gráficos en columnas
col1, col2 = st.columns(2)
st.header("Importaciones Totales segun época")
with col1:
    if mes_seleccionado == "Todos":
        df_plot = importaciones_por_mes
    else:
        df_plot = importaciones_por_mes.loc[[mes_seleccionado]]
    
    fig_mes = px.bar(
        df_plot.reset_index(),
        x='mes',
        y=columns_interes,
        title=f"Importaciones por Mes: {mes_seleccionado}",
        barmode='group'
    )
    # Asegurar que los meses de 1 a 12 aparezcan en el eje X
    fig_mes.update_layout(
        xaxis=dict(
            tickmode='linear',  # Configurar el modo de los ticks en 'linear'
            tick0=1,  # Primer valor en el eje
            dtick=1  # Distancia entre ticks (en este caso, 1)
        )
    )
    st.plotly_chart(fig_mes, use_container_width=True)

with col2:
    if año_seleccionado == "Todos":
        df_plot = importaciones_por_año
    else:
        df_plot = importaciones_por_año.loc[[año_seleccionado]]
    
    fig_año = px.bar(
        df_plot.reset_index(),
        x='año',
        y=columns_interes,
        title=f"Importaciones por Año:{año_seleccionado}",
        barmode='group'
    )
    # Asegurar que los meses de 1 a 12 aparezcan en el eje X
    fig_año.update_layout(
        xaxis=dict(
            tickmode='linear',  # Configurar el modo de los ticks en 'linear'
            tick0=1,  # Primer valor en el eje
            dtick=1  # Distancia entre ticks (en este caso, 1)
        )
    )
    st.plotly_chart(fig_año, use_container_width=True)

# Fila inferior: Gráficos agregados
col3, col4 = st.columns(2)

with col3:
    fig_mes_total = px.bar(
        df_long.groupby('mes')['importaciones'].sum().reset_index(),
        x='mes',
        y='importaciones',
        title="Importaciones Totales por Mes",
        color_discrete_sequence=['#3498DB']
    )
    st.plotly_chart(fig_mes_total, use_container_width=True)

with col4:
    fig_año_total = px.bar(
        df_long.groupby('año')['importaciones'].sum().reset_index(),
        x='año',
        y='importaciones',
        title="Importaciones Totales por Año",
        color_discrete_sequence=['#3498DB']
    )
    st.plotly_chart(fig_año_total, use_container_width=True)

# Filtro de tipo de combustible
tipo_seleccionado = st.selectbox("Selecciona un tipo de combustible:", columns_interes)

# Filtrar datos
df_filtrado = data[['Fecha', 'año', 'mes', tipo_seleccionado]].dropna()  # Añadimos 'año' y 'mes' al filtrado

# Análisis estadístico con Plotly
st.subheader(f"Análisis del combustible: {tipo_seleccionado}")
col5, col6 = st.columns(2)

with col5:
    # Histograma con Plotly
    fig_hist = px.histogram(
        df_filtrado,
        x=tipo_seleccionado,
        nbins=30,
        title=f'Histograma de {tipo_seleccionado}'
    )
    fig_hist.add_trace(
        go.Histogram(
            x=df_filtrado[tipo_seleccionado],
            nbinsx=30,
            name="count",
            showlegend=False
        )
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col6:
    # Q-Q plot con Plotly
    qq = stats.probplot(df_filtrado[tipo_seleccionado], dist="norm")
    fig_qq = go.Figure()
    fig_qq.add_scatter(x=qq[0][0], y=qq[0][1], mode='markers', name='Q-Q Plot')
    fig_qq.add_scatter(x=qq[0][0], y=qq[0][0] * qq[1][0] + qq[1][1], 
                      mode='lines', name='Línea de referencia')
    fig_qq.update_layout(
        title=f'Gráfico Q-Q de {tipo_seleccionado}',
        xaxis_title='Cuantiles teóricos',
        yaxis_title='Cuantiles observados'
    )
    st.plotly_chart(fig_qq, use_container_width=True)

# Gráfico de tendencia temporal
st.subheader("Tendencia de importaciones a lo largo del tiempo")
fig_line = px.line(
    df_filtrado,
    x='Fecha',
    y=tipo_seleccionado,
    title=f'{tipo_seleccionado}',
    color_discrete_sequence=['#F39C12']
)
st.plotly_chart(fig_line, use_container_width=True)

# Modelo de Regresión Lineal
st.subheader(f"Modelo de Regresión Lineal para {tipo_seleccionado}")

# Preparar datos para el modelo
X = df_filtrado[['año', 'mes']]
y = df_filtrado[tipo_seleccionado]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = root_mean_squared_error(y_test, y_pred)

# Mostrar métricas
st.write(f"**Error cuadrático medio (RMSE):** {mse:.2f}")

# Crear un DataFrame con los valores reales y predichos
df_prediccion = pd.DataFrame({
    'Índice': range(len(y_test)),
    'Valores Reales': y_test.values,
    'Valores Predichos': y_pred
})

# Gráfico de predicción vs valores reales con líneas
fig_pred = px.line(
    df_prediccion, 
    x='Índice',
    y=['Valores Reales', 'Valores Predichos'],
    title=f'Valores Reales vs Predichos para {tipo_seleccionado}',
    color_discrete_map={
        'Valores Reales': '#00FF00',      # Verde brillante
        'Valores Predichos': '#FF69B4'    # Rosa
    }
)

# Actualizar el diseño
fig_pred.update_layout(
    xaxis_title='Índice de muestra',
    yaxis_title='Valores',
    showlegend=True,
    legend_title_text='',
    # Asegurar que las líneas sean más visibles
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
    )
)

# Actualizar las líneas para hacerlas más visibles
fig_pred.update_traces(
    line=dict(width=2),    # Hacer las líneas más gruesas
)

st.plotly_chart(fig_pred, use_container_width=True)

# Añadir métricas adicionales
r2 = r2_score(y_test, y_pred)
st.write(f"**Coeficiente de determinación (R²):** {r2:.4f}")