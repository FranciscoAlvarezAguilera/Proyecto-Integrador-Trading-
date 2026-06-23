import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import random
from datetime import datetime, timedelta

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA Y ESTILOS VISUALES (PREMIUM DARK THEME)
# ==============================================================================
st.set_page_config(
    page_title="Algoritmo Genético Inmunitario - Quantitative Trading",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para estética premium de grado institucional
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Configuración de fuentes */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stHeading, h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
    }

    /* Tarjetas de métricas personalizadas (Glassmorphism effect) */
    .metric-container {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-top: 3px solid #00e5ff;
        border-radius: 12px;
        padding: 22px 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
        transition: all 0.3s ease-in-out;
        text-align: center;
        margin-bottom: 15px;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 229, 255, 0.5);
        box-shadow: 0 10px 40px 0 rgba(0, 229, 255, 0.15);
    }
    
    .metric-val {
        font-size: 2.1rem;
        font-weight: 700;
        margin-top: 5px;
        letter-spacing: -0.5px;
    }
    
    .metric-lbl {
        font-size: 0.85rem;
        color: #9ca3af;
        text-transform: uppercase;
        font-weight: 500;
        letter-spacing: 1.2px;
    }
    
    /* Colores para valores de métricas */
    .val-positive { color: #10b981; }
    .val-negative { color: #ef4444; }
    .val-neutral { color: #00e5ff; }
    .val-warning { color: #f59e0b; }

    /* Estilo del título principal con degradado */
    .header-title {
        background: linear-gradient(135deg, #00e5ff 0%, #7c4dff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 2px;
        letter-spacing: -1px;
    }
    
    .header-subtitle {
        color: #9ca3af;
        font-size: 1.15rem;
        text-align: center;
        margin-bottom: 35px;
        font-weight: 300;
    }
    
    /* Contenedor del ADN del cromosoma */
    .dna-badge {
        background-color: rgba(124, 77, 255, 0.15);
        border: 1px solid rgba(124, 77, 255, 0.3);
        color: #d1c4e9;
        padding: 4px 8px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.95rem;
    }

    /* Tabla interactiva limpia */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 0.9rem;
        min-width: 400px;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
    }
    
    .styled-table th {
        background-color: #1e1b4b;
        color: #ffffff;
        text-align: left;
        font-weight: bold;
        padding: 12px 15px;
    }
    
    .styled-table td {
        padding: 12px 15px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CARGA DE DATOS (CON CACHÉ Y TRATAMIENTO DE COLUMNAS MULTINIVEL)
# ==============================================================================
@st.cache_data(show_spinner=False)
def fetch_ticker_data(ticker, start_dt, end_dt):
    """
    Descarga los datos históricos usando yfinance, limpia las columnas multinivel
    si es necesario, y retorna un DataFrame listo para analizar.
    """
    try:
        # Descarga
        df = yf.download(ticker, start=start_dt, end=end_dt)
        if df.empty:
            return None
        
        # Corrección de columnas multinivel que a veces devuelve yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Aseguramos el orden del índice temporal
        df = df.sort_index()
        return df
    except Exception as e:
        st.error(f"Error al descargar datos para {ticker}: {str(e)}")
        return None


# ==============================================================================
# MOTOR DE BACKTESTING Y LOGICA DE LA ESTRATEGIA (INMUNIDAD)
# ==============================================================================
def calculate_rsi(prices, period):
    """
    Calcula el RSI estándar (Wilder's EMA) para evitar dependencias de librerías externas.
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Media móvil exponencial de Wilder
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def run_backtest(df, params):
    """
    Simula la ejecución de la estrategia de trading sobre el DataFrame con base en
    los genes del cromosoma. Retorna métricas de rendimiento y señales detalladas.
    """
    n = len(df)
    close = df['Close'].values
    high = df['High'].values if 'High' in df.columns else close
    low = df['Low'].values if 'Low' in df.columns else close
    
    # Calcular indicadores del cromosoma
    p_fast = int(params['periodo_MA_rapida'])
    p_slow = int(params['periodo_MA_lenta'])
    p_rsi = int(params['periodo_RSI'])
    
    ma_fast = df['Close'].rolling(window=p_fast).mean().values
    ma_slow = df['Close'].rolling(window=p_slow).mean().values
    rsi = calculate_rsi(df['Close'], p_rsi).values
    
    # Inicialización del portafolio
    portfolio_value = np.zeros(n)
    portfolio_value[0] = 1.0  # Empezamos con capital normalizado de 1.0
    
    position = 0  # 0: Fuera del mercado (Cash), 1: Dentro del mercado (Long)
    entry_price = 0.0
    buy_signals = []
    sell_signals = []
    trades = []
    
    # Determinamos el índice de inicio válido (para evitar valores nulos de las medias)
    start_idx = p_slow + 1
    if start_idx >= n:
        return {
            'fitness': 0.0, 'sharpe': 0.0, 'drawdown': 0.0, 'total_return': 0.0,
            'equity': np.ones(n), 'buy_signals': [], 'sell_signals': [], 'trades': []
        }
        
    portfolio_value[:start_idx] = 1.0
    
    # Bucle principal de simulación temporal
    for t in range(start_idx, n):
        price_t = close[t]
        price_prev = close[t-1]
        
        # Evaluar cruces
        cross_above = (price_t > ma_fast[t]) and (price_prev <= ma_fast[t-1])
        cross_below = (price_t < ma_fast[t]) and (price_prev >= ma_fast[t-1])
        
        if position == 0:
            # ----------------- REGLA DE COMPRA (INMUNIDAD) -----------------
            # Se compra si el precio cruza al alza la MA rápida Y el RSI está sobrevendido (o lo estuvo recientemente).
            # Filtro de memoria inmunitaria: comprobamos los últimos 3 días de RSI.
            rsi_oversold = (
                (rsi[t] < params['limite_RSI_sobreventa']) or
                (rsi[t-1] < params['limite_RSI_sobreventa']) or
                (rsi[t-2] < params['limite_RSI_sobreventa'])
            )
            if cross_above and rsi_oversold and (price_t > ma_slow[t]):
                position = 1
                entry_price = price_t
                buy_signals.append((df.index[t], price_t))
            
            # Mantener valor del portafolio si estamos fuera
            portfolio_value[t] = portfolio_value[t-1]
        else:
            # ----------------- REGLA DE VENTA Y PROTECCIÓN -----------------
            # Comprobar si se ejecuta el Stop Loss en base al mínimo diario
            stopped_out = False
            if low[t] <= entry_price * (1.0 - params['stop_loss']):
                stopped_out = True
                # Estimamos salida al valor de la orden de Stop Loss
                exit_price = entry_price * (1.0 - params['stop_loss'])
                # Acotamos el precio por si hay gaps de apertura extremos
                exit_price = min(max(exit_price, low[t]), high[t])
            else:
                # Comprobar señales técnicas de salida
                # Filtro de memoria inmunitaria para sobrecompra (últimos 2 días)
                rsi_overbought = (
                    (rsi[t] > params['limite_RSI_sobrecompra']) or
                    (rsi[t-1] > params['limite_RSI_sobrecompra'])
                )
                if cross_below or rsi_overbought:
                    exit_price = price_t
                else:
                    exit_price = None
            
            if stopped_out or exit_price is not None:
                # Liquidar posición
                portfolio_value[t] = portfolio_value[t-1] * (exit_price / price_prev)
                sell_signals.append((df.index[t], exit_price))
                trades.append({
                    'entry_date': buy_signals[-1][0],
                    'exit_date': df.index[t],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': (exit_price / entry_price) - 1.0,
                    'stop_loss_hit': stopped_out
                })
                position = 0
            else:
                # Mantener posición, acumular rendimiento del día
                portfolio_value[t] = portfolio_value[t-1] * (price_t / price_prev)
                
    # Calcular retornos diarios para métricas de riesgo
    returns = np.diff(portfolio_value) / portfolio_value[:-1]
    
    # Sharpe Ratio (Anualizado)
    if len(returns) > 0 and np.std(returns) > 1e-8:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
    else:
        sharpe = 0.0
        
    # Drawdown Máximo (Maximum Drawdown)
    peaks = np.maximum.accumulate(portfolio_value)
    drawdowns = (portfolio_value - peaks) / peaks
    max_dd = np.abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
    
    # Retorno Total de la Estrategia
    total_ret = portfolio_value[-1] - 1.0
    
    # Función de Aptitud (Fitness)
    # Penaliza el Sharpe Ratio según la magnitud del Maximum Drawdown.
    # Si no hay trades o el Sharpe es negativo, el fitness es cero.
    if sharpe <= 0 or len(trades) == 0:
        fitness = 0.0
    else:
        fitness = sharpe * (1.0 - max_dd)
        
    return {
        'fitness': fitness,
        'sharpe': sharpe,
        'drawdown': max_dd,
        'total_return': total_ret,
        'equity': portfolio_value,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'trades': trades,
        'ma_fast': ma_fast,
        'ma_slow': ma_slow,
        'rsi': rsi
    }


# ==============================================================================
# MOTOR DEL ALGORITMO GENÉTICO (DESDE CERO)
# ==============================================================================
def create_random_chromosome():
    """
    Genera un individuo aleatorio con genes dentro de los rangos numéricos requeridos.
    """
    return {
        'periodo_MA_rapida': random.randint(5, 50),
        'periodo_MA_lenta': random.randint(51, 200),
        'limite_RSI_sobrecompra': random.randint(65, 85),
        'limite_RSI_sobreventa': random.randint(15, 35),
        'periodo_RSI': random.randint(7, 21),
        'stop_loss': round(random.uniform(0.01, 0.05), 4)
    }

def tournament_selection(population, fitnesses, k=3):
    """
    Selecciona un individuo usando selección por torneo de tamaño k.
    """
    selected_indices = random.sample(range(len(population)), k)
    best_idx = max(selected_indices, key=lambda idx: fitnesses[idx])
    return population[best_idx]

def crossover(parent1, parent2):
    """
    Aplica cruce en un punto balanceado sobre los genes ordenados.
    """
    genes_order = [
        'periodo_MA_rapida', 'periodo_MA_lenta', 
        'limite_RSI_sobrecompra', 'limite_RSI_sobreventa', 
        'periodo_RSI', 'stop_loss'
    ]
    # Punto de corte aleatorio (entre 1 y 5 para que ambos hereden de ambos padres)
    cut = random.randint(1, len(genes_order) - 1)
    
    child1 = {}
    child2 = {}
    
    for i, gene in enumerate(genes_order):
        if i < cut:
            child1[gene] = parent1[gene]
            child2[gene] = parent2[gene]
        else:
            child1[gene] = parent2[gene]
            child2[gene] = parent1[gene]
            
    return child1, child2

def mutate(individual, mutation_rate):
    """
    Mutación gaussiana adaptativa por gen, asegurando que los genes se mantengan
    dentro de sus límites permitidos.
    """
    bounds = {
        'periodo_MA_rapida': (5, 50),
        'periodo_MA_lenta': (51, 200),
        'limite_RSI_sobrecompra': (65, 85),
        'limite_RSI_sobreventa': (15, 35),
        'periodo_RSI': (7, 21),
        'stop_loss': (0.01, 0.05)
    }
    
    mutated = individual.copy()
    for gene, (low, high) in bounds.items():
        if random.random() < mutation_rate:
            if isinstance(low, int):
                # Generamos una perturbación basada en la amplitud del rango
                span = high - low
                noise = random.normalvariate(0, span * 0.12)
                delta = int(round(noise))
                if delta == 0:
                    delta = random.choice([-1, 1])
                mutated[gene] = min(max(mutated[gene] + delta, low), high)
            else:
                # Perturbación flotante para el stop loss
                span = high - low
                noise = random.normalvariate(0, span * 0.15)
                mutated[gene] = round(min(max(mutated[gene] + noise, low), high), 4)
    return mutated

def genetic_algorithm_generator(df, pop_size, generations, mutation_rate):
    """
    Generador que ejecuta la evolución heurística.
    Yields información en cada generación para animar la interfaz de Streamlit.
    """
    # 1. Inicializar Población
    population = [create_random_chromosome() for _ in range(pop_size)]
    history = []
    
    for gen in range(generations):
        # 2. Evaluar Aptitud (Fitness)
        fitnesses = []
        backtest_results = []
        
        for individual in population:
            res = run_backtest(df, individual)
            fitnesses.append(res['fitness'])
            backtest_results.append(res)
            
        # Calcular estadísticas de la generación
        max_fit = max(fitnesses)
        avg_fit = sum(fitnesses) / len(fitnesses)
        best_idx = np.argmax(fitnesses)
        best_ind = population[best_idx]
        best_res = backtest_results[best_idx]
        
        history.append({
            'generation': gen + 1,
            'max_fitness': max_fit,
            'avg_fitness': avg_fit
        })
        
        # Retornamos el estado actual para la UI antes del cruce/mutación
        yield gen + 1, max_fit, avg_fit, best_ind, best_res, history
        
        # 3. Construir Nueva Generación (Selección, Crossover, Mutación)
        new_population = []
        
        # ELITISMO: Preservamos a los 2 mejores de la población actual sin alteración
        sorted_indices = np.argsort(fitnesses)[::-1]
        new_population.append(population[sorted_indices[0]])
        if pop_size > 1:
            new_population.append(population[sorted_indices[1]])
            
        # Rellenar población
        while len(new_population) < pop_size:
            # Selección por torneo
            parent1 = tournament_selection(population, fitnesses, k=3)
            parent2 = tournament_selection(population, fitnesses, k=3)
            
            # Cruce
            child1, child2 = crossover(parent1, parent2)
            
            # Mutación
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
                
        # Actualizamos población acotando al tamaño máximo
        population = new_population[:pop_size]


# ==============================================================================
# INTERFAZ GRÁFICA DE USUARIO EN STREAMLIT
# ==============================================================================

# Inicialización de estados de sesión
if 'opt_done' not in st.session_state:
    st.session_state.opt_done = False
    st.session_state.best_params = None
    st.session_state.best_res = None
    st.session_state.history = None
    st.session_state.ticker_used = ""
    st.session_state.start_date_used = None
    st.session_state.end_date_used = None

# Encabezado Principal de la App
st.markdown('<div class="header-title">🧬 Proyecto Integrador Final: Algoritmo de Trading Inmunitario</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Optimización Evolutiva y Adaptación Bio-inspirada ante Regímenes de Mercado</div>', unsafe_allow_html=True)

# ----------------- SIDEBAR (CONFIGURACIÓN) -----------------
st.sidebar.markdown("### ⚙️ Configuración de Mercado")
ticker_input = st.sidebar.text_input("Activo Financiero (Ticker Yahoo Finance)", value="AAPL", help="Ejemplos: AAPL, TSLA, BTC-USD, MSFT, EURUSD=X").strip().upper()

# Selectores de Fechas con valores predeterminados razonables
col_d1, col_d2 = st.sidebar.columns(2)
with col_d1:
    fecha_inicio = st.date_input("Fecha Inicio", datetime.now() - timedelta(days=365*2))
with col_d2:
    fecha_fin = st.date_input("Fecha Fin", datetime.now())

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧬 Hiperparámetros del Algoritmo Genético")
pop_size_input = st.sidebar.slider("Tamaño de la Población", min_value=20, max_value=100, value=50, step=5,
                                  help="Número de estrategias competidoras por generación.")
generations_input = st.sidebar.slider("Número de Generaciones", min_value=10, max_value=50, value=30, step=5,
                                    help="Iteraciones de la evolución darwiniana.")
mutation_rate_input = st.sidebar.slider("Tasa de Mutación", min_value=0.01, max_value=0.20, value=0.08, step=0.01,
                                       help="Probabilidad de alteración aleatoria en los genes de la descendencia.")

# Mensaje de ayuda en Sidebar
st.sidebar.info("""
**Regla de Decisión:**
- **Compra (Long):** Cruce alcista de MA rápida + RSI en sobreventa + Filtro de tendencia (MA lenta).
- **Salida:** Cruce bajista de MA rápida o RSI en sobrecompra o límite de Stop Loss alcanzado.
""")

# Cargar Datos
data = None
if ticker_input:
    data = fetch_ticker_data(ticker_input, fecha_inicio, fecha_fin)

# Pestañas principales
tab1, tab2, tab3, tab4 = st.tabs([
    "🧬 Evolución y Selección Natural", 
    "📊 Rendimiento del Bot Ganador", 
    "📋 Desglose del ADN Cuantitativo",
    "🌡️ Termómetro del Mercado"
])

# ==============================================================================
# TAB 1: EVOLUCIÓN Y SELECCIÓN NATURAL
# ==============================================================================
with tab1:
    # Ficha del Alumno e Imagen Institucional
    col_logo, col_info = st.columns([1, 3])
    with col_logo:
        st.image("https://raw.githubusercontent.com/FranciscoAlvarezAguilera/popo/f7eb7ba220e600b5ecd02459f46a975fda22fa9f/logo-removebg-preview.png", width=220)
    with col_info:
        st.markdown("""
        <div style="background: rgba(30, 27, 75, 0.4); padding: 18px; border-radius: 12px; border: 1px solid rgba(124, 77, 255, 0.2); margin-bottom: 20px;">
            <h4 style="margin: 0 0 10px 0; color: #00e5ff; font-family: 'Space Grotesk', sans-serif;">🎓 Identificación Académica</h4>
            <p style="margin: 4px 0; font-size: 0.95rem;"><strong>Alumno:</strong> Francisco Alvarez Aguilera</p>
            <p style="margin: 4px 0; font-size: 0.95rem;"><strong>Licenciatura:</strong> Actuaría y Ciencia de Datos</p>
            <p style="margin: 4px 0; font-size: 0.95rem;"><strong>Asignatura:</strong> Finanzas Computacionales y Algorithmic Trading</p>
            <p style="margin: 4px 0; font-size: 0.95rem;"><strong>Profesor:</strong> Dr. Oscar Valdemar De la Torre Torres</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### 🧬 Simulación Evolutiva")
    st.markdown("""
    Esta sección ejecuta un **Algoritmo Genético** para encontrar los parámetros óptimos del bot de trading. 
    Al presionar el botón de inicio, se simulará el proceso de selección natural, mostrando la mejora en tiempo real de la aptitud (*Fitness*) del sistema.
    """)
    
    if data is None:
        st.warning("⚠️ No se han cargado datos. Verifica el ticker o el rango de fechas en el panel lateral.")
    else:
        # Mostrar rango de datos descargados
        st.caption(f"Datos cargados exitosamente: {len(data)} registros diarios del **{fecha_inicio}** al **{fecha_fin}**.")
        
        btn_evolve = st.button("Iniciar Simulación Evolutiva", type="primary", width="stretch")
        
        # Contenedores dinámicos para mostrar la barra de progreso e información en tiempo real
        progress_bar = st.empty()
        status_text = st.empty()
        chart_placeholder = st.empty()
        
        # Comprobar si se ha ejecutado una optimización previa para el ticker actual
        prev_run_exists = (
            st.session_state.opt_done and 
            st.session_state.ticker_used == ticker_input and
            st.session_state.start_date_used == fecha_inicio and
            st.session_state.end_date_used == fecha_fin
        )
        
        # Ejecutar optimización al hacer clic
        if btn_evolve:
            st.session_state.opt_done = False
            
            # Validación de datos suficientes para correr las medias
            if len(data) < 200:
                st.error("❌ El conjunto de datos es demasiado pequeño (se requieren al menos 200 días históricos para calcular la MA lenta).")
            else:
                # Ejecutar el generador evolutivo
                ga_gen = genetic_algorithm_generator(
                    data, 
                    pop_size_input, 
                    generations_input, 
                    mutation_rate_input
                )
                
                # Iterar el algoritmo generación por generación para animar la pantalla
                for gen, max_fit, avg_fit, best_ind, best_res, history in ga_gen:
                    # Actualizar barra de progreso
                    progress_pct = int((gen / generations_input) * 100)
                    progress_bar.progress(progress_pct, text=f"Progreso Evolutivo: {progress_pct}%")
                    
                    # Mensaje de estado
                    status_text.markdown(f"""
                    🚀 **Generación {gen} de {generations_input}** completa. 
                    * **Aptitud Máxima (Mejor Sharpe Ajustado):** `{max_fit:.4f}`
                    * **Aptitud Promedio de la Población:** `{avg_fit:.4f}`
                    """)
                    
                    # Graficar historial de evolución en tiempo real
                    df_history = pd.DataFrame(history)
                    fig_evol = go.Figure()
                    fig_evol.add_trace(go.Scatter(
                        x=df_history['generation'], 
                        y=df_history['max_fitness'],
                        name="Aptitud Máxima (Elitista)", 
                        line=dict(color="#00e5ff", width=3)
                    ))
                    fig_evol.add_trace(go.Scatter(
                        x=df_history['generation'], 
                        y=df_history['avg_fitness'],
                        name="Aptitud Promedio", 
                        line=dict(color="#7c4dff", width=2, dash='dash')
                    ))
                    
                    fig_evol.update_layout(
                        title="Trayectoria de Convergencia del Algoritmo Genético",
                        xaxis_title="Generación",
                        yaxis_title="Función de Aptitud (Fitness)",
                        template="plotly_dark",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=40, r=40, t=80, b=45),
                        plot_bgcolor="rgba(17, 24, 39, 0.4)",
                        paper_bgcolor="rgba(17, 24, 39, 0.4)"
                    )
                    
                    chart_placeholder.plotly_chart(fig_evol, width="stretch", key=f"evol_chart_{gen}")
                
                # Guardar resultados finales en el estado de sesión para persistir cambios
                st.session_state.opt_done = True
                st.session_state.best_params = best_ind
                st.session_state.best_res = best_res
                st.session_state.history = history
                st.session_state.ticker_used = ticker_input
                st.session_state.start_date_used = fecha_inicio
                st.session_state.end_date_used = fecha_fin
                
                st.success("🎯 ¡Evolución finalizada con éxito! Los resultados óptimos han sido calculados e implementados.")
                st.balloons()
                
        # Mostrar gráfico estático si ya se había ejecutado y no se presionó de nuevo el botón
        elif prev_run_exists:
            df_history = pd.DataFrame(st.session_state.history)
            fig_evol = go.Figure()
            fig_evol.add_trace(go.Scatter(
                x=df_history['generation'], 
                y=df_history['max_fitness'],
                name="Aptitud Máxima (Elitista)", 
                line=dict(color="#00e5ff", width=3)
            ))
            fig_evol.add_trace(go.Scatter(
                x=df_history['generation'], 
                y=df_history['avg_fitness'],
                name="Aptitud Promedio", 
                line=dict(color="#7c4dff", width=2, dash='dash')
            ))
            
            fig_evol.update_layout(
                title="Trayectoria de Convergencia del Algoritmo Genético (Datos Previos)",
                xaxis_title="Generación",
                yaxis_title="Función de Aptitud (Fitness)",
                template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=80, b=45),
                plot_bgcolor="rgba(17, 24, 39, 0.4)",
                paper_bgcolor="rgba(17, 24, 39, 0.4)"
            )
            chart_placeholder.plotly_chart(fig_evol, width="stretch", key="evol_chart_static")
            status_text.info(f"💡 Se están mostrando los resultados de la simulación anterior para **{st.session_state.ticker_used}**.")
        else:
            st.info("💡 Presiona el botón de arriba para iniciar el proceso de selección y entrenamiento.")

# ==============================================================================
# TAB 2: RENDIMIENTO DEL BOT GANADOR (BACKTESTING)
# ==============================================================================
with tab2:
    st.markdown("### 📊 Análisis de Backtesting Histórico")
    
    # Validar si hay una optimización en memoria
    if not st.session_state.opt_done:
        st.warning("⚠️ Debes completar la optimización genética en la pestaña 'Evolución y Selección Natural' para analizar los rendimientos.")
    else:
        best_res = st.session_state.best_res
        best_params = st.session_state.best_params
        
        # Calcular rentabilidad de Buy & Hold
        p_first = data['Close'].iloc[0]
        p_last = data['Close'].iloc[-1]
        bh_return = (p_last / p_first - 1.0) * 100
        
        strat_return = best_res['total_return'] * 100
        sharpe = best_res['sharpe']
        max_dd = best_res['drawdown'] * 100
        
        # Construcción visual de métricas en columnas (Premium Cards)
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            strat_color = "val-positive" if strat_return >= 0 else "val-negative"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-lbl">Retorno de Estrategia</div>
                <div class="metric-val {strat_color}">{strat_return:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_m2:
            bh_color = "val-positive" if bh_return >= 0 else "val-negative"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-lbl">Retorno Buy & Hold</div>
                <div class="metric-val {bh_color}">{bh_return:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_m3:
            sharpe_color = "val-neutral" if sharpe > 1 else ("val-warning" if sharpe > 0 else "val-negative")
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-lbl">Ratio de Sharpe final</div>
                <div class="metric-val {sharpe_color}">{sharpe:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_m4:
            # El Drawdown siempre es negativo en porcentaje de caída, pero lo expresamos positivo
            dd_color = "val-positive" if max_dd < 10 else ("val-warning" if max_dd < 25 else "val-negative")
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-lbl">Max Drawdown</div>
                <div class="metric-val {dd_color}">{max_dd:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # 1. Gráfico de Precio con Señales de Compra y Venta (y RSI en subgráfico)
        st.subheader("📈 Señales del Bot y Oscilador RSI")
        
        from plotly.subplots import make_subplots
        
        fig_signals = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.08, 
            row_heights=[0.65, 0.35],
            subplot_titles=("Precio de Cierre ($)", "Oscilador RSI")
        )
        
        # Línea de precios
        fig_signals.add_trace(go.Scatter(
            x=data.index, y=data['Close'],
            name="Precio de Cierre ($)", 
            line=dict(color="#ffffff", width=1.5)
        ), row=1, col=1)
        
        # Medias móviles encontradas por el cromosoma
        p_fast = int(best_params['periodo_MA_rapida'])
        p_slow = int(best_params['periodo_MA_lenta'])
        ma_fast_plot = data['Close'].rolling(window=p_fast).mean()
        ma_slow_plot = data['Close'].rolling(window=p_slow).mean()
        
        fig_signals.add_trace(go.Scatter(
            x=data.index, y=ma_fast_plot,
            name=f"MA Rápida ({p_fast})", 
            line=dict(color="#00e5ff", width=1, dash="dot")
        ), row=1, col=1)
        
        fig_signals.add_trace(go.Scatter(
            x=data.index, y=ma_slow_plot,
            name=f"MA Lenta ({p_slow})", 
            line=dict(color="#7c4dff", width=1, dash="dot")
        ), row=1, col=1)
        
        # Señales de compra
        buy_sig = best_res['buy_signals']
        if len(buy_sig) > 0:
            buy_dates, buy_prices = zip(*buy_sig)
            fig_signals.add_trace(go.Scatter(
                x=buy_dates, y=buy_prices,
                mode="markers", name="Compra (Llegada de Antígeno)",
                marker=dict(symbol="triangle-up", size=13, color="#10b981", line=dict(color="#064e3b", width=1.5))
            ), row=1, col=1)
            
        # Señales de venta
        sell_sig = best_res['sell_signals']
        if len(sell_sig) > 0:
            sell_dates, sell_prices = zip(*sell_sig)
            fig_signals.add_trace(go.Scatter(
                x=sell_dates, y=sell_prices,
                mode="markers", name="Venta (Respuesta Inmunitaria)",
                marker=dict(symbol="triangle-down", size=13, color="#ef4444", line=dict(color="#7f1d1d", width=1.5))
            ), row=1, col=1)
            
        # RSI en la subgráfica inferior
        rsi_plot = best_res['rsi']
        fig_signals.add_trace(go.Scatter(
            x=data.index, y=rsi_plot,
            name=f"RSI ({int(best_params['periodo_RSI'])})",
            line=dict(color="#fbbf24", width=1.5)
        ), row=2, col=1)
        
        # Líneas de referencia horizontales en 30 y 70
        fig_signals.add_hline(y=70, line_dash="dash", line_color="#ef4444", line_width=1, row=2, col=1)
        fig_signals.add_hline(y=30, line_dash="dash", line_color="#10b981", line_width=1, row=2, col=1)
        
        # Sombrear rango 30-70
        fig_signals.add_hrect(y0=30, y1=70, fillcolor="rgba(156, 163, 175, 0.12)", line_width=0, row=2, col=1)
        
        fig_signals.update_layout(
            template="plotly_dark",
            height=650,
            margin=dict(l=40, r=40, t=50, b=45),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            plot_bgcolor="rgba(17, 24, 39, 0.4)",
            paper_bgcolor="rgba(17, 24, 39, 0.4)"
        )
        
        fig_signals.update_yaxes(title_text="Precio ($)", row=1, col=1)
        fig_signals.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        fig_signals.update_xaxes(title_text="Fecha", row=2, col=1)
        
        st.plotly_chart(fig_signals, width="stretch", key="signals_chart")
        
        st.markdown("---")
        
        # 2. Gráfico comparativo de la Curva de Capital (Equity Curve)
        st.subheader("⚖️ Curva de Crecimiento del Patrimonio (Base 100)")
        
        fig_equity = go.Figure()
        
        # Multiplicamos por 100 para simular una cuenta inicial de $100
        equity_strat = best_res['equity'] * 100
        equity_bh = (data['Close'].values / data['Close'].iloc[0]) * 100
        
        fig_equity.add_trace(go.Scatter(
            x=data.index, y=equity_strat,
            name="Estrategia Genética Inmunitaria",
            line=dict(color="#00e5ff", width=2.5)
        ))
        
        fig_equity.add_trace(go.Scatter(
            x=data.index, y=equity_bh,
            name="Comprar y Mantener (Buy & Hold)",
            line=dict(color="#6b7280", width=1.5, dash="dash")
        ))
        
        fig_equity.update_layout(
            template="plotly_dark",
            xaxis_title="Fecha",
            yaxis_title="Valor de la Cuenta ($)",
            margin=dict(l=40, r=40, t=40, b=45),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            plot_bgcolor="rgba(17, 24, 39, 0.4)",
            paper_bgcolor="rgba(17, 24, 39, 0.4)"
        )
        
        st.plotly_chart(fig_equity, width="stretch", key="equity_chart")
        
        # Tabla detallada de transacciones
        st.markdown("### 📝 Historial de Transacciones Ejecutadas")
        if len(best_res['trades']) == 0:
            st.info("No se registraron transacciones en este periodo de backtesting.")
        else:
            trades_df = pd.DataFrame(best_res['trades'])
            # Renombrar columnas para el usuario
            trades_df.columns = ['Fecha Entrada', 'Fecha Salida', 'Precio Entrada', 'Precio Salida', 'Retorno (%)', 'Trigger Stop Loss']
            trades_df['Retorno (%)'] = trades_df['Retorno (%)'] * 100
            
            # Formatear la salida
            trades_df['Precio Entrada'] = trades_df['Precio Entrada'].apply(lambda x: f"${x:,.2f}")
            trades_df['Precio Salida'] = trades_df['Precio Salida'].apply(lambda x: f"${x:,.2f}")
            trades_df['Retorno (%)'] = trades_df['Retorno (%)'].apply(lambda x: f"{x:.2f}%")
            
            st.dataframe(trades_df, width="stretch", hide_index=True)


# ==============================================================================
# TAB 3: DESGLOSE DEL ADN CUANTITATIVO Y CONCLUSIONES
# ==============================================================================
with tab3:
    st.markdown("### 📋 Genoma Optimizado y Conclusiones Académicas")
    
    if not st.session_state.opt_done:
        st.warning("⚠️ Debes completar la optimización genética en la pestaña 'Evolución y Selección Natural' para analizar el ADN del bot.")
    else:
        best_params = st.session_state.best_params
        best_res = st.session_state.best_res
        
        # Diseño de columnas: una para el genoma en sí, y otra para las conclusiones
        col_dna, col_insights = st.columns([1, 1])
        
        with col_dna:
            st.markdown("#### 🧬 Mapa Genético del Individuo Alfa")
            st.markdown("Los siguientes parámetros fueron refinados darwinianamente a través del algoritmo inmunitario:")
            
            # Tabla estilizada HTML para presentar el cromosoma
            html_table = f"""
            <table class="styled-table">
                <thead>
                    <tr>
                        <th>Parámetro (Gen)</th>
                        <th>Valor Encontrado</th>
                        <th>Rango de Búsqueda</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Periodo Media Rápida</strong></td>
                        <td><span class="dna-badge">{int(best_params['periodo_MA_rapida'])}</span> días</td>
                        <td>[5, 50]</td>
                    </tr>
                    <tr>
                        <td><strong>Periodo Media Lenta</strong></td>
                        <td><span class="dna-badge">{int(best_params['periodo_MA_lenta'])}</span> días</td>
                        <td>[51, 200]</td>
                    </tr>
                    <tr>
                        <td><strong>Umbral RSI Sobrecompra</strong></td>
                        <td><span class="dna-badge">{int(best_params['limite_RSI_sobrecompra'])}</span></td>
                        <td>[65, 85]</td>
                    </tr>
                    <tr>
                        <td><strong>Umbral RSI Sobreventa</strong></td>
                        <td><span class="dna-badge">{int(best_params['limite_RSI_sobreventa'])}</span></td>
                        <td>[15, 35]</td>
                    </tr>
                    <tr>
                        <td><strong>Periodo RSI</strong></td>
                        <td><span class="dna-badge">{int(best_params['periodo_RSI'])}</span> días</td>
                        <td>[7, 21]</td>
                    </tr>
                    <tr>
                        <td><strong>Ajuste de Stop Loss</strong></td>
                        <td><span class="dna-badge">{best_params['stop_loss'] * 100:.2f}%</span></td>
                        <td>[1.0%, 5.0%]</td>
                    </tr>
                </tbody>
            </table>
            """
            st.markdown(html_table, unsafe_allow_html=True)
            
        with col_insights:
            st.markdown("#### 📝 Reporte de Eficiencia de la Estrategia")
            st.markdown("A continuación, se presenta un análisis automatizado del desempeño de la estrategia bio-inspirada en base a las métricas del backtest:")
            
            # Cálculo de variables de comparación
            p_first = data['Close'].iloc[0]
            p_last = data['Close'].iloc[-1]
            bh_ret_val = (p_last / p_first - 1.0)
            strat_ret_val = best_res['total_return']
            
            beat_market = strat_ret_val > bh_ret_val
            drawdown_saved = best_res['drawdown'] < 0.25 # consideramos 25% como umbral de control
            
            # Tarjeta de Conclusión sobre Rendimiento Absoluto
            if beat_market:
                st.success("🏆 **Superación de Mercado:** La estrategia bio-inspirada logró batir el rendimiento acumulado de Buy & Hold, demostrando la viabilidad de la optimización genética en este intervalo temporal.")
            else:
                st.warning("⚠️ **Rendimiento Absoluto:** El bot no logró superar el retorno bruto del mercado tradicional (Buy & Hold). Sin embargo, esto es común en periodos de fuertes tendencias alcistas donde las estrategias activas suelen incurrir en costes de oportunidad al estar fuera de mercado.")
                
            # Tarjeta de Conclusión sobre Gestión de Riesgos (Inmunidad)
            if best_res['drawdown'] < 0.15:
                st.info(f"🛡️ **Protección Inmunitaria Fuerte:** El Maximum Drawdown se mantuvo en niveles sumamente conservadores (`{best_res['drawdown']*100:.2f}%`), lo que comprueba la efectividad del stop loss de `{best_params['stop_loss']*100:.2f}%` como mecanismo de apoptosis del capital ante eventos de cola.")
            else:
                st.error(f"⚠️ **Vulnerabilidad Sistémica:** El drawdown experimentado fue de `{best_res['drawdown']*100:.2f}%`. Aunque el algoritmo buscó maximizar la aptitud, las fluctuaciones extremas del activo en el régimen elegido afectaron la consistencia protectora.")
                
            # Estadísticas de Trade
            trades_list = best_res['trades']
            num_trades = len(trades_list)
            
            if num_trades > 0:
                winning_trades = sum(1 for t in trades_list if t['return'] > 0)
                win_ratio = (winning_trades / num_trades) * 100
                st.markdown(f"""
                **Estadísticas de Operación:**
                * **Número Total de Transacciones:** `{num_trades}`
                * **Ratio de Acierto (Win Rate):** `{win_ratio:.2f}%` (`{winning_trades}` ganadoras de `{num_trades}`)
                * **Ratio de Sharpe (Rendimiento/Volatilidad):** `{best_res['sharpe']:.3f}`
                """)
            else:
                st.markdown("""
                **Estadísticas de Operación:**
                * *No se registraron transacciones válidas debido a rigidez de condiciones genéticas.*
                """)
                
            st.markdown("""
            > **Nota Académica:** En Computación Bio-inspirada, la analogía del sistema inmune reside en la habilidad de la estrategia para proteger la cuenta contra 'patógenos' de mercado (caídas abruptas de precio) mediante la desconexión rápida (Stop Loss) y filtros de tendencia (MA lenta), permitiendo una supervivencia sostenida en el tiempo.
            """)

# ==============================================================================
# TAB 4: TERMÓMETRO DEL MERCADO
# ==============================================================================
with tab4:
    st.markdown("### 🌡️ Termómetro de Convicción del Mercado")
    st.markdown("""
    Esta pestaña analiza los datos en tiempo real de la sesión más reciente para estimar la dirección y convicción del mercado en base a las reglas de los indicadores optimizados.
    """)
    
    if not st.session_state.opt_done:
        st.warning("⚠️ Debes completar la simulación evolutiva en la pestaña 'Evolución y Selección Natural' para habilitar el Termómetro de Convicción.")
    else:
        best_params = st.session_state.best_params
        
        # Recalcular indicadores para los últimos valores
        p_fast = int(best_params['periodo_MA_rapida'])
        p_slow = int(best_params['periodo_MA_lenta'])
        p_rsi = int(best_params['periodo_RSI'])
        
        ma_fast = data['Close'].rolling(window=p_fast).mean()
        ma_slow = data['Close'].rolling(window=p_slow).mean()
        rsi = calculate_rsi(data['Close'], p_rsi)
        
        # Obtener los datos del día más reciente (t = n-1)
        price_t = data['Close'].iloc[-1]
        ma_fast_t = ma_fast.iloc[-1]
        ma_slow_t = ma_slow.iloc[-1]
        rsi_t = rsi.iloc[-1]
        fecha_t = data.index[-1].strftime('%Y-%m-%d')
        
        # 1. Distancia porcentual a las medias móviles (Normalizada)
        # Asumimos que 5% de distancia de la MA rápida es el límite para convicción total a corto plazo
        d_fast = (price_t - ma_fast_t) / ma_fast_t
        s_fast = min(max(d_fast / 0.05, -1.0), 1.0)
        
        # Asumimos que 15% de distancia de la MA lenta es el límite de convicción a largo plazo
        d_slow = (price_t - ma_slow_t) / ma_slow_t
        s_slow = min(max(d_slow / 0.15, -1.0), 1.0)
        
        # Puntuación combinada de tendencia
        s_trend = 0.5 * s_fast + 0.5 * s_slow
        
        # 2. Puntuación del RSI respecto a los umbrales optimizados
        os_limit = best_params['limite_RSI_sobreventa']
        ob_limit = best_params['limite_RSI_sobrecompra']
        mid_rsi = (os_limit + ob_limit) / 2.0
        
        # Si el RSI está en la mitad exacta, la convicción es neutral (0)
        # Si baja hacia sobreventa, aumenta convicción compradora (tiende a +1)
        # Si sube hacia sobrecompra, aumenta convicción vendedora (tiende a -1)
        s_rsi = (mid_rsi - rsi_t) / (mid_rsi - os_limit)
        s_rsi = min(max(s_rsi, -1.0), 1.0)
        
        # 3. Puntuación final (50% tendencia de medias, 50% momentum de RSI)
        conviction = 0.5 * s_trend + 0.5 * s_rsi
        conviction_pct = conviction * 100
        
        # Clasificación cualitativa
        if conviction_pct >= 75:
            estado = "Compra Fuerte"
            estilo_color = "val-positive"
            explicacion = "El precio se encuentra sólidamente por encima de sus soportes dinámicos y el RSI muestra niveles de acumulación de alta probabilidad."
        elif conviction_pct >= 25:
            estado = "Compra Moderada"
            estilo_color = "val-neutral"
            explicacion = "El mercado muestra un sesgo alcista moderado. El precio mantiene tendencia positiva con margen para seguir subiendo."
        elif conviction_pct > -25:
            estado = "Neutral"
            estilo_color = "val-warning"
            explicacion = "Las fuerzas alcistas y bajistas están en equilibrio dinámico. Los osciladores están centrados y el precio consolida."
        elif conviction_pct > -75:
            estado = "Venta Moderada"
            estilo_color = "val-warning"
            explicacion = "Se observa un sesgo correctivo. El precio opera bajo resistencias y el momentum de compra es débil."
        else:
            estado = "Venta Fuerte"
            estilo_color = "val-negative"
            explicacion = "Alerta de sobrecalentamiento o ruptura bajista severa. El precio cotiza lejos de sus promedios y el RSI muestra sobrecompra extrema."
            
        col_gauge, col_details = st.columns([1.2, 1])
        
        with col_gauge:
            # Crear velocímetro interactivo de Plotly
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = conviction_pct,
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [-100, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#ffffff", 'thickness': 0.25},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 1,
                    'bordercolor': "rgba(255, 255, 255, 0.2)",
                    'steps': [
                        {'range': [-100, -75], 'color': 'rgba(239, 68, 68, 0.75)'}, # Rojo
                        {'range': [-75, -25], 'color': 'rgba(249, 115, 22, 0.75)'}, # Naranja
                        {'range': [-25, 25], 'color': 'rgba(234, 179, 8, 0.75)'},   # Amarillo
                        {'range': [25, 75], 'color': 'rgba(52, 211, 153, 0.75)'},   # Verde Claro
                        {'range': [75, 100], 'color': 'rgba(16, 185, 129, 0.75)'}   # Verde Oscuro
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.8,
                        'value': conviction_pct
                      }
                  }
              ))
            fig_gauge.update_layout(
                template="plotly_dark",
                height=380,
                margin=dict(l=40, r=40, t=50, b=30),
                plot_bgcolor="rgba(17, 24, 39, 0.4)",
                paper_bgcolor="rgba(17, 24, 39, 0.4)"
            )
            st.plotly_chart(fig_gauge, width="stretch", key="gauge_conviction")
            
        with col_details:
            st.markdown(f"#### 🔍 Análisis del Cierre ({fecha_t})")
            
            st.markdown(f"""
            <div class="metric-container" style="border-top: 3px solid #7c4dff;">
                <div class="metric-lbl">Diagnóstico de Tendencia</div>
                <div class="metric-val {estilo_color}">{estado}</div>
                <p style="margin-top:10px; font-size:0.92rem; color:#9ca3af;">Convicción Calculada: {conviction_pct:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**Justificación Cuantitativa:**")
            st.write(explicacion)
            
            # Pequeño desglose numérico
            st.markdown(f"""
            - **Precio de Cierre:** `${price_t:,.2f}`
            - **Media Rápida ({p_fast}d):** `${ma_fast_t:,.2f}` (Diferencia: `{d_fast * 100:.2f}%`)
            - **Media Lenta ({p_slow}d):** `${ma_slow_t:,.2f}` (Diferencia: `{d_slow * 100:.2f}%`)
            - **Valor RSI ({p_rsi}d):** `{rsi_t:.1f}` (Límites optimizados: `{os_limit}` - `{ob_limit}`)
            """)
