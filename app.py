
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="CFO Industrial Mindset", layout="wide", initial_sidebar_state="expanded")

# --- ESTILIZAÇÃO CUSTOMIZADA ---
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-left: 20px;
        padding-right: 20px;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0e1117 !important;
        color: white !important;
    }
    .main-header {
        font-size: 45px;
        font-weight: bold;
        color: #0e1117;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 18px;
        color: #555;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LÓGICA DE GERAÇÃO DE DADOS ---
@st.cache_data
def gerar_dados(volatilidade_global=1.0):
    np.random.seed(42)
    meses = pd.date_range('2022-01-01', periods=48, freq='MS')
    linhas_config = {
        'SaaS B2B': {'vol_base': 850, 'preco': 420, 'cvu': 85, 'fixo': 90000, 'vol_risco': 0.05, 'dso': 35, 'dpo': 20, 'est': 2},
        'Serviços': {'vol_base': 620, 'preco': 510, 'cvu': 240, 'fixo': 70000, 'vol_risco': 0.09, 'dso': 42, 'dpo': 25, 'est': 1},
        'Indústria Equip.': {'vol_base': 180, 'preco': 4800, 'cvu': 2950, 'fixo': 140000, 'vol_risco': 0.12, 'dso': 58, 'dpo': 36, 'est': 48},
        'Varejo Premium': {'vol_base': 1600, 'preco': 380, 'cvu': 235, 'fixo': 85000, 'vol_risco': 0.15, 'dso': 12, 'dpo': 28, 'est': 32},
        'Distribuição': {'vol_base': 2400, 'preco': 150, 'cvu': 118, 'fixo': 65000, 'vol_risco': 0.07, 'dso': 30, 'dpo': 40, 'est': 24}
    }
    
    registros = []
    for linha, p in linhas_config.items():
        for i, data in enumerate(meses):
            risco = p['vol_risco'] * volatilidade_global
            ruido = np.random.normal(0, risco)
            volume = p['vol_base'] * (1 + 0.008*i) * (1 + ruido)
            receita = max(volume * p['preco'], 0)
            ebitda = (receita - (volume * p['cvu'])) - p['fixo']
            cg = (receita * p['dso']/30) - (volume * p['cvu'] * p['dpo']/30)
            registros.append([data, linha, receita, ebitda, cg, p['dso'], p['dpo'], p['est']])
            
    return pd.DataFrame(registros, columns=['data', 'linha', 'receita', 'ebitda', 'capital_giro', 'dso', 'dpo', 'estoque'])

def portfolio_stats(weights, exp_returns, cov_matrix):
    ret = np.dot(weights, exp_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = ret / vol if vol != 0 else 0
    return ret, vol, sharpe

# --- SIDEBAR: SIMULAÇÃO EM TEMPO REAL ---
st.sidebar.image("https://img.icons8.com/fluency/96/dashboard.png", width=80)
st.sidebar.header("🕹️ Centro de Comando")
st.sidebar.markdown("---")

volat_slider = st.sidebar.slider("Volatilidade do Mercado (%)", 50, 200, 100) / 100
exposicao_max = st.sidebar.slider("Exposição Máxima por Linha (%)", 20, 100, 60) / 100
st.sidebar.markdown("---")
st.sidebar.info("Ajuste a volatilidade para ver como o risco individual impacta a Fronteira Eficiente.")

# --- PREPARAÇÃO DOS DADOS ---
df = gerar_dados(volat_slider)
resumo = df.groupby('linha').agg({
    'receita': 'mean',
    'ebitda': ['mean', 'std'],
    'capital_giro': 'mean'
}).reset_index()
resumo.columns = ['linha', 'receita_media', 'ebitda_medio', 'risco_ebitda', 'cg_medio']
resumo['sharpe_corp'] = resumo['ebitda_medio'] / resumo['risco_ebitda']

# Cálculos de Otimização
pivot_ebitda = df.pivot_table(index='data', columns='linha', values='ebitda')
retornos = pivot_ebitda.mean().values
cov = pivot_ebitda.cov().values
nomes = pivot_ebitda.columns.tolist()

rec_pesos = df.groupby('linha')['receita'].mean().reindex(nomes)
pesos_atuais = (rec_pesos / rec_pesos.sum()).values

def obj_func(w): return -portfolio_stats(w, retornos, cov)[2]
opt_res = minimize(obj_func, np.repeat(1/len(nomes), len(nomes)), 
                  bounds=[(0, exposicao_max)]*len(nomes), 
                  constraints={'type':'eq', 'fun': lambda w: np.sum(w)-1})
pesos_otim = opt_res.x

# --- HEADER PRINCIPAL ---
st.markdown('<p class="main-header">📊 CFO Industrial Mindset</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Transformando o FP&A Tradicional em uma Engine de Otimização de Portfólio Operacional</p>', unsafe_allow_html=True)

# --- NAVEGAÇÃO POR ABAS COM INDICADORES VISUAIS ---
tab_labels = [
    "📌 1. O CONCEITO", 
    "🔍 2. DIAGNÓSTICO DE RISCO", 
    "🎯 3. OTIMIZAÇÃO DE MIX", 
    "🌡️ 4. HEATMAP GERENCIAL",
    "📋 5. DADOS E EXPORTAÇÃO"
]
tabs = st.tabs(tab_labels)

# --- TAB 1: O CONCEITO ---
with tabs[0]:
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.markdown("""
        ### O Problema: A 'Miopia' da Margem
        Muitas empresas focam apenas em **Margem EBITDA**, ignorando a **Volatilidade** dessa margem. 
        Uma linha de negócio com 40% de margem mas 30% de volatilidade pode destruir o caixa da empresa em meses de baixa.
        
        ### A Solução: Industrial Mindset
        Nesta abordagem, tratamos cada Unidade de Negócio como um **Ativo Financeiro**:
        - **Retorno:** EBITDA Médio Mensal.
        - **Risco:** Desvio Padrão do EBITDA (Incerteza).
        - **Peso:** Alocação comercial e de capacidade.
        """)
        st.success("💡 **Objetivo:** Encontrar o mix de vendas que maximiza o lucro para um nível de estresse aceitável.")
    
    with col2:
        consolidado = df.groupby('data')[['receita', 'ebitda']].sum().reset_index()
        fig = px.area(consolidado, x='data', y=['receita', 'ebitda'], 
                     title="Fluxo Consolidado da Empresa", template='plotly_white',
                     color_discrete_sequence=['#A6CEE3', '#1F78B4'])
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: DIAGNÓSTICO ---
with tabs[1]:
    st.header("Análise de Risco x Retorno")
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        fig_scatter = px.scatter(resumo, x='risco_ebitda', y='ebitda_medio', size='receita_media',
                                color='sharpe_corp', text='linha', hover_name='linha',
                                title="Eficiência Operacional por Linha de Negócio",
                                labels={'risco_ebitda': 'Incerteza (Risco)', 'ebitda_medio': 'EBITDA Médio (Retorno)'},
                                color_continuous_scale='Viridis', height=500)
        fig_scatter.update_traces(textposition='top center')
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col_b:
        st.markdown("### Ranking de Eficiência (Sharpe)")
        st.write("O 'Sharpe Corporativo' indica quanto lucro cada linha gera por unidade de risco assumida.")
        st.dataframe(resumo[['linha', 'sharpe_corp']].sort_values('sharpe_corp', ascending=False), 
                     hide_index=True, use_container_width=True)
        st.info("Linhas no topo são 'âncoras de estabilidade'. Linhas na base são 'apostas voláteis'.")

# --- TAB 3: OTIMIZAÇÃO ---
with tabs[2]:
    st.header("A Fronteira Eficiente")
    col_c, col_d = st.columns(2)
    
    with col_c:
        # Simulação Monte Carlo
        num_portfolios = 2000
        results = np.zeros((3 + len(nomes), num_portfolios))
        for i in range(num_portfolios):
            weights = np.random.random(len(nomes))
            weights /= np.sum(weights)
            ret, vol, sharpe = portfolio_stats(weights, retornos, cov)
            results[0,i], results[1,i], results[2,i] = ret, vol, sharpe
        
        fig_front = px.scatter(x=results[1,:], y=results[0,:], color=results[2,:],
                              title="Curva de Possibilidades (Markowitz)",
                              labels={'x': 'Risco do Portfólio', 'y': 'EBITDA do Portfólio', 'color': 'Sharpe'},
                              opacity=0.4, color_continuous_scale='Bluered')
        
        # Marcar Atual e Otimizado
        r_at, v_at, _ = portfolio_stats(pesos_atuais, retornos, cov)
        r_ot, v_ot, _ = portfolio_stats(pesos_otim, retornos, cov)
        fig_front.add_trace(go.Scatter(x=[v_at], y=[r_at], name="Cenário Atual", mode='markers+text', 
                                     text=["ATUAL"], textposition="top center",
                                     marker=dict(size=15, color='black', symbol='diamond')))
        fig_front.add_trace(go.Scatter(x=[v_ot], y=[r_ot], name="Cenário Otimizado", mode='markers+text', 
                                     text=["OTIMIZADO"], textposition="top center",
                                     marker=dict(size=18, color='gold', symbol='star')))
        st.plotly_chart(fig_front, use_container_width=True)
    
    with col_d:
        mix_data = pd.DataFrame({'Linha': nomes, 'Atual': pesos_atuais, 'Otimizado': pesos_otim})
        fig_bar = px.bar(mix_data.melt(id_vars='Linha'), x='Linha', y='value', color='variable',
                        barmode='group', title="Shift Recomendado no Mix Comercial",
                        labels={'value': '% no Mix', 'variable': 'Cenário'},
                        template='plotly_white')
        st.plotly_chart(fig_bar, use_container_width=True)
        
        delta_ret = (r_ot/r_at - 1) * 100
        delta_risk = (v_ot/v_at - 1) * 100
        st.metric("Melhoria no EBITDA Médio", f"{r_ot:,.0f}", f"{delta_ret:+.1f}%")
        st.metric("Redução de Risco (Volatilidade)", f"{v_ot:,.0f}", f"{delta_risk:.1f}%", delta_color="inverse")

# --- TAB 4: HEATMAP ---
with tabs[3]:
    st.header("Heatmap Gerencial de Alinhamento")
    st.write("Visão normalizada comparando Margem, Risco e Capital de Giro.")
    
    heat_df = resumo.set_index('linha')[['ebitda_medio', 'risco_ebitda', 'cg_medio', 'sharpe_corp']]
    # Normalização 0-1
    heat_norm = (heat_df - heat_df.min()) / (heat_df.max() - heat_df.min())
    
    fig_heat = px.imshow(heat_norm, text_auto='.2f', color_continuous_scale='RdYlGn',
                        labels=dict(x="Métrica", y="Linha de Negócio", color="Score"),
                        aspect="auto")
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown("""
    **Como ler:** 
    - Cores **verdes** em EBITDA e Sharpe com **vermelho** em Risco é o cenário ideal.
    - Se o **Capital de Giro** estiver muito alto (verde), a linha consome muito caixa operacional.
    """)

# --- TAB 5: DADOS ---
with tabs[4]:
    st.header("Exploração de Dados Brutos")
    st.write("Abaixo você pode filtrar e baixar a base utilizada na simulação.")
    st.dataframe(df, use_container_width=True)
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Baixar Base Completa (CSV)", data=csv, file_name="cfo_industrial_mindset.csv", mime="text/csv")

# --- FOOTER ---
st.markdown("---")
st.markdown("🚀 **CFO Industrial Mindset** | Desenvolvido para transformar dados em estratégia de capital.")
