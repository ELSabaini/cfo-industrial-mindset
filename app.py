import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="CFO Industrial Mindset", layout="wide", initial_sidebar_state="expanded")

# --- ESTILIZAÇÃO CUSTOMIZADA (RESPONSIVA + ADAPTÁVEL AO TEMA) ---
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: clamp(4px, 1vw, 12px);
        flex-wrap: wrap;
    }

    .stTabs [data-baseweb="tab"] {
        min-height: clamp(44px, 5vw, 64px);
        padding: clamp(8px, 1vw, 14px) clamp(10px, 1.6vw, 24px);
        border-radius: clamp(6px, 0.8vw, 10px) clamp(6px, 0.8vw, 10px) 0 0;
        font-weight: 600;
        font-size: clamp(11px, 1.05vw, 18px);
        line-height: 1.2;
        white-space: normal;
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
        flex: 1 1 auto;

        background: var(--secondary-background-color);
        color: var(--text-color) !important;
        border: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
        transition: all 0.22s ease-in-out;
        transform: translateY(0) scale(1);
    }

    .stTabs [data-baseweb="tab"] p {
        margin: 0;
        font-size: inherit;
        line-height: 1.2;
        text-align: center;
        word-break: break-word;
        color: inherit !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: color-mix(in srgb, var(--secondary-background-color) 88%, var(--primary-color) 12%);
        color: var(--text-color) !important;
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 6px 14px rgba(0,0,0,0.12);
    }

    .stTabs [aria-selected="true"] {
        background: color-mix(in srgb, var(--secondary-background-color) 82%, var(--primary-color) 18%) !important;
        color: var(--text-color) !important;
        transform: translateY(-2px) scale(1.02);
        border: 1px solid color-mix(in srgb, var(--primary-color) 55%, transparent) !important;
        box-shadow: 0 8px 18px rgba(0,0,0,0.10);
    }

    .stTabs [aria-selected="true"] p {
        color: var(--text-color) !important;
        font-weight: 700;
    }

    .stTabs [aria-selected="true"]::after {
        content: "";
        display: block;
        height: clamp(2px, 0.3vw, 4px);
        background: var(--primary-color);
        margin-top: 4px;
        border-radius: 999px;
        animation: pulsebar 1.2s ease-in-out infinite;
    }

    @keyframes pulsebar {
        0% { opacity: 0.65; transform: scaleX(0.96); }
        50% { opacity: 1; transform: scaleX(1); }
        100% { opacity: 0.65; transform: scaleX(0.96); }
    }

    @media (max-width: 900px) {
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
        }

        .stTabs [data-baseweb="tab"] {
            min-height: 46px;
            padding: 8px 10px;
            font-size: 12px;
        }
    }

    @media (max-width: 640px) {
        .stTabs [data-baseweb="tab"] {
            min-height: 42px;
            padding: 7px 8px;
            font-size: 11px;
            border-radius: 6px 6px 0 0;
        }
    }

    .main-header {
        font-size: 45px;
        font-weight: bold;
        color: var(--text-color);
        margin-bottom: 0px;
    }

    .sub-header {
        font-size: 18px;
        color: color-mix(in srgb, var(--text-color) 65%, transparent);
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- RESTO DO CÓDIGO (INALTERADO) ---
# (mantive exatamente igual ao que você já estava usando)

@st.cache_data
def gerar_dados(volatilidade_global=1.0, exposicao_maxima=0.6):
    np.random.seed(42)
    meses = pd.date_range('2022-01-01', periods=48, freq='MS')
    linhas_config = {
        'SaaS B2B': {'vol_base': 850, 'preco': 420, 'cvu': 85, 'fixo': 90000, 'vol_risco': 0.05, 'dso': 35, 'dpo': 20, 'est': 2},
        'Serviços': {'vol_base': 620, 'preco': 510, 'cvu': 240, 'fixo': 70000, 'vol_risco': 0.09, 'dso': 42, 'dpo': 25, 'est': 1},
        'Indústria Equip.': {'vol_base': 180, 'preco': 4800, 'cvu': 2950, 'fixo': 140000, 'vol_risco': 0.12, 'dso': 58, 'dpo': 36, 'est': 48},
        'Varejo Premium': {'vol_base': 1600, 'preco': 380, 'cvu': 235, 'fixo': 85000, 'vol_risco': 0.15, 'dso': 12, 'dpo': 28, 'est': 32},
        'Distribuição': {'vol_base': 2400, 'preco': 150, 'cvu': 118, 'fixo': 65000, 'vol_risco': 0.07, 'dso': 30, 'dpo': 40, 'est': 24}
    }

    fator_vol = 1 + ((volatilidade_global - 1.0) * 3.2)
    fator_exp = 1 + ((exposicao_maxima - 0.6) * 4.0)

    registros = []
    for linha, p in linhas_config.items():
        for i, data in enumerate(meses):
            risco = max(p['vol_risco'] * fator_vol * fator_exp, 0.01)
            choque = np.random.normal(0, risco)
            sazonalidade = 0.06 * np.sin(i / 2.8)
            estresse = np.random.normal(0, risco * 0.8)

            volume = p['vol_base'] * (1 + 0.008 * i) * (1 + sazonalidade + choque + 0.7 * estresse)
            volume = max(volume, p['vol_base'] * 0.15)

            preco = p['preco'] * (1 + np.random.normal(0, risco * 0.25))
            cvu = p['cvu'] * (1 + np.random.normal(0, risco * 0.35))

            receita = max(volume * preco, 0)
            ebitda = (receita - (volume * cvu)) - p['fixo']

            cg = (
                (receita * (p['dso'] * (1 + risco * 0.35)) / 30)
                - (volume * cvu * (p['dpo'] * (1 - risco * 0.18)) / 30)
            )

            registros.append([data, linha, receita, ebitda, cg, p['dso'], p['dpo'], p['est']])

    return pd.DataFrame(registros, columns=['data', 'linha', 'receita', 'ebitda', 'capital_giro', 'dso', 'dpo', 'estoque'])
