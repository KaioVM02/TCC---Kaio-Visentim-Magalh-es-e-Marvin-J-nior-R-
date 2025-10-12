import streamlit as st
import pandas as pd
from main import RecomendadorGPU

st.set_page_config(page_title="Recomendador de GPUs", layout="wide")

@st.cache_resource
def carregar_recomendador(nome_arquivo):
    print("Carregando e treinando o modelo de IA... (só na primeira vez)")
    recomendador = RecomendadorGPU(nome_arquivo)
    return recomendador

st.title("Sistema de Análise e Recomendação de GPUs com IA 🧠")
st.markdown("Use as abas abaixo para obter uma recomendação personalizada ou para explorar as GPUs por perfil.")

nome_arquivo_csv = "gpu_dataset_simplificado.csv"
recomendador = carregar_recomendador(nome_arquivo_csv)

tab1, tab2, tab3 = st.tabs(["Recomendação Personalizada", "Explorar GPUs por Perfil", "Análise do Mercado"])

with tab1:
    st.header("Encontre a GPU ideal com filtros personalizados")
    
    with st.sidebar:
        st.header("Defina seus Critérios")
        preco_max = st.slider("Orçamento Máximo (USD)", min_value=100, max_value=3000, value=700, step=50)
        vram_min = st.select_slider("VRAM Mínima (GB)", options=[2, 4, 6, 8, 10, 12, 16, 20, 24], value=8)
        benchmark_score_min = st.number_input("Pontuação Mínima de Benchmark (0-100)", min_value=0, max_value=100, value=30)
        tdp_max = st.number_input("Consumo Máximo (TDP em Watts)", min_value=0, max_value=500, step=10, value=250)
        ano_lancamento_min = st.number_input("Ano Mínimo de Lançamento", min_value=2013, max_value=2025, value=2020)
        
        col1_sidebar, col2_sidebar = st.columns(2)
        with col1_sidebar:
            ray_tracing = st.checkbox("Requer Ray Tracing", value=True)
        with col2_sidebar:
            upscaling = st.checkbox("Requer Upscaling", value=True)

    if st.sidebar.button("🔎 Encontrar GPUs Recomendadas"):
        criterios = {
            "vram_min": vram_min, "preco_max": preco_max, "ray_tracing": ray_tracing,
            "upscaling": upscaling, "tdp_max": tdp_max, "benchmark_score_min": benchmark_score_min,
            "ano_lancamento_min": ano_lancamento_min
        }
        resultado_filtro = recomendador.recomendar_por_filtro(**criterios)
        
        st.subheader("Resultados da Recomendação Personalizada")
        if resultado_filtro.empty:
            st.warning("Nenhuma GPU encontrada com os critérios especificados. Tente filtros mais flexíveis.")
        else:
            X_pred = resultado_filtro[recomendador.features]
            predicoes_enc = recomendador.clf.predict(X_pred)
            resultado_filtro['Perfil Sugerido pela IA'] = recomendador.perfil_encoder.inverse_transform(predicoes_enc)
            
            st.success(f"Encontramos {len(resultado_filtro)} GPUs que atendem aos seus critérios!")
            st.dataframe(resultado_filtro[['modelo', 'ano_lancamento', 'vram', 'preco', 'benchmark_score', 'Perfil Sugerido pela IA']])
            
            resultado_filtro['custo_beneficio'] = resultado_filtro['benchmark_score'] / resultado_filtro['preco']
            melhor_desempenho = resultado_filtro.loc[resultado_filtro['benchmark_score'].idxmax()]
            melhor_custo_beneficio = resultado_filtro.loc[resultado_filtro['custo_beneficio'].idxmax()]

            st.divider()
            st.subheader("💡 Recomendações em Destaque")
            col1_destaque, col2_destaque = st.columns(2)
            with col1_destaque:
                st.success("🏆 Melhor Desempenho")
                st.markdown(f"**{melhor_desempenho['modelo']}**")
                st.markdown(f"**Benchmark:** {melhor_desempenho['benchmark_score']:.1f} | **Preço:** ${melhor_desempenho['preco']:.2f}")
            with col2_destaque:
                st.info("💰 Melhor Custo-Benefício")
                st.markdown(f"**{melhor_custo_beneficio['modelo']}**")
                st.markdown(f"**Benchmark:** {melhor_custo_beneficio['benchmark_score']:.1f} | **Preço:** ${melhor_custo_beneficio['preco']:.2f}")
    else:
        st.info("Ajuste os filtros na barra lateral esquerda e clique no botão para ver as recomendações.")

with tab2:
    st.header("Explore todas as GPUs de um perfil específico")
    perfis_disponiveis = sorted(recomendador.gpus['perfil_ideal'].unique())
    perfil_selecionado = st.selectbox("Selecione um perfil para visualizar:", options=perfis_disponiveis)
    
    if perfil_selecionado:
        gpus_do_perfil = recomendador.gpus[recomendador.gpus['perfil_ideal'] == perfil_selecionado]
        gpus_do_perfil = gpus_do_perfil.sort_values(by="benchmark_score", ascending=False)
        st.write(f"Análise do perfil **'{perfil_selecionado}'** ({len(gpus_do_perfil)} GPUs encontradas):")
        
        col1_metric, col2_metric, col3_metric = st.columns(3)
        col1_metric.metric("Preço Médio (USD)", f"${gpus_do_perfil['preco'].mean():.0f}")
        col2_metric.metric("Benchmark Médio", f"{gpus_do_perfil['benchmark_score'].mean():.1f}")
        col3_metric.metric("VRAM Média (GB)", f"{gpus_do_perfil['vram'].mean():.1f} GB")
        
        st.write(f"Distribuição de Preços para '{perfil_selecionado}':")
        st.bar_chart(gpus_do_perfil.set_index('modelo')['preco'])
        st.write("Tabela de GPUs do Perfil:")
        st.dataframe(gpus_do_perfil[['modelo', 'ano_lancamento', 'vram', 'preco', 'benchmark_score', 'tdp']])

with tab3:
    st.header("Dashboard: Visão Geral do Mercado de GPUs")
    st.subheader("Preço (USD) vs. Benchmark Score")
    st.markdown("Cada ponto representa uma GPU. A cor indica o perfil, e o tamanho do ponto representa a quantidade de VRAM.")
    st.scatter_chart(recomendador.gpus, x='benchmark_score', y='preco', color='perfil_ideal', size='vram')
    
    st.subheader("Evolução Média do Desempenho (Benchmark) por Ano de Lançamento")
    evolucao = recomendador.gpus.groupby('ano_lancamento')['benchmark_score'].mean()
    st.line_chart(evolucao)
    st.info("Observe a tendência de crescimento do desempenho ao longo dos anos.")