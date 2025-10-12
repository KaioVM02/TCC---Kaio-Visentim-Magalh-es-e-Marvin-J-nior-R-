import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import re

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

pd.options.mode.chained_assignment = None

class RecomendadorGPU:
    def __init__(self, csv_path):
        self.gpus = self._carregar_e_limpar_dados(csv_path)
        self.perfis_predefinidos = self._carregar_perfis_predefinidos()
        self.clf, self.scaler, self.resultados_comparacao_ia = self._comparar_e_selecionar_modelo()
        self.resultados = []

    def _carregar_e_limpar_dados(self, csv_path):
        
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.lower()
        except FileNotFoundError:
            print(f"Erro Crítico: O arquivo '{csv_path}' não foi encontrado.")
            exit()
        
        df = df.rename(columns={
            "modelo": "modelo", "ano": "ano_lancamento", "vram": "vram",
            "preço": "preco", "upscaling": "upscaling", "raytracing": "ray_tracing",
            "tdp": "tdp", "benchmark": "benchmark_score", "perfilideal": "perfil_ideal_raw"
        })

        df['vram'] = df['vram'].astype(str).str.extract(r'(\d+)').fillna(0).astype(int)
        df['vram'] = df['vram'].apply(lambda x: x // 10 if x > 64 else x)
        df['preco'] = df['preco'].astype(int)
        df['upscaling'] = df['upscaling'].apply(lambda x: 0 if pd.isna(x) or str(x).strip() == 'N/A' else 1)
        df['ray_tracing'] = df['ray_tracing'].apply(lambda x: 0 if pd.isna(x) or str(x).strip() in ['N/A', 'Software'] else 1)
        
        def mapear_perfil(perfil_str):
            if isinstance(perfil_str, str):
                perfil_lower = perfil_str.lower()
                if "criador" in perfil_lower: return "Editor de Vídeo"
                if "entusiasta 4k/vr" in perfil_lower: return "Entusiasta 4K/VR"
                if "1440p" in perfil_lower: return "Gamer 1440p"
                if "1080p (aaa)" in perfil_lower: return "Gamer AAA 1080p"
                if "esports" in perfil_lower: return "eSports"
            return "Gamer Casual"
        
        df['perfil_ideal'] = df['perfil_ideal_raw'].apply(mapear_perfil)
        df.loc[df['vram'] >= 24, 'perfil_ideal'] = 'Pesquisador em IA'

        print("✅ Dados carregados e pré-processados!")
        return df

    def _carregar_perfis_predefinidos(self):
        
        return {
            "eSports": {"vram_min": 4, "preco_max": 200, "benchmark_score_min": 10, "ano_lancamento_min": 2015},
            "Gamer AAA 1080p": {"vram_min": 6, "preco_max": 400, "benchmark_score_min": 25, "ano_lancamento_min": 2019},
            "Gamer 1440p": {"vram_min": 8, "preco_max": 700, "benchmark_score_min": 40, "ano_lancamento_min": 2019},
            "Editor de Vídeo": {"vram_min": 12, "preco_max": 1200, "benchmark_score_min": 40, "ano_lancamento_min": 2019},
            "Entusiasta 4K/VR": {"vram_min": 12, "preco_max": 2000, "benchmark_score_min": 60, "ano_lancamento_min": 2020},
            "Pesquisador em IA": {"vram_min": 24, "preco_max": 3000, "benchmark_score_min": 70, "ano_lancamento_min": 2020},
        }

    def _comparar_e_selecionar_modelo(self):
        """FUNÇÃO ATUALIZADA para incluir otimização de hiperparâmetros com GridSearchCV."""
        print("\n--- INICIANDO COMPARAÇÃO E OTIMIZAÇÃO DE MODELOS DE IA (PODE LEVAR UM TEMPO) ---")
        df_treino = self.gpus.dropna(subset=['perfil_ideal'])
        
        self.perfil_encoder = LabelEncoder()
        df_treino["perfil_enc"] = self.perfil_encoder.fit_transform(df_treino["perfil_ideal"])
        
        self.features = ['ano_lancamento', 'vram', 'preco', 'upscaling', 'ray_tracing', 'tdp', 'benchmark_score']
        X = df_treino[self.features]
        y = df_treino["perfil_enc"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        modelos = {
            "Árvore de Decisão": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "SVM": SVC(random_state=42)
        }
        
        
        param_grids = {
            "Árvore de Decisão": {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            },
            "Random Forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            "SVM": {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            }
        }

        best_score, best_model, resultados_comparacao = 0, None, []
        
       
        for nome, modelo in modelos.items():
            print(f"\nOtimizando o modelo: {nome}...")
            
            
            grid_search = GridSearchCV(estimator=modelo, param_grid=param_grids[nome], cv=5, scoring='accuracy', n_jobs=-1)
            
            
            grid_search.fit(X_train_scaled, y_train)
            
            
            acuracia = grid_search.best_score_
            
            print(f"Melhor acurácia encontrada para {nome}: {acuracia:.2%}")
            print(f"Melhores parâmetros: {grid_search.best_params_}")
            
            resultados_comparacao.append({'modelo': nome, 'acuracia': acuracia})
            
            if acuracia > best_score:
                best_score = acuracia
                
                best_model = grid_search.best_estimator_
        
        print(f"\n🏆 Melhor modelo geral selecionado: {type(best_model).__name__} com acurácia de {best_score:.2%}")

        final_scaler = StandardScaler().fit(X)
        
        best_model.fit(final_scaler.transform(X), y)
        
        return best_model, final_scaler, pd.DataFrame(resultados_comparacao)
    
    def recomendar_por_filtro(self, **criterios):
        resultado = self.gpus.copy()
        for chave, valor in criterios.items():
            if valor is None: continue
            coluna = chave.replace("_min", "").replace("_max", "")
            if "_min" in chave:
                resultado = resultado[resultado[coluna] >= valor]
            elif "_max" in chave:
                resultado = resultado[resultado[coluna] <= valor]
            else:
                resultado = resultado[resultado[chave] == int(valor)]
        return resultado.sort_values(by="benchmark_score", ascending=False)

    def _analisar_e_salvar_resultado(self, resultado_df, perfil):
        
        if resultado_df.empty: return
        X_pred = resultado_df[self.features]
        X_pred_scaled = self.scaler.transform(X_pred)
        predicoes_enc = self.clf.predict(X_pred_scaled)
        resultado_df['perfil_predito_ia'] = self.perfil_encoder.inverse_transform(predicoes_enc)
        resultado_csv = resultado_df.copy()
        resultado_csv["perfil_filtro"] = perfil
        self.resultados.append(resultado_csv)

    def salvar_csv_e_graficos(self):
        """Salva um CSV consolidado e gera todos os gráficos de análise."""
        self._gerar_grafico_comparacao_modelos(self.resultados_comparacao_ia)

        if not self.resultados:
            print("\nNenhum resultado de recomendação foi gerado para análise de perfis.")
            return

        df_final = pd.concat(self.resultados, ignore_index=True)
        df_final.to_csv("resultados_analise.csv", index=False)
        print("\n✅ Análise completa de perfis salva em resultados_analise.csv")
        
        self._gerar_graficos_aprimorados(df_final)

    

    def _gerar_graficos_aprimorados(self, df_final):
        """Gera e salva um painel de gráficos com estética aprimorada para apresentação."""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

        
        df_final["custo_beneficio"] = df_final["benchmark_score"] / df_final["preco"]
        df_final["ia_match_filtro"] = (df_final["perfil_predito_ia"] == df_final["perfil_filtro"]).astype(int)
        grouped = df_final.groupby("perfil_filtro")
        
        indicadores = {
            "Quantidade de GPUs Encontradas": grouped["modelo"].count(),
            "Coerência (IA vs. Filtro) %": (grouped["ia_match_filtro"].sum() / grouped["ia_match_filtro"].count()) * 100,
            
            "Benchmark Médio": grouped["benchmark_score"].mean(),
            "Preço Médio (R$)": grouped["preco"].mean(),
            "Custo-Benefício Médio": grouped["custo_beneficio"].mean(),
            "Consumo Médio (TDP)": grouped["tdp"].mean(),
        }

        fig, axes = plt.subplots(3, 2, figsize=(18, 16))
        fig.suptitle("Análise Comparativa dos Perfis de Recomendação", fontsize=24, fontweight='bold', y=0.97)
        cores = sns.color_palette("crest", n_colors=len(indicadores))

        for ax, (titulo, serie), cor in zip(axes.flatten(), indicadores.items(), cores):
            serie.plot(kind="bar", ax=ax, color=cor, edgecolor='black', width=0.75, alpha=0.9)
            ax.set_title(titulo, fontsize=16, fontweight='bold', pad=12)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['left', 'bottom']].set_color('grey')
            ax.tick_params(axis="x", rotation=15, labelsize=12)
            ax.tick_params(axis="y", labelsize=11)
            
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=12, fontweight='medium', color='black', 
                            xytext=(0, 10), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.6))

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig("grafico_analise_aprimorado.png", dpi=300, bbox_inches='tight')
        print("✅ Gráfico de análise de perfis salvo em grafico_analise_aprimorado.png")
        plt.show()

    def _gerar_grafico_comparacao_modelos(self, df_resultados):
        """Gera um gráfico de barras comparando a acurácia dos modelos de IA."""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

        df_resultados = df_resultados.sort_values('acuracia', ascending=False)

        plt.figure(figsize=(12, 7))
        ax = sns.barplot(x='acuracia', y='modelo', data=df_resultados, palette='viridis', orient='h')
        
        ax.set_title('Comparativo de Acurácia entre Modelos de IA', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Acurácia', fontsize=14, fontweight='bold')
        ax.set_ylabel('Modelo de IA', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.05)
        ax.tick_params(axis='both', labelsize=12)
        ax.spines[['top', 'right']].set_visible(False)

        for p in ax.patches:
            width = p.get_width()
            ax.text(width + 0.01, p.get_y() + p.get_height() / 2, f'{width:.2%}', va='center', 
                    fontsize=12, fontweight='medium')
            
        plt.tight_layout()
        plt.savefig('grafico_comparacao_modelos.png', dpi=300)
        print("✅ Gráfico de comparação de modelos salvo em grafico_comparacao_modelos.png")
        plt.show()


if __name__ == "__main__":
    
    arquivo_csv = "gpu_dataset_simplificado.csv" 
    
    recomendador = RecomendadorGPU(arquivo_csv)
    #recomendador.aplicar_perfis_predefinidos()
    #recomendador.perfil_manual()
    recomendador.salvar_csv_e_graficos()