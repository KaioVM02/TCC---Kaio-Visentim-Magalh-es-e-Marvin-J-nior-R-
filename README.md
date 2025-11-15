🤖 Recomendador de GPUs com IA
Este é um dashboard interativo construído com Streamlit e Scikit-learn, projetado para analisar e recomendar placas de vídeo (GPUs). O sistema usa um modelo de Machine Learning treinado para sugerir a GPU ideal com base em filtros personalizados e prever o perfil de uso de cada placa.

🚀 Pré-requisitos
Antes de começar, certifique-se de que você tem o Python 3.8 ou superior instalado em sua máquina.

🛠️ Roteiro de Instalação
Siga estes passos para configurar e executar o projeto localmente.


1. Obtenha os Arquivos
Crie uma pasta para o projeto e coloque os seguintes arquivos essenciais dentro dela:
 * main.py (A lógica do recomendador e modelo de IA)
 * app.py (O código da aplicação web Streamlit)
 * gpu_dataset_simplificado.csv (O conjunto de dados para treino e análise)
   
2. Crie e Ative um Ambiente Virtual
É uma prática recomendada usar um ambiente virtual para isolar as dependências do projeto.
a. Crie o ambiente virtual:
Abra um terminal (Prompt de Comando, PowerShell, ou o terminal do VS Code) dentro da pasta do projeto e execute:
python -m venv .venv

(Use python3 se o comando python não for encontrado)
b. Ative o ambiente virtual:
Para que as bibliotecas sejam instaladas no lugar certo, você precisa ativar o ambiente.
 * No Windows:
   .\.venv\Scripts\activate

 * No macOS ou Linux:
   source .venv/bin/activate

O seu terminal deve agora mostrar um (.venv) no início da linha, indicando que o ambiente está ativo.

3. Instale as Dependências
Com o ambiente virtual ativado, instale todas as bibliotecas necessárias com um único comando:
pip install streamlit pandas scikit-learn matplotlib seaborn

4. Execute a Aplicação
Após a instalação ser concluída, inicie o servidor do Streamlit:
streamlit run app.py

5. Acesse o Dashboard
Seu navegador será aberto automaticamente. Caso contrário, acesse o endereço fornecido no terminal (geralmente http://localhost:8501).

💡 Como Usar
 * A aplicação será iniciada na aba "Recomendação Personalizada".
 * Use os filtros na barra lateral esquerda para definir seu orçamento, VRAM mínima, consumo (TDP) e outros critérios.
 * Clique no botão "🔎 Encontrar GPUs Recomendadas" para ver os resultados.
 * A tabela de resultados mostrará as GPUs que atendem aos seus filtros e o perfil de uso sugerido pela IA.
 * Explore as abas "Explorar GPUs por Perfil" e "Análise do Mercado" para ver mais gráficos e insights sobre o conjunto de dados.

