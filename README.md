# RL Buyer Agent - Streamlit Interface 🤖

Esta é uma interface interativa standalone desenvolvida em Streamlit para configurar, treinar, testar e efetuar o *fine-tuning* do **Agente PPO Constrangido** de compra de inventário e gestão de logística de frutas.

---

## 📂 Estrutura da Pasta
```text
rl_buyer_interface/
├── agent/
│   ├── __init__.py
│   ├── actor_critic_v2.py      # Definição das Redes Neurais (Actor e Critic MLP)
│   └── ppo_agent.py            # Algoritmo de Treino PPO Constrangido
├── environment_constrained.py  # Simulação Física e Biológica do Armazém e Frutas (FEFO)
├── training_runner.py          # Runner assíncrono isolado (para segurança de multiprocessamento)
├── app.py                      # Aplicação Streamlit (Frontend/Visualizações)
├── requirements.txt            # Dependências necessárias para alojar online
└── README.md                   # Este ficheiro de documentação
```

---

## 🚀 Como Executar Localmente

### 1. Criar e Ativar Ambiente Virtual (Recomendado)
```bash
python -m venv .venv
# No Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Em Linux/macOS:
source .venv/bin/activate
```

### 2. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 3. Correr a Aplicação Streamlit
```bash
streamlit run app.py
```

A interface abrirá automaticamente no seu navegador em `http://localhost:8501`.

---

## 💡 Funcionalidades Principais

### 🏋️ Tab 1: Treinar Modelo
- **Upload do Dataset**: Carregue o dataset de procura histórica (formatos `.xlsx` ou `.csv`).
- **Definição de Hiperparâmetros**: Ajuste taxas de aprendizagem (learning rates), desconto (Gamma), limite PPO, número de episódios, tamanho do batch, etc.
- **Divisão Treino/Teste**: Selecione a percentagem deslizante de dados históricos dedicados ao treino (ex: 60%).
- **Log em Tempo Real**: Visualize o progresso em direto através de uma caixa que simula um terminal de sistema (`Episodes: X/Y | Profit Avg: Z€`).
- **Descarregar Modelo ZIP**: Ao finalizar, baixe os pesos finais do Actor, Critic, Scaler e estatísticas Z-score num único ficheiro `.zip`.
- **Modos de Execução**: Escolha **Single-Core** para maior estabilidade e compatibilidade, ou **Multi-Core** para tirar partido de múltiplos processadores.

### 🧪 Tab 2: Testar e Fine-Tuning
- **Flexibilidade de Dados**: Use a percentagem de dados restante (ex: 40%) que não foi exposta ao treino ou carregue um novo dataset de dias futuros.
- **Upload de Pesos**: Insira o ficheiro ZIP gerado no treino (ou carregue individualmente os ficheiros de pesos `.pth` do Actor e do Econ Stat).
- **Ajuste de Hiperparâmetros**: Altere os limiares Min-Max (`s`, `S`) da política baseline para comparação.
- **Simulação Ativa**: Clique em "Iniciar Teste" para ver a simulação dia a dia.
- **Gráficos Plotly em Tempo Real**: Veja as curvas de lucro acumulado do **RL Agent**, do **Min-Max** e do **Oráculo (God Mode)** crescerem a cada passo da simulação, com marcas visuais verticais vermelhas nos dias onde ocorreu o *fine-tuning*.
- **Scorecard de KPIs**: Obtenha os valores totais agregados de perdas por apodrecimento, stockouts, vendas perdidas e lucros no final da simulação.
- **Exportação de Relatórios**: Descarregue a planilha Excel detalhada de registos diários ou o novo modelo adaptado e otimizado pós-*fine-tuning*.

---

## ☁️ Como Publicar Online (Streamlit Cloud)

Para colocar a sua interface online gratuitamente através da infraestrutura do Streamlit:

1. **Submeter código no GitHub**:
   Certifique-se de que os ficheiros da pasta `rl_buyer_interface` estão integrados no seu repositório Git público ou privado:
   ```bash
   git add rl_buyer_interface/
   git commit -m "feat: adicionar interface streamlit do agent buyer"
   git push origin main
   ```
2. **Aceder ao Streamlit Community Cloud**:
   Visite [share.streamlit.io](https://share.streamlit.io/) e faça login com a sua conta GitHub.
3. **Criar uma nova App ("New app")**:
   - Selecione o seu repositório.
   - Escolha a branch (ex: `main`).
   - Indique o caminho do ficheiro principal (`rl_buyer_interface/app.py`).
4. **Implementar ("Deploy")**:
   Clique em Deploy. O Streamlit Cloud irá provisionar um contentor, instalar as dependências de `requirements.txt` e disponibilizar um link público permanente da aplicação.

> 💡 **Nota Importante para Streamlit Cloud**: 
> Aconselha-se a utilização do modo **Single-Core** nas definições da Tab de Treino ao rodar na Cloud, de forma a não exceder os limites de uso de memória/CPU atribuídos por contentor gratuito.
