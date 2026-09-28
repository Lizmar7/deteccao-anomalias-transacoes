# 🔍 Detecção de Anomalias em Transações

> Desafio de Projeto — Bootcamp Bradesco: GenAI, Dados & Cyber (DIO)
> Módulo: *Análise de Dados com Python: Da Preparação à Aplicação com Segurança*

Sistema completo em Python para identificar transações bancárias potencialmente
fraudulentas em um cenário realista de **classes extremamente desbalanceadas**
(menos de 0,2% das transações são fraude). O projeto vai além de "treinar um
modelo": discute por que acurácia é enganosa aqui, quais métricas usar,
como o threshold de decisão é, ele mesmo, uma escolha de negócio, e como
tornar as previsões explicáveis.

---

## 🎯 Objetivo

> "Imagine um sistema bancário analisando milhares de transações. A maioria
> seguirá padrões conhecidos... até que uma delas foge desse padrão. Isso é
> uma fraude? Nem sempre."

Este projeto aplica Ciência de Dados e Machine Learning para transformar uma
**anomalia** (comportamento estatisticamente diferente do padrão) em
**evidência investigável** — sem tratar o modelo como um oráculo.

## 🗂️ Estrutura do projeto

```
deteccao-anomalias-transacoes/
├── data/
│   └── creditcard.csv           # dataset (sintético por padrão — ver seção "Dataset")
├── src/
│   ├── generate_synthetic_data.py  # gera dados de exemplo com a estrutura do dataset real
│   ├── data_utils.py             # carregamento, EDA básica, pré-processamento
│   ├── train.py                  # treino de 5 modelos (supervisionados e não supervisionados)
│   ├── evaluate.py               # métricas para classes desbalanceadas + análise de threshold
│   └── explain.py                # explicabilidade com SHAP
├── notebooks/
│   └── deteccao_anomalias.ipynb  # notebook completo, já executado, com gráficos
├── tests/
│   └── test_pipeline.py          # testes unitários do pipeline (pytest)
├── img/                           # gráficos gerados (EDA, ROC/PR, matriz de confusão, SHAP)
├── results/                       # métricas (metrics.json) e análise de threshold (csv)
├── main.py                        # roda o pipeline completo via linha de comando
├── requirements.txt
└── README.md
```

## 📊 Dataset

O pipeline foi construído para o dataset público **[Credit Card Fraud
Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**
(284.807 transações, apenas 492 fraudes — 0,17%), com colunas `Time`,
`V1`..`V28` (componentes de PCA, anonimizadas) e `Amount`.

Como o arquivo real requer login no Kaggle para download, este repositório
inclui um **gerador de dados sintéticos** (`src/generate_synthetic_data.py`)
que recria a mesma estrutura, escala e nível de desbalanceamento (~0,17% de
fraude), para que o projeto rode imediatamente, do zero:

```bash
python src/generate_synthetic_data.py
```

**Para reproduzir com o dataset real**, baixe `creditcard.csv` do Kaggle e
salve em `data/creditcard.csv` — nenhuma outra alteração é necessária, pois
as colunas têm o mesmo nome e formato.

## ⚙️ Como rodar

```bash
# 1. Clonar o repositório e entrar na pasta
git clone <url-do-seu-repo>
cd deteccao-anomalias-transacoes

# 2. Criar ambiente virtual (opcional, recomendado) e instalar dependências
python -m venv .venv && source .venv/bin/activate  # no Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Gerar o dataset de exemplo (ou colocar o creditcard.csv real em data/)
python src/generate_synthetic_data.py

# 4. Rodar o pipeline completo
python main.py

# 5. (Opcional) rodar os testes
pytest tests/ -v

# 6. (Opcional) explorar o notebook
jupyter notebook notebooks/deteccao_anomalias.ipynb
```

O `main.py` imprime o progresso no terminal e salva:
- `results/metrics.json` — métricas de todos os modelos
- `results/threshold_sweep.csv` — efeito do threshold no melhor modelo
- `img/*.png` — todos os gráficos (distribuição de classes, curvas ROC/PR,
  matrizes de confusão, análise de threshold, SHAP)

## 🧠 Metodologia

### 1. Pré-processamento
Split **estratificado** (preserva a proporção de fraudes em treino/teste) e
padronização de `Time`/`Amount` (as demais colunas já vêm normalizadas via PCA).

### 2. Modelos treinados e comparados

| Modelo | Tipo | Como lida com o desbalanceamento |
|---|---|---|
| Isolation Forest | Não supervisionado | Isola pontos "fáceis de separar" do restante |
| Local Outlier Factor | Não supervisionado | Densidade local de vizinhança |
| Regressão Logística | Supervisionado | `class_weight="balanced"` |
| Random Forest | Supervisionado | `class_weight="balanced_subsample"` |
| Regressão Logística + SMOTE | Supervisionado | Oversampling sintético da classe minoritária (só no treino) |

Os modelos não supervisionados **não usam o rótulo `Class` no treino** —
simulam um cenário realista, em que fraudes novas ainda não têm rótulo
confirmado.

### 3. Por que não usamos acurácia
Com ~0,17% de fraude, um modelo que sempre prevê "normal" atinge >99,8% de
acurácia e é completamente inútil. Por isso avaliamos com:

- **Precision** — das transações sinalizadas, quantas eram fraude de fato?
- **Recall** — das fraudes reais, quantas o modelo capturou?
- **F1-score** — equilíbrio entre as duas.
- **ROC-AUC** — separação geral entre as classes.
- **PR-AUC (Average Precision)** — a métrica mais robusta quando a classe
  positiva é rara; é o critério usado para eleger o "melhor modelo".

### 4. O threshold também é uma decisão
Um classificador produz uma probabilidade, não uma decisão. `src/evaluate.py`
inclui uma varredura de thresholds (`threshold_sweep`) que mostra como
Precision e Recall mudam — e portanto como aumentar detecção de fraude
custa mais falsos positivos (atrito para clientes legítimos), e vice-versa.
Essa escolha deve ser feita com o negócio, não só com a métrica.

### 5. Anomalia ≠ fraude: explicabilidade com SHAP
`src/explain.py` gera um gráfico SHAP para o Random Forest, mostrando quais
variáveis mais contribuem para uma transação ser sinalizada — transformando
"o modelo sinalizou" em "o modelo sinalizou e sabemos por quê", uma camada
de investigação essencial antes de tratar uma anomalia como fraude confirmada.

## 📈 Resultados (sobre o dataset sintético incluído no repositório)

| Modelo | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Random Forest (balanceado) | 1.000 | 0.960 | 0.980 | 1.000 | 1.000 |
| Isolation Forest | 0.652 | 0.600 | 0.625 | 0.881 | 0.604 |
| Local Outlier Factor | 0.593 | 0.640 | 0.615 | 0.827 | 0.644 |
| Regressão Logística + SMOTE | 0.112 | 0.840 | 0.197 | 0.994 | 0.484 |
| Regressão Logística (balanceada) | 0.094 | 0.840 | 0.169 | 0.993 | 0.484 |

> ⚠️ Estes números são do dataset **sintético** de exemplo — servem para
> validar que o pipeline funciona ponta a ponta. Com o dataset real do
> Kaggle, os resultados serão diferentes (tipicamente mais desafiadores) e
> mais representativos da literatura sobre esse problema. Vale notar como a
> Regressão Logística tem recall alto mas precision baixa: ela sinaliza
> muitas fraudes reais, mas às custas de muitos falsos positivos — exatamente
> o trade-off que a análise de threshold explora.

### Gráficos gerados

- `img/class_distribution.png` — desbalanceamento entre classes (escala log)
- `img/roc_pr_curves.png` — curvas ROC e Precision-Recall de todos os modelos
- `img/confusion_*.png` — matriz de confusão por modelo
- `img/threshold_sweep.png` — Precision/Recall/F1 em função do threshold
- `img/shap_summary.png` — importância de variáveis (SHAP) para o Random Forest

## 🔐 Conexão com Segurança da Informação

Detecção de anomalias não é só um problema de Machine Learning — é também um
problema de segurança: um adversário que conhece os padrões do detector pode
tentar produzir comportamento parecido o suficiente com o normal para não
ser identificado. Por isso, um sistema de detecção maduro precisa de:
monitoramento contínuo, atualização periódica dos modelos, explicabilidade
das decisões e camadas complementares (regras de negócio + estatística +
Machine Learning), não apenas um modelo estático.

## 🚧 Limitações e próximos passos

- O dataset sintético não reproduz toda a complexidade estatística do
  dataset real do Kaggle — os resultados numéricos aqui são apenas ilustrativos.
- Não há validação temporal (rolling window), relevante em produção, já que
  padrões de fraude mudam ao longo do tempo (*concept drift*).
- Um próximo passo natural seria expor o melhor modelo via uma API simples
  (ex.: FastAPI) e adicionar monitoramento de drift do modelo em produção.

## 🛠️ Tecnologias utilizadas

Python 3 · pandas · NumPy · scikit-learn · imbalanced-learn (SMOTE) ·
SHAP · matplotlib · seaborn · Jupyter · pytest

## 📚 Referência

Artigo-base do desafio: *"Detecção de Anomalias em Transações: quando o
normal deixa de ser normal"* — DIO.

---

*Projeto desenvolvido como parte do Bootcamp Bradesco: GenAI, Dados & Cyber (DIO).*
