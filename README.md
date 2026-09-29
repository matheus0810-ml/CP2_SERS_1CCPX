# Energias renováveis com APIs públicas e aprendizado de máquina

Checkpoint 02 — FIAP · Ciência da Computação · 2º semestre
**Integrantes:**
Ângelo Malta Reina — RM 570769
Gustavo Mendonça Duarte — RM 570561
Matheus Carpinheiro Moreno — RM 571770
Renan de Castro Albuquerque — RM 570532
Vinícius Souza Ferraz — RM 570622

## Objetivo

Consultar duas APIs públicas de dados abertos, gerar dois conjuntos de dados e resolver duas tarefas independentes em Python, comparando **três algoritmos em cada uma**:

1. **Classificação** — prever a fonte de um empreendimento de geração (Solar, Eólica ou Hidráulica) a partir da potência outorgada e da localização.
2. **Regressão** — estimar a radiação solar horária em Petrolina (PE) a partir de condições meteorológicas e da hora do dia.

## Dados

| Conjunto | Fonte | Recorte | Linhas |
|---|---|---|---|
| `aneel_classificacao_orange.csv` | [ANEEL — SIGA](https://dadosabertos.aneel.gov.br/dataset/siga-sistema-de-informacoes-de-geracao-da-aneel) (API CKAN/DataStore, recurso `11ec447d-698d-4ab8-977f-b424d5deee6a`) | Até 1200 registros por sigla (`UFV`, `EOL`, `UHE`, `PCH`, `CGH`); siglas hidráulicas agrupadas em *Hidráulica* | 3876 |
| `meteo_regressao_orange.csv` | [Open-Meteo — API histórica](https://open-meteo.com/en/docs/historical-weather-api) | Petrolina (−9,39; −40,50), **01/04/2025 a 30/06/2025**, fuso `America/Recife`, horas locais de 7h a 17h | 1001 |

- As duas APIs são públicas e **não exigem token**. Nenhuma credencial é usada ou publicada.
- Os dados da ANEEL são um **cadastro** (inclui empreendimentos em diferentes fases) e **não medem energia gerada**. A quantidade de exemplos por fonte vem do limite da consulta e **não representa a matriz energética brasileira**.
- Os dados do Open-Meteo são estimativas de modelo/reanálise, não medições de um painel fotovoltaico.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `Aula_APIs_Energia_Renovavel_ML.ipynb` | Notebook completo: consulta às APIs, análise exploratória, treinamento e comparação dos seis modelos, gráficos e interpretação |
| `aneel_classificacao_orange.csv` | Dados gerados pela API da ANEEL (tarefa de classificação) |
| `meteo_regressao_orange.csv` | Dados gerados pela API do Open-Meteo (tarefa de regressão) |

## Como executar

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook Aula_APIs_Energia_Renovavel_ML.ipynb   # Kernel → Restart & Run All
```

O notebook roda do início ao fim, na ordem das células, e salva os gráficos na pasta `figuras/` (criada automaticamente). Por padrão (`ATUALIZAR_DADOS = False`) ele usa os CSVs versionados, que geraram os resultados abaixo. Para consultar as APIs de novo, altere para `ATUALIZAR_DADOS = True` na primeira célula de código — os CSVs serão regenerados (como o cadastro da ANEEL é atualizado continuamente, os números podem mudar um pouco).

---

## Tarefa 1 — Classificação da fonte (ANEEL)

**Entradas (X):** `potencia_kw`, `latitude`, `longitude` · **Alvo (y):** `fonte`
**Limpeza:** 47 linhas com coordenadas inválidas (longitude = 0 ou fora do Brasil) removidas → 3829 empreendimentos.
**Avaliação:** divisão **estratificada 80/20** (3063 treino / 766 teste), semente 42. Métricas por classe com média **macro**.
**Pré-processamento:** `log1p(potência)` + padronização ajustada **apenas no treino** (dentro de um `Pipeline`) para Regressão Logística e kNN; nenhuma para Random Forest.

| Modelo | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| Regressão Logística | 0,832 | 0,847 | 0,828 | 0,825 |
| kNN (k = 15) | 0,950 | 0,951 | 0,950 | 0,950 |
| **Random Forest (300 árvores)** | **0,974** | **0,975** | **0,973** | **0,974** |


**Conclusões**

- A **Random Forest** é o melhor modelo; a ordem se mantém na validação cruzada de 5 dobras no treino. A Regressão Logística fica bem atrás porque as classes formam regiões geográficas e faixas de potência que não se separam com fronteiras lineares.
- **Classes mais confundidas:** Solar → Eólica e Solar → Hidráulica. Usinas solares centralizadas do Nordeste têm potência (~26–34 MW) e localização quase idênticas às dos parques eólicos; solares pequenas no Sudeste se parecem com CGHs/PCHs. Eólica ↔ Hidráulica aparece no Sul.
- **Limitações:** em uma validação espacial (regiões de 0,5° × 0,5° inteiras fora do treino), o F1 da Random Forest cai de 0,974 para **≈ 0,89** — parte do acerto vem de reconhecer complexos de usinas vizinhas. Potência e posição não descrevem a física da fonte (vento, irradiação, relevo, rios); a amostra é limitada por sigla e não é aleatória. O modelo não serve para definir a fonte de um empreendimento novo; no máximo, para sinalizar cadastros suspeitos.

## Tarefa 2 — Regressão da radiação solar (Open-Meteo)

**Entradas (X):** `temperatura_c`, `umidade_pct`, `nuvens_pct`, `vento_kmh`, `hora` · **Alvo (y):** `radiacao_w_m2` · `data_hora` só ordena/separa.
**Avaliação:** divisão **temporal** — as primeiras 801 horas (01/04 a 12/06) para treino e as últimas 200 (12/06 a 30/06) para teste, sem embaralhar.

| Modelo | MAE (W/m²) | MSE ((W/m²)²) | RMSE (W/m²) | R² |
|---|---|---|---|---|
| Regressão Linear | 145,9 | 30 176,6 | 173,7 | 0,357 |
| Árvore de Decisão (prof. ≤ 8) | 84,9 | 13 189,6 | 114,8 | 0,719 |
| **Random Forest (300 árvores)** | **69,4** | **7 904,2** | **88,9** | **0,832** |
| *Referência: média por hora do treino* | *132,6* | *33 643,0* | *183,4* | *0,283* |


**Conclusões**

- A **Random Forest** é a melhor; a Regressão Linear mal supera a previsão ingênua.
- **Papel da hora:** é o atributo mais importante (~49% da importância na Random Forest), mas tem correlação linear de apenas 0,12 com o alvo, porque a radiação segue um **sino** ao longo do dia (7h e 17h são ambas baixas). Modelos baseados em árvores dividem a hora em faixas; o modelo linear não consegue e acaba usando a temperatura como substituta do ciclo solar.
- **Erros:** o teste fica perto do solstício de inverno (radiação média ~372 W/m² contra ~498 W/m² no treino). A Regressão Linear subestima em média ~125 W/m²; os maiores erros da Random Forest aparecem entre 10h e 15h, com nuvens passageiras.
- **Radiação ≠ geração elétrica:** W/m² em superfície horizontal é potência por área, não energia (kWh). A geração depende de área e eficiência dos módulos, inclinação/orientação, temperatura das células, perdas de inversor, cabos e sujeira, limite do inversor e integração no tempo.
