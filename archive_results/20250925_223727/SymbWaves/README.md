# SymbWaves

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Descobrindo equações da física de ondas oceânicas a partir de dados brutos utilizando Regressão Simbólica.

## Visão Geral

**SymbWaves** é um pipeline de modelagem projetado para analisar grandes conjuntos de dados de reanálise oceânica (como ERA5) e descobrir, de forma automática, equações matemáticas que descrevem o estado do mar. Em vez de usar modelos de machine learning de "caixa-preta", este projeto utiliza a biblioteca [PySR](https://github.com/MilesCranmer/pysr) para encontrar fórmulas interpretáveis e fisicamente consistentes.

O objetivo principal é ir além da simples previsão, buscando entender as relações fundamentais entre os parâmetros das ondas (altura, período) e as forçantes ambientais (vento, localização).

## Metodologia Principal

A abordagem deste projeto é construída sobre três pilares fundamentais da modelagem física:

1.  **Adimensionalização:** O modelo não prevê a altura de onda (`Hs`) diretamente. Em vez disso, ele aprende a prever a altura de onda adimensional (`y = g * Hs / U10²`). Isso remove os efeitos de escala e força o modelo a se concentrar nas relações físicas subjacentes, resultando em equações mais robustas e generalizáveis.

2.  **Engenharia de Features Físicas:** As entradas do modelo não são dados brutos, mas sim parâmetros adimensionais com significado físico claro, como a **Idade da Onda** (`Wave_age`) e a **localização normalizada** (`lat_norm`, `lon_norm`).

3.  **Regressão Ponderada (Weighted Regression):** Para lidar com a complexidade dos diferentes regimes de mar (Wind Sea vs. Swell), o modelo emprega um sistema de pesos. As amostras de dados que representam regimes físicos claros (ondas muito jovens ou muito velhas) recebem uma importância maior durante o treinamento. Isso guia o algoritmo a encontrar uma fórmula que seja excelente nos casos mais importantes, em vez de ser medíocre em todos.

## Estrutura do Projeto

O projeto é organizado de forma modular para separar configuração, processamento e modelagem.

```
SymbWaves/
├── config/
│   └── config_weighted.py      # Painel de controle para todos os parâmetros do experimento.
├── scripts/
│   ├── 01_create_data_v2.py    # Script para ler dados brutos (NetCDF) e gerar o CSV processado.
│   └── 02_run_pysr_weighted.py # Script para treinar o modelo PySR com pesos.
├── data/
│   ├── raw/                    # Onde os dados brutos (ex: .nc) devem ser colocados.
│   └── processed/              # Onde os arquivos CSV gerados são salvos.
├── results/                    # Onde os gráficos e mapas de resultados são salvos.
├── outputs/                    # Onde os modelos treinados são salvos (.pkl).
├── .gitignore                  # Especifica quais arquivos o Git deve ignorar.
└── README.md                   # Este arquivo.
```

## Como Começar

### Pré-requisitos

-   Python 3.8+
-   Git
-   Um ambiente virtual (recomendado)

### Instalação

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/lefarina/SymbWaves.git
    cd SymbWaves
    ```

2.  **Crie e ative um ambiente virtual:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Instale as dependências:**
    (É recomendado criar um arquivo `requirements.txt` com as bibliotecas necessárias)
    ```bash
    pip install pysr pandas numpy xarray tqdm matplotlib basemap
    ```

## Fluxo de Trabalho (Como Executar)

O processo é dividido em duas etapas principais:

#### Etapa 1: Pré-processamento dos Dados

Antes de treinar, você precisa converter seus dados brutos (NetCDF) em um arquivo CSV limpo e com todas as features calculadas.

1.  **Coloque seus dados brutos** na pasta `data/raw/`.
2.  **Ajuste os parâmetros** no arquivo `config/config_weighted.py`, principalmente o `raw_df_path` e o `processed_df_path`.
3.  **Execute o script de criação de dados:**
    ```bash
    python scripts/01_create_data_v2.py
    ```
    Este processo pode levar um tempo considerável, dependendo do tamanho do seu dataset.

#### Etapa 2: Treinamento do Modelo

Com o arquivo CSV processado pronto, você pode iniciar o treinamento do modelo de regressão simbólica.

1.  **Ajuste os parâmetros de treinamento** no arquivo `config/config_weighted.py`, como `total_iterations`, `feature_var` e as configurações de peso (`WEIGHT_SETTINGS`).
2.  **Execute o script de treinamento:**
    ```bash
    python scripts/02_run_pysr_weighted.py
    ```
3.  **Analise os Resultados:** O script irá imprimir a equação final encontrada pelo PySR. Os resultados visuais (mapas, gráficos) e os modelos salvos estarão nas pastas `results/` e `outputs/`.

## Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
