# Triagem de TEA via Eye-Tracking (Dataset Cilia)

Projeto de Iniciação Científica para classificação de Transtorno do Espectro Autista (TEA/ASD) vs. Desenvolvimento Típico (TD) a partir de dados de rastreamento ocular do dataset **Cilia et al. (2022)**, com validação estritamente independente por sujeito (**LOSO-CV**).

> **Status atual:** replicação inicial do baseline multimodal de **Benabderrahmane et al. (2024)**. A arquitetura proprietária do projeto será desenvolvida em `models/`.

---

## Estrutura do repositório

```text
├── data/                        # Dados locais (NÃO versionados — ver seção "Dados")
│   ├── Eye-tracking Output/     # 25 CSVs originais de gravação (SMI RED, 50 Hz)
│   ├── Metadata_Participants.csv
│   └── processed/cilia_clean.parquet
├── baselines/
│   ├── auditoria_cilia.py       # Auditoria do dataset -> gera relatorio_inspecao_cilia.md
│   ├── preparar_cilia.py        # CSVs brutos -> data/processed/cilia_clean.parquet
│   └── benabderrahmane/         # Replicação do baseline
│       ├── dataset.py           # Leitura do parquet, features e Dataset PyTorch
│       ├── model.py             # BenabderrahmaneNet (CNN 2D + 2 GRUs + MLP, fusão tardia)
│       ├── training.py          # Treino, LOSO-CV e métricas
│       └── visualization.py     # Scan-paths e matriz de confusão
├── models/                      # (reservado) arquitetura proprietária
├── notebooks/
│   ├── 01_inspecao_cilia.ipynb          # Inspeção, limpeza e normalização
│   ├── 02_benabderrahmane_prep.ipynb    # Preparação das entradas do baseline
│   └── 03_benabderrahmane_model.ipynb   # Modelo, sanidade e LOSO-CV completo
├── relatorio_inspecao_cilia.md  # Relatório da auditoria de dados
└── requirements.txt
```

---

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Recomenda-se GPU (testado no Google Colab) para o LOSO-CV completo.

---

## Dados

Os dados **não estão no repositório** (restrição de privacidade/tamanho). Há duas formas de obtê-los:

### Opção A — Baixar o parquet pronto (recomendado)

1. Baixe o `cilia_clean.parquet` (~18 MB): **[Google Drive — cilia_clean.parquet](https://drive.google.com/file/d/1CE88Wtr4PcHliwAQk4CvULqXMgJ7gHj6/view?usp=sharing)**
2. Salve em `data/processed/cilia_clean.parquet`.

Isso basta para rodar os notebooks `02` e `03`.

### Opção B — Regenerar a partir dos CSVs brutos

Com os arquivos originais do Cilia em `data/Eye-tracking Output/` (25 CSVs) e `data/Metadata_Participants.csv`:

```bash
python -m baselines.preparar_cilia
```

O script [`baselines/preparar_cilia.py`](baselines/preparar_cilia.py) reproduz a limpeza do notebook `01` e gera um parquet **idêntico** ao disponibilizado no Drive.

### Descrição

- **Participantes:** 59 recrutados, **57 com gravações válidas** (IDs 12 e 16 não possuem nenhuma linha gravada e são descartados). Classes: **27 ASD / 30 TD**. Ver [`relatorio_inspecao_cilia.md`](relatorio_inspecao_cilia.md).
- **`data/processed/cilia_clean.parquet`:** 1.351.157 amostras temporais (olho direito), com as colunas:

| Coluna | Descrição |
| :--- | :--- |
| `Participant` | ID do participante |
| `Stimulus` | Nome do estímulo apresentado (148 únicos) |
| `RecordingTime [ms]` | Instante da amostra |
| `x_clean`, `y_clean` | Coordenadas do olhar normalizadas em [0, 1] (tela 1280×1024), zeros espúrios removidos e blinks curtos (≤ 150 ms) interpolados |
| `valid_mask` | 1.0 se a amostra tem sinal válido |
| `Category Right` | Evento ocular do SMI (`Fixation`, `Saccade`, `Blink`, `Separator`; eventos sem classificação = `Unclassified`) |
| `Class` | `ASD` ou `TD` |

Pipeline de limpeza (sem estatísticas da distribuição, logo sem risco de vazamento entre folds): ordenação temporal → `(0, 0)` espúrios viram NaN → normalização pela resolução da tela → interpolação linear de gaps ≤ 8 amostras dentro de cada `(Participant, Stimulus)` → `valid_mask`.

---

## Baseline: Benabderrahmane et al. (2024)

**Unidade de amostra:** par `(Participante, Estímulo)` → **2.115 amostras** após descartar grupos sem sinal válido.

**Entradas do modelo (4 modalidades):**

| Ramo | Entrada | Shape |
| :--- | :--- | :--- |
| CNN 2D | Imagem do scan-path (fundo branco, trajetória em linhas, fixações como círculos) | `(1, 400, 600)` |
| GRU 1 | Distâncias euclidianas entre pontos válidos consecutivos (zero-padding; `AvgPool1d(5)` → 2.074 passos) | `(10370,)` |
| GRU 2 | Durações (ms) dos blocos contíguos de fixação/sacada (zero-padding) | `(1313,)` |
| MLP | `[Fixation Rate, Stimulus ID]` | `(2,)` |

**Arquitetura:** CNN (4 blocos conv, latente 128) + 2 GRUs unidirecionais de 2 camadas (hidden 64) + MLP escalar (latente 16) → concatenação (272) → classificador denso com Dropout 0.3 → 2 classes. Total: **275.682 parâmetros**.

**Treino:** Adam (lr = 3e-4), batch 16, 5 épocas por fold, `CrossEntropyLoss` com pesos de classe recalculados **apenas no treino** de cada fold.

**Validação:** Leave-One-Subject-Out com **57 folds**. Em cada fold, todas as amostras de um participante ficam exclusivamente no teste e o modelo é reinicializado.

### Resultados (LOSO-CV, 57 folds, 5 épocas)

Métricas calculadas **por amostra** (participante × estímulo, N = 2.115):

| Acurácia | Sensibilidade (ASD) | Especificidade (TD) | F1-macro | AUC-ROC |
| :---: | :---: | :---: | :---: | :---: |
| 67,04% | 60,17% | 71,63% | 0,658 | 0,697 |

### Limitações conhecidas / próximos passos

- Métricas ainda **por amostra**; falta a decisão agregada **por participante**.
- As GRUs usam o estado do último passo sobre sequências com zero-padding no final (sem `pack_padded_sequence`), o que pode diluir o sinal real.
- Durações (ms) e distâncias entram sem normalização; se adotada, deve ser ajustada por fold apenas no treino.
- `Stimulus ID` entra como valor numérico contínuo (embedding/one-hot seria mais adequado).
- Apenas 5 épocas, sem early stopping nem busca de hiperparâmetros.

---

## Como executar

1. Obter os dados (seção **Dados**: baixar do Drive ou `python -m baselines.preparar_cilia`).
2. Auditoria (opcional): `python baselines/auditoria_cilia.py`
3. Rodar os notebooks `02` → `03` (o `03` executa o LOSO-CV completo). O `01` é exploratório.

Uso direto do pacote:

```python
from baselines.benabderrahmane import (
    load_clean_parquet, build_stimulus_mapping, filter_valid_groups,
    BenabderrahmaneDataset, precompute_dataset_cache, run_loso_cv,
)

df = load_clean_parquet()
stim_map = build_stimulus_mapping(df)
groups, _, _ = filter_valid_groups(df)
dataset = BenabderrahmaneDataset(list(groups.keys()), groups, stim_map)
cache = precompute_dataset_cache(dataset)
results = run_loso_cv(dataset, precomputed_cache=cache, num_epochs=5)
```

---

## Referências

- Cilia, F., Carette, R., Elbattah, M., Guérin, J.-L., & Dequen, G. (2022). *Eye-Tracking Dataset to Support the Research on Autism Spectrum Disorder*. In 1st Workshop on Scarce Data in Artificial Intelligence for Healthcare (SDAIH), IJCAI–ECAI 2022.
- Benabderrahmane, B., Gharzouli, M., & Benlecheb, A. (2024). *A novel multi-modal model to assist the diagnosis of autism spectrum disorder using eye-tracking data*. Health Information Science and Systems, 12(1).
