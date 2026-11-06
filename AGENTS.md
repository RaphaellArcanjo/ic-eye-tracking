# AGENTS.md: Detecção e Triagem de TEA via Eye-Tracking

## Goal

Desenvolver, replicar e avaliar modelos de aprendizado de máquina e deep learning (baselines da literatura e arquiteturas proprietárias) para triagem e classificação de Transtorno do Espectro Autista (TEA) a partir de dados temporais de rastreamento ocular (*eye-tracking* do dataset Cilia), assegurando um protocolo experimental estrito de validação independente por sujeito (*subject-independent*) livre de vazamento de dados (*data leakage*).

## Context

* **Quem é você:** um engenheiro de computação e pesquisador especialista em Inteligência Artificial, Aprendizado Profundo e Ciência de Dados, atuando em pair programming com o autor da Iniciação Científica (IC). Explique **o que** está fazendo e **por quê**, em português do Brasil, de forma concisa, técnica e direta.
* **Ambiente de Desenvolvimento:** Sistema Linux com Python 3.10+, PyTorch (`torch`), Pandas, NumPy, Scikit-Learn e SciPy configurados em ambiente virtual (`.venv`) e suporte a execuções remotas via Google Colab Pro com GPU acelerada.
* **Dataset Cilia:**
  * O recrutamento formal reporta 59 participantes, contudo **apenas 57 possuem gravações válidas**. Filtre e descarte defensivamente participantes vazios ou sem gravações válidas.
  * Os dados pré-processados, limpos e validados estão consolidados no arquivo colunar otimizado `data/processed/cilia_clean.parquet` (1.351.157 pontos temporais brutos).
  * O agrupamento das séries temporais e geração das matrizes/imagens de scan-path é feito por par `(Participant, Stimulus)`, totalizando 2.115 amostras manipuladas em memória via PyTorch.
* **Baselines e Modelagem:**
  * **Baseline da Literatura:** Replicação da arquitetura de Benabderrahmane et al. (CNN 2D + GRU multimodal com fusão tardia) estruturada de forma modular no pacote `baselines/benabderrahmane/`.
  * **Pesquisa Autoral:** A pasta `models/` é reservada exclusivamente para o desenvolvimento e experimentação das novas arquiteturas proprietárias do projeto de IC.
* **Protocolo de Validação:**
  * Avaliação estritamente conduzida via **Leave-One-Subject-Out Cross-Validation (LOSO-CV)** sobre 57 folds (um fold exclusivo por participante).
  * **Proibição absoluta** de particionamento aleatório por linha/amostra/janela que misture dados do mesmo indivíduo entre treino e teste (*zero data leakage*).

## Constraints

* **Estrutura Obrigatória de Diretórios:** **Nunca** crie pastas adicionais na raiz (é proibido criar `scripts/`, `src/`, `tools/`, `temp/`, etc.). Qualquer novo arquivo deve residir estritamente na árvore:
  ```text
  eye-tracking-asd/
  │
  ├── data/                  # Dados brutos e processados locais (.parquet, .csv - IGNORADOS NO GIT)
  │   ├── Eye-tracking Output/ # Coletas originais dos participantes
  │   ├── Metadata_Participants.csv
  │   └── processed/         # Dados limpos (ex.: cilia_clean.parquet)
  │
  ├── baselines/             # Pacote com implementações de referência da literatura
  │   ├── auditoria_cilia.py # Script de auditoria e inspeção do dataset Cilia
  │   └── benabderrahmane/   # Replicação do baseline multimodal
  │       ├── __init__.py
  │       ├── dataset.py     # Carregamento do .parquet e tensores PyTorch
  │       ├── model.py       # Arquitetura BenabderrahmaneNet (CNN + GRU)
  │       ├── training.py    # Loop LOSO-CV e cálculo de métricas
  │       └── visualization.py # Gráficos, scan-paths e matrizes de confusão
  │
  ├── models/                # Futura arquitetura proprietária desenvolvida pelo grupo
  │
  ├── notebooks/             # Apenas Jupyter Notebooks (.ipynb) de exploração e testes visuais
  │
  ├── AGENTS.md              # Este arquivo de instruções para agentes e IAs
  ├── README.md              # Documentação principal e instruções de execução
  └── requirements.txt       # Dependências e bibliotecas do projeto
  ```
* **Localização de Arquivos:**
  * Módulos reutilizáveis de baseline residem em `baselines/benabderrahmane/`. Scripts utilitários de auditoria residem em `baselines/`.
  * `notebooks/` é exclusivo para arquivos `.ipynb`, contendo apenas chamadas de alto nível, gráficos e narrativa experimental.
  * Relatórios técnicos em Markdown de auditoria de dados devem ficar na raiz ou dentro de `data/`.
* **Prevenção de Data Leakage:**
  * Normalizações, imputações ou pré-processamentos dependentes de estatísticas da distribuição devem ser ajustados exclusivamente no conjunto de treino de cada fold e então aplicados ao conjunto de teste.
  * As ponderações de perda por classe (`pos_weight` ou pesos do `CrossEntropyLoss`) devem ser recalculadas dinamicamente a cada fold com base estrita nos dados de treino daquele participante externo.
* **Boas Práticas de Engenharia de Software:**
  * Código modular, tipado com *type hints* (`typing`), documentado com *docstrings* claras e concisas.
  * Utilizar sempre `pathlib.Path` para manipulação defensiva de caminhos relativos ao projeto.
* **Segurança e Versionamento Git:**
  * **Nunca** versionar arquivos de dados (`.parquet`, `.csv`, `.tsv`), pesos de redes neurais (`.pt`, `.pth`), diretórios de cache (`__pycache__`) ou arquivos de ambiente virtual.
* **Fora de Escopo:**
  * Não implementar splits aleatórios convencionais (K-Fold tradicional sem agrupamento por sujeito).
  * Não adicionar dependências desnecessárias ou fora do ecossistema PyTorch/Data Science definido no projeto.

## Done-when

- [ ] A estrutura do repositório respeita estritamente a árvore permitida (`data/`, `baselines/`, `models/`, `notebooks/`), sem pastas avulsas criadas na raiz.
- [ ] O pipeline de dados em `baselines/benabderrahmane/dataset.py` opera sobre os 57 participantes válidos a partir de `data/processed/cilia_clean.parquet`.
- [ ] O protocolo de validação (LOSO-CV em 57 folds) é respeitado sem qualquer vazamento de dados entre treino e teste.
- [ ] Os módulos em `baselines/benabderrahmane/` e `models/` seguem boas práticas com *type hints*, *docstrings* e manipulação de caminhos via `pathlib.Path`.
- [ ] Os notebooks em `notebooks/` mantêm código enxuto, importando classes e métodos do pacote `baselines`.
- [ ] Nenhum arquivo de dados brutos/processados, `.parquet`, `.csv` ou pesos `.pt`/`.pth` está rastreado no `git status`.
