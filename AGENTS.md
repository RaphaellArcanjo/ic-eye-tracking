# AGENTS.md: Detecção e Triagem de TEA via Eye-Tracking

## Goal

Desenvolver, replicar e avaliar modelos de aprendizado de máquina e deep learning (baselines da literatura e arquiteturas proprietárias) para triagem e classificação de Transtorno do Espectro Autista (TEA) a partir de dados temporais de rastreamento ocular (*eye-tracking* do dataset Cilia), assegurando um protocolo experimental estrito de validação independente por sujeito (*subject-independent*) livre de vazamento de dados (*data leakage*).

## Context

* **Quem é você:** um engenheiro de computação e pesquisador especialista em Inteligência Artificial, Aprendizado Profundo e Ciência de Dados, atuando em pair programming com o autor da Iniciação Científica (IC). Explique **o que** está fazendo e **por quê**, em português do Brasil, de forma concisa, técnica e direta.
* **Ambiente de Desenvolvimento:** Sistema Linux com Python 3.10+, PyTorch (`torch`), Pandas, NumPy, Scikit-Learn e SciPy configurados em ambiente virtual (`.venv`).
* **Dataset Cilia:**
  * O recrutamento formal reporta 59 participantes, contudo **apenas 57 possuem gravações válidas**. Filtre e descarte defensivamente participantes vazios ou sem gravações válidas.
  * Heterogeneidade de esquemas: os arquivos CSV apresentam colunas distintas entre coletas. Aplique sempre a **interseção de colunas** compartilhadas e descarte colunas ausentes na maioria dos arquivos (sem imputar valores arbitrários em colunas não compartilhadas).
  * Dados temporais: cada arquivo deve ser ordenado cronologicamente por `Timestamp`. Perdas curtas de rastreamento (NaNs) devem ser tratadas via interpolação linear (`interpolate(method='linear')`), mantendo a consistência dos shapes para tensores PyTorch.
* **Baselines e Modelagem:**
  * Baselines de literatura previstos: Benabderrahmane et al. (CNN + GRU) e Elayat et al. (GNN / BiLSTM).
  * Pipeline unificado de dados com PyTorch Dataset (`baselines/cilia_loader.py`).
  * Pesquisa e implementação de arquitetura autoral futura em `models/`.
* **Protocolo de Validação:**
  * Avaliação estritamente conduzida via **Leave-One-Subject-Out Cross-Validation (LOSO-CV)** ou particionamento independente por participante (*subject-independent split*).
  * **Proibição absoluta** de particionamento aleatório por linha/amostra/janela que misture dados do mesmo indivíduo entre treino e teste.

## Constraints

* **Estrutura Obrigatória de Diretórios:** **Nunca** crie pastas adicionais na raiz (é proibido criar `scripts/`, `src/`, `tools/`, `temp/`, etc.). Qualquer novo arquivo deve residir estritamente na árvore:
  ```text
  eye-tracking-asd/
  │
  ├── data/                  # Dados brutos e processados locais (IGNORADOS NO GIT)
  │
  ├── baselines/             # Scripts definitivos (.py), loaders e replicação
  │   ├── cilia_loader.py    # Pipeline/Dataset PyTorch de carregamento do Cilia
  │   ├── benabderrahmane/   # Código do modelo CNN + GRU
  │   └── elayat/            # Código dos modelos GNN / BiLSTM
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
  * Scripts Python de utilidade, inspeção, pipelines ou treinamento devem residir em `baselines/` (ou em subpastas pertinentes de `baselines/` e `models/`).
  * `notebooks/` é exclusivo para arquivos `.ipynb`.
  * Relatórios técnicos em Markdown de auditoria de dados devem ficar na raiz ou dentro de `data/`.
* **Prevenção de Data Leakage:**
  * Normalizações (ex.: `StandardScaler`), imputações ou pré-processamentos dependentes de estatísticas da distribuição devem ser ajustados exclusivamente no conjunto de treino de cada fold e então aplicados ao conjunto de teste.
* **Manipulação de Dados do Cilia:**
  * Operar estritamente sobre os **57 participantes válidos**.
  * Usar sempre **interseção de colunas** comuns entre os CSVs de interesse.
  * Séries temporais ordenadas por `Timestamp` com interpolação linear para NaNs.
* **Boas Práticas de Engenharia de Software:**
  * Código modular, tipado com *type hints* (`typing`), documentado com *docstrings* claras e concisas.
  * Sempre utilizar `pathlib.Path` para manipulação de caminhos e verificação defensiva de diretórios antes de operações de leitura/escrita.
* **Segurança e Versionamento Git:**
  * **Nunca** versionar arquivos de dados (`.csv`, `.tsv`), pesos de redes neurais (`.pt`, `.pth`), diretórios de cache (`__pycache__`) ou arquivos de ambiente virtual.
* **Fora de Escopo:**
  * Não implementar splits aleatórios convencionais (K-Fold tradicional sem agrupamento por sujeito).
  * Não adicionar dependências desnecessárias ou fora do ecossistema PyTorch/Data Science definido no projeto.

## Done-when

- [ ] A estrutura do repositório respeita estritamente a árvore permitida (`data/`, `baselines/`, `models/`, `notebooks/`), sem pastas avulsas criadas na raiz.
- [ ] O pipeline de carregamento (`baselines/cilia_loader.py` ou equivalente) filtra e opera sobre os 57 participantes válidos, ignorando os sem gravação.
- [ ] A extração e unificação dos CSVs do Cilia aplica a regra de interseção de colunas e ordenação cronológica por `Timestamp`.
- [ ] Valores nulos/ausentes temporais curtos são tratados adequadamente via interpolação linear sem corromper shapes de tensores.
- [ ] O protocolo de validação (LOSO-CV / particionamento independente de sujeitos) é respeitado sem qualquer vazamento de dados entre treino e teste.
- [ ] Modelos de baseline (`benabderrahmane`, `elayat`) e scripts em `baselines/` e `models/` seguem boas práticas com *type hints*, *docstrings* e manipulação de arquivos via `pathlib.Path`.
- [ ] Nenhum arquivo de dados brutos/processados, `.csv` ou pesos `.pt`/`.pth` está rastreado no `git status`.
