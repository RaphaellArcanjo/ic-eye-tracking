# Diretrizes de Desenvolvimento e Estrutura do Repositório (AI Guidelines)

Este arquivo define as regras estritas de arquitetura, fluxo de trabalho e organização que **qualquer assistente de IA ou agente autônomo** deve seguir ao sugerir, modificar ou criar arquivos neste repositório.

---

## 1. Visão Geral do Projeto
- **Objetivo:** Iniciação Científica (IC) em Engenharia de Computação focada na detecção/triagem de Transtorno do Espectro Autista (TEA) a partir de dados de *eye-tracking*.
- **Frameworks:** Python, PyTorch, Pandas, NumPy, Scikit-Learn.
- **Protocolo de Validação:** Estritamente **Leave-One-Subject-Out Cross-Validation (LOSO-CV)** ou particionamento independente por participante (*subject-independent split*). **Proibido** particionar amostras/arquivos aleatoriamente para evitar vazamento de dados (*data leakage*).

---

## 2. Estrutura Obrigatória de Diretórios
**REGRA CRÍTICA:** Nunca crie diretórios adicionais na raiz (como `scripts/`, `src/`, `temp/`, etc.). Qualquer novo arquivo deve pertencer estritamente à árvore abaixo:

```text
eye-tracking-asd/
│
├── data/                  # Dados brutos e processados locais (IGNORADOS NO GIT)
│
├── baselines/             # Scripts definitivos (.py) e código de replicação
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

---

## 3. Regras de Localização de Arquivos (Onde salvar cada coisa)
1. **Nunca crie pastas chamadas `scripts/`, `tools/` ou `src/`:**
   - Se for um script Python de processamento, inspeção ou treinamento de replicação, salve em `baselines/` ou dentro da subpasta do modelo correspondente em `baselines/`.
   - Se for um script de inspeção rápida ou geração de relatório técnico, mantenha dentro de `baselines/` (ex.: `baselines/inspect_dataset.py`).
2. **Notebooks:**
   - Todo arquivo `.ipynb` deve residir em `notebooks/`.
   - Use notebooks primariamente para exploração visual, verificação rápida de shapes de tensores e prototipação inicial.
3. **Relatórios gerados em Markdown:**
   - Relatórios automáticos de auditoria de dados (como `relatorio_inspecao_cilia.md`) devem ser salvos preferencialmente na raiz ou dentro de `data/` se forem documentação de dados locais.

---

## 4. Regras Técnicas para Manipulação do Dataset Cilia
1. **Contagem de Participantes:**
   - O recrutamento formal cita 59 participantes, mas apenas **57 possuem gravações válidas**.
   - Scripts e loaders devem filtrar os dados defensivamente e operar com os 57 sujeitos válidos, descartando sujeitos sem gravações válidas ou vazios.
2. **Tratamento de Colunas:**
   - O dataset contém múltiplos arquivos CSV com esquemas heterogêneos.
   - Aplique sempre a **interseção de colunas** (manter apenas colunas presentes em todos os arquivos de interesse).
   - Ignore e descarte colunas ausentes na maioria dos arquivos. Não impute valores fictícios em colunas não compartilhadas.
3. **Integridade de Dados:**
   - Séries temporais devem ser ordenadas por `Timestamp`.
   - Perdas curtas de rastreamento (NaNs) devem ser tratadas via interpolação linear (`interpolate(method='linear')`), mantendo os shapes consistentes para tensores PyTorch.

---

## 5. Boas Práticas de Código
- Escreva código modular, tipado com *type hints* (`typing`) e documentado com *docstrings*.
- Não versione arquivos binários de dados, arquivos `.csv`, pesos de modelos (`.pt`, `.pth`) nem pastas de cache (`__pycache__`).
- Sempre verifique a existência de diretórios antes de salvar arquivos utilizando `pathlib.Path`.