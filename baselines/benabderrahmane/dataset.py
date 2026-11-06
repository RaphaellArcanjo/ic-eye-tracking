"""Carregamento do dataset Cilia, agrupamento por (Participant, Stimulus) e Dataset PyTorch multimodal."""

from pathlib import Path
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset

# Constantes globais do protocolo Benabderrahmane et al. (2024)
MAX_DIST_LEN: int = 10370
MAX_DUR_LEN: int = 1313
CLASS_TO_LABEL: dict[str, int] = {"TD": 0, "ASD": 1}


def resolve_parquet_path(custom_path: Optional[str | Path] = None) -> Path:
    """Resolve defensivamente o caminho do arquivo `cilia_clean.parquet`.

    Args:
        custom_path: Caminho explícito opcional para o arquivo Parquet.

    Returns:
        Objeto `Path` resolvido apontando para `cilia_clean.parquet`.

    Raises:
        FileNotFoundError: Caso o arquivo não seja encontrado em nenhum diretório candidato.
    """
    if custom_path is not None:
        candidate = Path(custom_path)
        if candidate.exists():
            return candidate.resolve()

    repo_root = Path(__file__).resolve().parents[2]
    possible_paths = [
        repo_root / "data" / "processed" / "cilia_clean.parquet",
        Path.cwd() / "data" / "processed" / "cilia_clean.parquet",
        Path.cwd().parent / "data" / "processed" / "cilia_clean.parquet",
        Path("data/processed/cilia_clean.parquet"),
        Path("../data/processed/cilia_clean.parquet"),
        Path("/content/cilia_clean.parquet"),
        Path("/content/data/processed/cilia_clean.parquet"),
        Path("cilia_clean.parquet"),
    ]

    for path in possible_paths:
        if path.exists():
            return path.resolve()

    raise FileNotFoundError(
        "Arquivo 'cilia_clean.parquet' não localizado em 'data/processed/'. "
        "Certifique-se de ter executado o pré-processamento preliminar."
    )


def load_clean_parquet(parquet_path: Optional[str | Path] = None, verbose: bool = True) -> pd.DataFrame:
    """Carrega os dados limpos consolidados em formato Parquet.

    Args:
        parquet_path: Caminho opcional do arquivo Parquet.
        verbose: Se True, imprime resumo de carregamento no console.

    Returns:
        DataFrame contendo os registros limpos de rastreamento ocular.
    """
    resolved_path = resolve_parquet_path(parquet_path)
    t0 = time.time()
    df = pd.read_parquet(resolved_path)
    elapsed = time.time() - t0

    if verbose:
        print(f"Carregando dados limpos de: {resolved_path}")
        print(f"Total de registros carregados: {len(df):,} (em {elapsed:.2f}s)")
        print(f"Colunas disponíveis: {list(df.columns)}")

    return df


def summarize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula a distribuição absoluta e percentual dos eventos em `Category Right`.

    Args:
        df: DataFrame consolidado do dataset Cilia.

    Returns:
        DataFrame com colunas `Ocorrências` e `Proporção (%)`.
    """
    cat_counts = df["Category Right"].value_counts()
    cat_pcts = df["Category Right"].value_counts(normalize=True) * 100
    return pd.DataFrame({
        "Ocorrências": cat_counts,
        "Proporção (%)": cat_pcts.round(2),
    })


def summarize_participant_stimulus_groups(
    df: pd.DataFrame,
    max_groups: int = 10,
    sampling_rate_hz: float = 50.0,
) -> tuple[int, pd.DataFrame]:
    """Agrupa o DataFrame por `(Participant, Stimulus)` e gera tabela de resumo dos primeiros grupos.

    Args:
        df: DataFrame consolidado do dataset Cilia.
        max_groups: Quantidade máxima de grupos a listar na tabela de resumo.
        sampling_rate_hz: Taxa de amostragem nominal do rastreador ocular (padrão 50 Hz).

    Returns:
        Tupla `(n_groups, df_summary)` com o total de grupos únicos e o DataFrame de resumo.
    """
    grouped = df.groupby(["Participant", "Stimulus"], sort=False)
    n_groups = grouped.ngroups

    summary_rows: list[dict[str, Any]] = []
    for idx, ((pid, stim), group) in enumerate(grouped):
        if idx >= max_groups:
            break
        summary_rows.append({
            "Participant": pid,
            "Class": group["Class"].iloc[0],
            "Stimulus": stim,
            "Total Quadros": len(group),
            "Duração Aprox (s)": round(len(group) / sampling_rate_hz, 2),
            "Fixações": int((group["Category Right"] == "Fixation").sum()),
            "Sacadas": int((group["Category Right"] == "Saccade").sum()),
            "Blinks": int((group["Category Right"] == "Blink").sum()),
            "Taxa Sinal Válido (%)": round(float(group["valid_mask"].mean() * 100), 2),
        })

    return n_groups, pd.DataFrame(summary_rows)


def filter_valid_groups(
    df: pd.DataFrame,
    verbose: bool = True,
) -> tuple[dict[tuple[int, str], pd.DataFrame], int, int]:
    """Filtra os pares `(Participant, Stimulus)` mantendo apenas grupos com sinal ocular válido.

    Descarta grupos onde `valid_mask` soma 0 ou onde há menos de 2 pontos válidos
    com `Category Right` pertencente a `['Fixation', 'Saccade']`.

    Args:
        df: DataFrame consolidado do dataset Cilia.
        verbose: Se True, exibe estatísticas de filtragem.

    Returns:
        Tupla `(valid_groups, total_groups, discarded_groups)`.
    """
    grouped = df.groupby(["Participant", "Stimulus"], sort=False)
    total_groups = grouped.ngroups
    valid_groups: dict[tuple[int, str], pd.DataFrame] = {}
    discarded_groups = 0

    for (pid, stim), group in grouped:
        valid_points = group[
            (group["valid_mask"] == 1.0)
            & (group["Category Right"].isin(["Fixation", "Saccade"]))
        ]
        if group["valid_mask"].sum() > 0 and len(valid_points) >= 2:
            valid_groups[(int(pid), str(stim))] = group
        else:
            discarded_groups += 1

    if verbose:
        pct_valid = (len(valid_groups) / total_groups * 100) if total_groups > 0 else 0.0
        print("=" * 80)
        print(f"Total de grupos originais : {total_groups}")
        print(f"Grupos descartados (s/ sinal): {discarded_groups}")
        print(f"Grupos válidos restantes  : {len(valid_groups)} ({pct_valid:.2f}%)")
        print("=" * 80)

    return valid_groups, total_groups, discarded_groups


def build_stimulus_mapping(df: pd.DataFrame) -> dict[str, int]:
    """Constrói o mapeamento determinístico de nomes de estímulos para IDs inteiros.

    Args:
        df: DataFrame contendo a coluna `Stimulus`.

    Returns:
        Dicionário `{nome_estimulo: indice_inteiro}`.
    """
    unique_stimuli = sorted(df["Stimulus"].unique())
    return {str(stim): idx for idx, stim in enumerate(unique_stimuli)}


def generate_scanpath_image(
    group_df: pd.DataFrame,
    width: int = 600,
    height: int = 400,
) -> np.ndarray:
    """Gera o Scan-path Map 2D (`height=400`, `width=600`) seguindo Benabderrahmane et al. (2024).

    Constrói imagem em escala de cinza com fundo branco (255), trajetória sacádica como linhas
    pretas finas (0) entre pontos válidos consecutivos e círculos preenchidos (0) nos pontos de fixação.

    Args:
        group_df: DataFrame de um único par `(Participant, Stimulus)`.
        width: Largura da imagem em pixels (padrão 600).
        height: Altura da imagem em pixels (padrão 400).

    Returns:
        Array NumPy 2D de shape `(height, width)` e tipo `uint8`.
    """
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)

    valid_df = group_df[group_df["valid_mask"] == 1.0]
    if len(valid_df) == 0:
        return np.array(img, dtype=np.uint8)

    px = np.clip((valid_df["x_clean"].to_numpy() * width).astype(int), 0, width - 1)
    py = np.clip((valid_df["y_clean"].to_numpy() * height).astype(int), 0, height - 1)
    categories = valid_df["Category Right"].to_numpy()

    points = list(zip(px.tolist(), py.tolist()))
    if len(points) >= 2:
        draw.line(points, fill=0, width=1)

    radius = 3
    for (x, y), cat in zip(points, categories):
        if cat == "Fixation":
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=0, outline=0)

    return np.array(img, dtype=np.uint8)


def extract_group_features(
    group_df: pd.DataFrame,
    stim_map: dict[str, int],
    max_dist_len: int = MAX_DIST_LEN,
    max_dur_len: int = MAX_DUR_LEN,
) -> dict[str, Any]:
    """Extrai as sequências temporais (distâncias e durações com zero-padding) e escalares do grupo.

    Args:
        group_df: DataFrame de um único par `(Participant, Stimulus)`.
        stim_map: Dicionário mapeando nome do estímulo para seu ID inteiro.
        max_dist_len: Comprimento máximo para zero-padding do vetor de distâncias.
        max_dur_len: Comprimento máximo para zero-padding do vetor de durações.

    Returns:
        Dicionário contendo `distances`, `durations`, `fixation_rate`, `stimulus_id` e `label`.
    """
    g = group_df.sort_values("RecordingTime [ms]", kind="mergesort")
    valid_df = g[g["valid_mask"] == 1.0]

    # 1. Distâncias Euclidianas entre pontos válidos consecutivos
    if len(valid_df) >= 2:
        x = valid_df["x_clean"].to_numpy(dtype=np.float32)
        y = valid_df["y_clean"].to_numpy(dtype=np.float32)
        dx = np.diff(x)
        dy = np.diff(y)
        distances = np.sqrt(dx**2 + dy**2).astype(np.float32)
    else:
        distances = np.array([], dtype=np.float32)

    distances = distances[:max_dist_len]
    pad_dist = max_dist_len - len(distances)
    distances_padded = np.pad(
        distances, (0, pad_dist), mode="constant", constant_values=0.0
    ).astype(np.float32)

    # 2. Durações de Eventos (blocos contíguos de 'Fixation' e 'Saccade')
    cats = g["Category Right"]
    block_ids = (cats != cats.shift()).cumsum()

    durations_list: list[float] = []
    fixation_time_ms = 0.0

    for _, block in g.groupby(block_ids, sort=False):
        cat = block["Category Right"].iloc[0]
        if cat in ("Fixation", "Saccade"):
            t_start = float(block["RecordingTime [ms]"].iloc[0])
            t_end = float(block["RecordingTime [ms]"].iloc[-1])
            dur = max(0.0, t_end - t_start)
            durations_list.append(dur)
            if cat == "Fixation":
                fixation_time_ms += dur

    durations = np.array(durations_list[:max_dur_len], dtype=np.float32)
    pad_dur = max_dur_len - len(durations)
    durations_padded = np.pad(
        durations, (0, pad_dur), mode="constant", constant_values=0.0
    ).astype(np.float32)

    # 3. Escalares
    total_valid_event_time = float(durations.sum())
    if total_valid_event_time > 0.0:
        fixation_rate = float(fixation_time_ms / total_valid_event_time)
    elif len(valid_df) > 0:
        fixation_rate = float((valid_df["Category Right"] == "Fixation").mean())
    else:
        fixation_rate = 0.0

    stim_name = str(g["Stimulus"].iloc[0])
    stimulus_id = int(stim_map[stim_name])
    class_label = int(CLASS_TO_LABEL[str(g["Class"].iloc[0])])

    return {
        "distances": distances_padded,
        "durations": durations_padded,
        "fixation_rate": fixation_rate,
        "stimulus_id": stimulus_id,
        "label": class_label,
    }


class BenabderrahmaneDataset(Dataset):
    """Dataset PyTorch multimodal com extração sob demanda siguiendo Benabderrahmane et al. (2024)."""

    def __init__(
        self,
        valid_keys: list[tuple[int, str]],
        groups_dict: Optional[dict[tuple[int, str], pd.DataFrame]] = None,
        stim_map: Optional[dict[str, int]] = None,
        df_processed: Optional[pd.DataFrame] = None,
    ) -> None:
        """Inicializa o Dataset multimodal.

        Args:
            valid_keys: Lista de chaves `(Participant, Stimulus)` válidas.
            groups_dict: Dicionário mapeando `(Participant, Stimulus)` para o respectivo DataFrame.
            stim_map: Mapeamento `{Stimulus: stimulus_id}`.
            df_processed: DataFrame bruto opcional (utilizado caso `groups_dict` ou `stim_map` sejam omitidos).
        """
        self.valid_keys = list(valid_keys)

        if groups_dict is not None:
            self.groups = groups_dict
        elif df_processed is not None:
            grouped_df = df_processed.groupby(["Participant", "Stimulus"], sort=False)
            self.groups = {k: grouped_df.get_group(k) for k in self.valid_keys}
        else:
            raise ValueError("Forneça 'groups_dict' ou 'df_processed' para instanciar BenabderrahmaneDataset.")

        if stim_map is not None:
            self.stim_map = stim_map
        elif df_processed is not None:
            self.stim_map = build_stimulus_mapping(df_processed)
        else:
            all_stimuli = sorted({str(k[1]) for k in self.valid_keys})
            self.stim_map = {stim: idx for idx, stim in enumerate(all_stimuli)}

    def __len__(self) -> int:
        return len(self.valid_keys)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        key = self.valid_keys[idx]
        group_df = self.groups[key]

        scanpath_np = generate_scanpath_image(group_df, width=600, height=400).astype(np.float32) / 255.0
        scanpath_tensor = torch.from_numpy(scanpath_np).unsqueeze(0)

        feat = extract_group_features(group_df, stim_map=self.stim_map)
        distances_tensor = torch.from_numpy(feat["distances"]).to(torch.float32)
        durations_tensor = torch.from_numpy(feat["durations"]).to(torch.float32)
        scalars_tensor = torch.tensor(
            [feat["fixation_rate"], float(feat["stimulus_id"])],
            dtype=torch.float32,
        )
        label_tensor = torch.tensor(feat["label"], dtype=torch.long)
        participant_tensor = torch.tensor(int(key[0]), dtype=torch.long)

        return {
            "scanpath": scanpath_tensor,
            "distances": distances_tensor,
            "durations": durations_tensor,
            "scalars": scalars_tensor,
            "label": label_tensor,
            "participant_id": participant_tensor,
        }


class CachedTensorDataset(Dataset):
    """Dataset PyTorch ultrarrápido alimentado por tensores pré-computados em memória RAM."""

    def __init__(self, samples: list[dict[str, torch.Tensor]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


def precompute_dataset_cache(
    dataset: BenabderrahmaneDataset,
    verbose: bool = True,
) -> dict[tuple[int, str], dict[str, torch.Tensor]]:
    """Pré-computa todos os tensores do Dataset na memória RAM para acelerar múltiplos folds LOSO-CV.

    Args:
        dataset: Instância de `BenabderrahmaneDataset`.
        verbose: Se True, imprime o tempo gasto na pré-computação.

    Returns:
        Dicionário mapeando cada chave `(Participant, Stimulus)` para seu dicionário de tensores.
    """
    if verbose:
        print(f"Pré-computando tensores para {len(dataset):,} amostras...")
    t0 = time.time()
    cache: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    for idx in range(len(dataset)):
        key = dataset.valid_keys[idx]
        cache[key] = dataset[idx]

    if verbose:
        print(f"✓ Pré-computação concluída em {time.time() - t0:.1f}s! Tensores prontos na memória RAM.")
    return cache


def inspect_sample_features(
    samples: list[tuple[str, tuple[int, str], pd.DataFrame]],
    stim_map: dict[str, int],
) -> dict[str, dict[str, Any]]:
    """Extrai e exibe um resumo detalhado das sequências e escalares para amostras selecionadas.

    Args:
        samples: Lista de tuplas `(group_name, key, group_df)`.
        stim_map: Mapeamento `{Stimulus: stimulus_id}`.

    Returns:
        Dicionário `{group_name: features_dict}` com os atributos extraídos.
    """
    results: dict[str, dict[str, Any]] = {}
    for group_name, key, group_df in samples:
        feat = extract_group_features(group_df, stim_map=stim_map)
        results[group_name] = feat

        dist_arr = feat["distances"]
        dur_arr = feat["durations"]
        dist_nonzero = dist_arr[dist_arr > 0]
        dur_nonzero = dur_arr[dur_arr > 0]

        print("=" * 85)
        print(f"INSPEÇÃO DA AMOSTRA [{group_name}] -> Participant: {key[0]} | Stimulus: '{key[1]}'")
        print("=" * 85)
        print(f"Shape Array de Distâncias : {dist_arr.shape} | dtype: {dist_arr.dtype}")
        print(f"Shape Array de Durações   : {dur_arr.shape}  | dtype: {dur_arr.dtype}")
        print(f"Fixation Rate (float)     : {feat['fixation_rate']:.4f} ({type(feat['fixation_rate']).__name__})")
        print(f"Stimulus ID (int)         : {feat['stimulus_id']} ({type(feat['stimulus_id']).__name__})")
        print(f"Class Label (int)         : {feat['label']} ({type(feat['label']).__name__})")
        print("-" * 85)
        print(f"Distâncias -> 10 primeiros não-nulos : {np.round(dist_nonzero[:10], 5)}")
        print(f"Distâncias -> 5 últimos (padding)    : {dist_arr[-5:]}")
        print(f"Durações   -> 10 primeiros não-nulos : {np.round(dur_nonzero[:10], 3)}")
        print(f"Durações   -> 5 últimos (padding)    : {dur_arr[-5:]}\n")

    return results


def inspect_dataloader_batch(
    dataset: BenabderrahmaneDataset,
    batch_size: int = 2,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Valida e imprime shapes e tipos de dados de um lote extraído via DataLoader.

    Args:
        dataset: Instância de `BenabderrahmaneDataset`.
        batch_size: Tamanho do batch de inspeção.
        seed: Semente aleatória para reprodutibilidade.

    Returns:
        Dicionário contendo os tensores do primeiro batch.
    """
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batch = next(iter(dataloader))

    print("=" * 85)
    print(f"VALIDAÇÃO DO DATASET E DATALOADER (Total de amostras no Dataset: {len(dataset)})")
    print("=" * 85)
    for k, v in batch.items():
        print(f"Chave: '{k:<14}' | shape: {str(tuple(v.shape)):<18} | dtype: {v.dtype}")
    print("-" * 85)
    print(f"Scanpath min/max no batch : min={batch['scanpath'].min().item():.4f}, max={batch['scanpath'].max().item():.4f}")
    print(f"Scalars do batch          :\n{batch['scalars']}")
    print(f"Labels do batch           : {batch['label']}")
    print(f"Participant IDs do batch  : {batch['participant_id']}")

    return batch

