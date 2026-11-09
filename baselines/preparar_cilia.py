"""Gera `data/processed/cilia_clean.parquet` a partir dos CSVs brutos do dataset Cilia.

Reproduz o pipeline de limpeza do notebook `01_inspecao_cilia.ipynb`:

1. Leitura dos 25 CSVs de `data/Eye-tracking Output/` (apenas colunas do olho direito).
2. Descarte de linhas sem participante identificado e dos IDs sem gravação (12 e 16),
   mantendo os 57 participantes válidos de `Metadata_Participants.csv`.
3. Ordenação temporal por `(Participant, Stimulus, RecordingTime [ms])`.
4. Conversão de coordenadas espúrias `(0, 0)` em NaN.
5. Normalização espacial para [0, 1] (tela SMI RED 1280x1024) com clipping.
6. Interpolação linear de gaps curtos (blinks <= 8 amostras ~ 150 ms a 50 Hz),
   restrita a cada par `(Participant, Stimulus)`.
7. `valid_mask` = 1.0 onde há sinal real/recuperado; NaNs residuais viram 0.0.
8. Eventos sem classificação em `Category Right` ('-' no CSV) viram 'Unclassified'.

Nenhuma etapa usa estatísticas da distribuição (média, desvio etc.): a normalização
depende apenas das dimensões físicas da tela, portanto não há vazamento entre folds.

Uso:
    python -m baselines.preparar_cilia
    python -m baselines.preparar_cilia --output data/processed/cilia_clean.parquet
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

SCREEN_WIDTH_PX: float = 1280.0
SCREEN_HEIGHT_PX: float = 1024.0
MAX_BLINK_GAP_SAMPLES: int = 8  # 150 ms / 20 ms (50 Hz) ~ 7.5 -> 8 amostras
EXCLUDED_PARTICIPANT_IDS: frozenset[int] = frozenset({12, 16})

RAW_COLUMNS: list[str] = [
    "RecordingTime [ms]",
    "Point of Regard Right X [px]",
    "Point of Regard Right Y [px]",
    "Participant",
    "Stimulus",
    "Category Right",
]
OUTPUT_COLUMNS: list[str] = [
    "Participant",
    "Stimulus",
    "RecordingTime [ms]",
    "x_clean",
    "y_clean",
    "valid_mask",
    "Category Right",
    "Class",
]

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = REPO_ROOT / "data"


def load_metadata(metadata_path: Path) -> dict[int, str]:
    """Carrega o mapeamento `{ParticipantID: Class}` dos participantes válidos.

    Args:
        metadata_path: Caminho de `Metadata_Participants.csv`.

    Returns:
        Dicionário com a classe (`ASD`/`TD`) dos participantes, sem os IDs excluídos.
    """
    meta = pd.read_csv(metadata_path)
    meta.columns = [c.strip() for c in meta.columns]
    meta["ParticipantID"] = pd.to_numeric(meta["ParticipantID"], errors="coerce")
    meta = meta.dropna(subset=["ParticipantID"])
    meta["ParticipantID"] = meta["ParticipantID"].astype(int)
    meta["Class"] = meta["Class"].astype(str).str.strip().str.upper()
    meta = meta[~meta["ParticipantID"].isin(EXCLUDED_PARTICIPANT_IDS)]
    return dict(zip(meta["ParticipantID"], meta["Class"]))


def load_recordings(recordings_dir: Path, valid_ids: set[int]) -> pd.DataFrame:
    """Lê os CSVs de gravação mantendo apenas linhas de participantes válidos.

    Args:
        recordings_dir: Diretório `data/Eye-tracking Output/`.
        valid_ids: IDs dos participantes válidos.

    Returns:
        DataFrame concatenado com as colunas de `RAW_COLUMNS`.

    Raises:
        FileNotFoundError: Se nenhum CSV for encontrado.
    """
    csv_files = sorted(
        recordings_dir.glob("*.csv"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
    )
    if not csv_files:
        raise FileNotFoundError(f"Nenhum CSV encontrado em: {recordings_dir}")

    chunks: list[pd.DataFrame] = []
    total_raw = 0
    for fpath in csv_files:
        chunk = pd.read_csv(
            fpath,
            usecols=lambda c: c in RAW_COLUMNS,
            na_values=["-", "NA", "N/A", ""],
            low_memory=False,
        )
        total_raw += len(chunk)
        pid = pd.to_numeric(chunk["Participant"], errors="coerce")
        keep = pid.isin(valid_ids)
        chunk = chunk[keep].copy()
        chunk["Participant"] = pid[keep].astype(int)
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    print(f"CSVs lidos: {len(csv_files)} | linhas brutas: {total_raw:,} | "
          f"linhas mantidas: {len(df):,} | participantes: {df['Participant'].nunique()}")
    return df


def clean_and_normalize(
    df: pd.DataFrame,
    screen_width: float = SCREEN_WIDTH_PX,
    screen_height: float = SCREEN_HEIGHT_PX,
    max_blink_gap: int = MAX_BLINK_GAP_SAMPLES,
) -> pd.DataFrame:
    """Ordena, remove zeros espúrios, normaliza, interpola blinks e gera `valid_mask`.

    Args:
        df: DataFrame bruto retornado por `load_recordings`.
        screen_width: Largura da tela em pixels.
        screen_height: Altura da tela em pixels.
        max_blink_gap: Máximo de amostras consecutivas interpoladas.

    Returns:
        DataFrame com `x_clean`, `y_clean` e `valid_mask` adicionados.
    """
    out = df.sort_values(["Participant", "Stimulus", "RecordingTime [ms]"]).reset_index(drop=True)

    x_raw = pd.to_numeric(out["Point of Regard Right X [px]"], errors="coerce")
    y_raw = pd.to_numeric(out["Point of Regard Right Y [px]"], errors="coerce")
    spurious = (x_raw == 0.0) & (y_raw == 0.0)
    x_norm = (x_raw.mask(spurious, np.nan) / screen_width).clip(0.0, 1.0)
    y_norm = (y_raw.mask(spurious, np.nan) / screen_height).clip(0.0, 1.0)

    keys = [out["Participant"], out["Stimulus"]]

    def _interp(s: pd.Series) -> pd.Series:
        return s.interpolate(method="linear", limit=max_blink_gap, limit_direction="forward")

    x_interp = x_norm.groupby(keys, sort=False).transform(_interp)
    y_interp = y_norm.groupby(keys, sort=False).transform(_interp)

    # Eventos sem classificação do SMI ('-' no CSV, lido como NaN) -> 'Unclassified'
    out["Category Right"] = out["Category Right"].fillna("Unclassified")

    out["valid_mask"] = (x_interp.notna() & y_interp.notna()).astype(np.float32)
    out["x_clean"] = x_interp.fillna(0.0).astype(np.float32)
    out["y_clean"] = y_interp.fillna(0.0).astype(np.float32)
    return out


def build_clean_dataset(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Executa o pipeline completo e retorna o DataFrame no formato de `cilia_clean.parquet`.

    Args:
        data_dir: Diretório `data/` contendo metadados e gravações.

    Returns:
        DataFrame com as colunas de `OUTPUT_COLUMNS`.
    """
    class_map = load_metadata(data_dir / "Metadata_Participants.csv")
    raw = load_recordings(data_dir / "Eye-tracking Output", set(class_map))
    clean = clean_and_normalize(raw)
    clean["Class"] = clean["Participant"].map(class_map)

    assert clean["Participant"].nunique() == 57, "Esperados 57 participantes válidos."
    assert not clean["Participant"].isin(EXCLUDED_PARTICIPANT_IDS).any()
    assert clean["Class"].notna().all(), "Participante sem classe nos metadados."
    return clean[OUTPUT_COLUMNS]


def main(argv: Optional[list[str]] = None) -> None:
    """Ponto de entrada de linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "processed" / "cilia_clean.parquet")
    args = parser.parse_args(argv)

    df = build_clean_dataset(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Salvo: {args.output} | {len(df):,} linhas | "
          f"valid_mask=1: {int(df['valid_mask'].sum()):,} ({df['valid_mask'].mean() * 100:.2f}%)")


if __name__ == "__main__":
    main()
