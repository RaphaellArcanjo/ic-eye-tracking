"""Rotinas de visualização de scan-paths 2D, gráficos de dispersão/fixação e matriz de confusão."""

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baselines.benabderrahmane.dataset import generate_scanpath_image


def find_comparison_samples(
    valid_groups: dict[tuple[int, str], pd.DataFrame],
    target_stimulus: str = "1 coucou D.jpg",
    min_valid_rate: float = 0.85,
    min_fixations: int = 80,
) -> tuple[tuple[int, str], pd.DataFrame, tuple[int, str], pd.DataFrame]:
    """Seleciona um par de amostras representativas (1 ASD e 1 TD) para o mesmo estímulo visual.

    Args:
        valid_groups: Dicionário de grupos `(Participant, Stimulus)` válidos.
        target_stimulus: Nome do estímulo alvo para comparação.
        min_valid_rate: Proporção mínima de quadros com sinal válido.
        min_fixations: Quantidade mínima de quadros classificados como fixação.

    Returns:
        Tupla `(asd_key, asd_df, td_key, td_df)`.
    """
    asd_key: Optional[tuple[int, str]] = None
    asd_df: Optional[pd.DataFrame] = None
    td_key: Optional[tuple[int, str]] = None
    td_df: Optional[pd.DataFrame] = None

    for (pid, stim), g in valid_groups.items():
        if stim != target_stimulus:
            continue
        valid_rate = float(g["valid_mask"].mean())
        n_fix = int(((g["valid_mask"] == 1.0) & (g["Category Right"] == "Fixation")).sum())
        cls = str(g["Class"].iloc[0])

        if cls == "ASD" and asd_df is None and valid_rate >= min_valid_rate and n_fix >= min_fixations:
            asd_key, asd_df = (pid, stim), g
        elif cls == "TD" and td_df is None and valid_rate >= min_valid_rate and n_fix >= min_fixations:
            td_key, td_df = (pid, stim), g

        if asd_df is not None and td_df is not None:
            break

    if asd_key is None or asd_df is None or td_key is None or td_df is None:
        raise ValueError(f"Não foram encontradas amostras suficientes para o estímulo '{target_stimulus}'.")

    return asd_key, asd_df, td_key, td_df


def plot_scanpath_comparison(
    asd_df: pd.DataFrame,
    td_df: pd.DataFrame,
    asd_key: tuple[int, str],
    td_key: tuple[int, str],
    width: int = 600,
    height: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    """Gera e plota lado a lado os mapas de scan-path 2D de uma amostra ASD e uma amostra TD.

    Args:
        asd_df: DataFrame da amostra ASD.
        td_df: DataFrame da amostra TD.
        asd_key: Tupla `(Participant, Stimulus)` da amostra ASD.
        td_key: Tupla `(Participant, Stimulus)` da amostra TD.
        width: Largura do mapa em pixels.
        height: Altura do mapa em pixels.

    Returns:
        Tupla `(scanpath_asd, scanpath_td)` contendo os arrays 2D gerados.
    """
    scanpath_asd = generate_scanpath_image(asd_df, width=width, height=height)
    scanpath_td = generate_scanpath_image(td_df, width=width, height=height)

    print("Inspeção dos Tensores / Imagens Geradas:")
    print(
        f"[ASD] Amostra {asd_key} -> shape: {scanpath_asd.shape}, "
        f"dtype: {scanpath_asd.dtype}, min: {scanpath_asd.min()}, max: {scanpath_asd.max()}"
    )
    print(
        f"[TD]  Amostra {td_key} -> shape: {scanpath_td.shape}, "
        f"dtype: {scanpath_td.dtype}, min: {scanpath_td.min()}, max: {scanpath_td.max()}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_valid_asd = int((asd_df["valid_mask"] == 1.0).sum())
    n_fix_asd = int(((asd_df["valid_mask"] == 1.0) & (asd_df["Category Right"] == "Fixation")).sum())
    axes[0].imshow(scanpath_asd, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(
        f"Grupo: ASD | Participant ID: {asd_key[0]} | Estímulo: {asd_key[1]}\n"
        f"Quadros válidos: {n_valid_asd} | Fixações: {n_fix_asd}"
    )
    axes[0].set_xlabel("Largura (px)")
    axes[0].set_ylabel("Altura (px)")

    n_valid_td = int((td_df["valid_mask"] == 1.0).sum())
    n_fix_td = int(((td_df["valid_mask"] == 1.0) & (td_df["Category Right"] == "Fixation")).sum())
    axes[1].imshow(scanpath_td, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(
        f"Grupo: TD | Participant ID: {td_key[0]} | Estímulo: {td_key[1]}\n"
        f"Quadros válidos: {n_valid_td} | Fixações: {n_fix_td}"
    )
    axes[1].set_xlabel("Largura (px)")
    axes[1].set_ylabel("Altura (px)")

    plt.tight_layout()
    plt.show()

    return scanpath_asd, scanpath_td


def plot_fixation_scatter(
    group_df: pd.DataFrame,
    title: Optional[str] = None,
) -> None:
    """Plota o gráfico de dispersão espacial das fixações e trajetória sacádica normalizada `[0, 1]`.

    Args:
        group_df: DataFrame de um grupo `(Participant, Stimulus)`.
        title: Título customizado opcional para o gráfico.
    """
    valid_df = group_df[group_df["valid_mask"] == 1.0]
    fix_df = valid_df[valid_df["Category Right"] == "Fixation"]
    sac_df = valid_df[valid_df["Category Right"] == "Saccade"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    if len(valid_df) >= 2:
        ax.plot(
            valid_df["x_clean"],
            valid_df["y_clean"],
            color="#7f8c8d",
            linewidth=0.9,
            alpha=0.6,
            label="Trajetória Ocular",
        )
    if len(sac_df) > 0:
        ax.scatter(
            sac_df["x_clean"],
            sac_df["y_clean"],
            c="#2980b9",
            s=18,
            alpha=0.7,
            label="Sacadas",
        )
    if len(fix_df) > 0:
        ax.scatter(
            fix_df["x_clean"],
            fix_df["y_clean"],
            c="#c0392b",
            s=36,
            alpha=0.85,
            label="Fixações",
        )

    pid = group_df["Participant"].iloc[0]
    stim = group_df["Stimulus"].iloc[0]
    cls = group_df["Class"].iloc[0]
    ax.set_title(title or f"Dispersão Espacial de Fixações e Sacadas | ID {pid} ({cls}) - '{stim}'")
    ax.set_xlabel("Coordenada X Normalizada [0, 1]")
    ax.set_ylabel("Coordenada Y Normalizada [0, 1]")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1.0, 0.0)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray | Sequence[Sequence[int]] | Any,
    class_names: tuple[str, str] = ("TD (0)", "ASD (1)"),
    title: str = "Matriz de Confusão Consolidada (LOSO-CV)",
) -> None:
    """Renderiza visualmente a matriz de confusão 2x2 com valores absolutos e percentuais.

    Args:
        cm: Matriz de confusão 2x2 (array NumPy, tensor PyTorch ou lista aninhada).
        class_names: Nomes das classes para os eixos X e Y.
        title: Título do gráfico.
    """
    if hasattr(cm, "detach"):
        cm = cm.detach().cpu().numpy()
    cm_arr = np.asarray(cm, dtype=int)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm_arr, row_sums, out=np.zeros_like(cm_arr, dtype=float), where=row_sums > 0) * 100.0

    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Classe Real",
        xlabel="Classe Predita",
        title=title,
    )

    thresh = cm_arr.max() / 2.0 if cm_arr.size > 0 else 0.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            text_color = "white" if cm_arr[i, j] > thresh else "black"
            ax.text(
                j,
                i,
                f"{cm_arr[i, j]}\n({cm_pct[i, j]:.1f}%)",
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.show()
