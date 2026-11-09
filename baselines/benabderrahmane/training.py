"""Rotinas de treinamento, avaliação por sujeito (LOSO-CV) e métricas de benchmark."""

import json
from pathlib import Path
import time
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baselines.benabderrahmane.dataset import (
    CLASS_TO_LABEL,
    BenabderrahmaneDataset,
    CachedTensorDataset,
)
from baselines.benabderrahmane.model import BenabderrahmaneNet


def _to_numpy(data: Any) -> np.ndarray:
    """Converte tensores PyTorch (em CPU ou CUDA) ou sequências para NumPy de forma segura."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], torch.Tensor):
        return np.asarray([t.detach().cpu().numpy() for t in data])
    return np.asarray(data)


def get_device(
    device: Optional[torch.device | str] = None,
    verbose: bool = True,
) -> torch.device:
    """Seleciona ou valida dinamicamente o dispositivo de aceleração (`cuda` ou `cpu`).

    Args:
        device: Dispositivo opcional pré-definido (`torch.device` ou `str`).
        verbose: Se True, exibe detalhes de versão do PyTorch e hardware detectado.

    Returns:
        Instância de `torch.device`.
    """
    resolved_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    if verbose:
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Dispositivo selecionado: {resolved_device}")
        if resolved_device.type == "cuda":
            print(f"  └─ GPU Ativa: {torch.cuda.get_device_name(0)}")
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  └─ VRAM Disponível: {vram_gb:.2f} GB")
        else:
            print("  └─ CPU multiprocessada ativa (GPU não detectada neste kernel)")
    return resolved_device


def compute_class_weights(
    labels: Sequence[int] | np.ndarray | torch.Tensor,
    device: Optional[torch.device | str] = None,
) -> tuple[torch.Tensor, int, int]:
    """Calcula pesos inversamente proporcionais às frequências das classes no conjunto de treino.

    Args:
        labels: Rótulos binários (`0` = TD, `1` = ASD) exclusivamente do conjunto de treino.
        device: Dispositivo onde alocar o tensor de pesos.

    Returns:
        Tupla `(class_weights_tensor, n_td, n_asd)`.
    """
    device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    labels_np = _to_numpy(labels).astype(int)
    n_td = int(np.sum(labels_np == 0))
    n_asd = int(np.sum(labels_np == 1))
    total = len(labels_np)

    w0 = total / (2.0 * n_td) if n_td > 0 else 1.0
    w1 = total / (2.0 * n_asd) if n_asd > 0 else 1.0
    weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)
    return weights, n_td, n_asd


def compute_metrics(
    y_true: Sequence[int] | np.ndarray | torch.Tensor,
    y_pred: Sequence[int] | np.ndarray | torch.Tensor,
    verbose: bool = True,
) -> dict[str, Any]:
    """Calcula e imprime métricas básicas de classificação binária (TD vs ASD).

    Args:
        y_true: Vetor de rótulos reais (`0` ou `1`).
        y_pred: Vetor de rótulos preditos (`0` ou `1`).
        verbose: Se True, imprime o relatório formatado.

    Returns:
        Dicionário contendo `accuracy`, `f1_macro`, `f1_weighted` e `confusion_matrix`.
    """
    y_true_np = _to_numpy(y_true)
    y_pred_np = _to_numpy(y_pred)

    acc = accuracy_score(y_true_np, y_pred_np)
    f1_macro = f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1])

    metrics: dict[str, Any] = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm,
    }

    if verbose:
        print("=" * 65)
        print("MÉTRICAS DE AVALIAÇÃO DE DESEMPENHO (SKLEARN)")
        print("=" * 65)
        print(f"• Acurácia (Accuracy)         : {acc * 100:.2f}%")
        print(f"• F1-Score (Macro)            : {f1_macro:.4f}")
        print(f"• F1-Score (Ponderado)        : {f1_weighted:.4f}")
        print("-" * 65)
        print("Matriz de Confusão (Linhas: Real, Colunas: Predito):")
        print("                Pred TD (0)   Pred ASD (1)")
        print(f"  Real TD  (0):     {cm[0, 0]:<14} {cm[0, 1]}")
        print(f"  Real ASD (1):     {cm[1, 0]:<14} {cm[1, 1]}")
        print("=" * 65)

    return metrics


def compute_full_benchmark_metrics(
    all_y_true: Sequence[int] | np.ndarray | torch.Tensor,
    all_y_pred: Sequence[int] | np.ndarray | torch.Tensor,
    all_y_probs: Sequence[Any] | np.ndarray | torch.Tensor,
    verbose: bool = True,
) -> dict[str, Any]:
    """Calcula e exibe a tabela consolidada de métricas do benchmark Benabderrahmane et al. (2024).

    Args:
        all_y_true: Lista, array ou tensor com todos os rótulos reais avaliados nos folds.
        all_y_pred: Lista, array ou tensor com todas as classes preditas.
        all_y_probs: Probabilidades preditas (vetor 1D da classe positiva ou matriz 2D `(N, 2)`).
        verbose: Se True, exibe o relatório consolidado no console.

    Returns:
        Dicionário estruturado com todas as métricas e contagens da matriz de confusão.
    """
    y_true = _to_numpy(all_y_true)
    y_pred = _to_numpy(all_y_pred)
    y_probs = _to_numpy(all_y_probs)

    if y_probs.ndim == 2:
        y_prob_pos = y_probs[:, 1]
    else:
        y_prob_pos = y_probs

    acc = accuracy_score(y_true, y_pred)
    rec_asd = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec_td = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    if len(np.unique(y_true)) > 1:
        auc_roc = roc_auc_score(y_true, y_prob_pos)
    else:
        auc_roc = float("nan")

    metrics: dict[str, Any] = {
        "accuracy": float(acc),
        "sensitivity_recall_asd": float(rec_asd),
        "specificity_td": float(spec_td) if not np.isnan(spec_td) else None,
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "auc_roc": float(auc_roc) if not np.isnan(auc_roc) else None,
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_samples": int(len(y_true)),
    }

    if verbose:
        print("=" * 75)
        print("RELATÓRIO CONSOLIDADO DE MÉTRICAS (BENCHMARK BENABDERRAHMANE ET AL.)")
        print("=" * 75)
        print(f"• Total de Amostras Avaliadas : {len(y_true):,}")
        print(f"• Acurácia Global (Accuracy) : {acc * 100:.2f}%")
        print(f"• Sensibilidade / Recall (ASD): {rec_asd * 100:.2f}%")
        spec_str = f"{spec_td * 100:.2f}%" if not np.isnan(spec_td) else "N/A (sem amostras TD no teste)"
        print(f"• Especificidade (TD)        : {spec_str}")
        print(f"• F1-Score Macro              : {f1_macro:.4f}")
        print(f"• F1-Score Ponderado          : {f1_weighted:.4f}")
        auc_str = f"{auc_roc:.4f}" if not np.isnan(auc_roc) else "N/A (amostras de teste de apenas 1 classe)"
        print(f"• AUC-ROC                     : {auc_str}")
        print("-" * 75)
        print("Matriz de Confusão Consolidada 2x2:")
        print("                       Predito TD (0)     Predito ASD (1)")
        print(f"    Real TD  (0)       {tn:<18} {fp:<18}")
        print(f"    Real ASD (1)       {fn:<18} {tp:<18}")
        print("=" * 75)

    return metrics


def train_single_fold(
    valid_groups: dict[tuple[int, str], pd.DataFrame],
    stim_map: dict[str, int],
    test_pid: int = 1,
    num_epochs: int = 5,
    batch_size: int = 16,
    lr: float = 3e-4,
    device: Optional[torch.device | str] = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Executa o treinamento e avaliação de um único fold LOSO-CV (padrão: Participante ID 1).

    Args:
        valid_groups: Dicionário de grupos `(Participant, Stimulus)` válidos.
        stim_map: Mapeamento `{Stimulus: stimulus_id}`.
        test_pid: ID do participante reservado exclusivamente para teste.
        num_epochs: Número de épocas de treinamento.
        batch_size: Tamanho do lote (batch size).
        lr: Taxa de aprendizado do otimizador Adam.
        device: Dispositivo PyTorch (`cuda` ou `cpu`).
        seed: Semente de reprodutibilidade.

    Returns:
        Dicionário com métricas finais, histórico de perda e o modelo treinado.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    train_keys = [k for k in valid_groups.keys() if k[0] != test_pid]
    test_keys = [k for k in valid_groups.keys() if k[0] == test_pid]

    train_labels = [int(CLASS_TO_LABEL[valid_groups[k]["Class"].iloc[0]]) for k in train_keys]
    class_weights, n_td, n_asd = compute_class_weights(train_labels, device=device)
    class_weights = class_weights.to(device)
    total_train = len(train_labels)
    w0, w1 = float(class_weights[0].detach().cpu().item()), float(class_weights[1].detach().cpu().item())

    print("=" * 75)
    print(f"CONFIGURAÇÃO DO FOLD ÚNICO (TESTE: PARTICIPANTE ID {test_pid})")
    print("=" * 75)
    print(
        f"• Amostras de Treino : {len(train_keys):,} | "
        f"TD: {n_td} ({n_td / total_train * 100:.1f}%), ASD: {n_asd} ({n_asd / total_train * 100:.1f}%)"
    )
    print(f"• Pesos de Classe    : TD={w0:.4f}, ASD={w1:.4f}")
    print(f"• Amostras de Teste  : {len(test_keys):,} (Participante {test_pid})")
    print(f"• Dispositivo        : {device}")
    print("=" * 75)

    train_loader = DataLoader(
        BenabderrahmaneDataset(train_keys, valid_groups, stim_map),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        BenabderrahmaneDataset(test_keys, valid_groups, stim_map),
        batch_size=batch_size,
        shuffle=False,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = BenabderrahmaneNet()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(
        f"\nIniciando treinamento de {num_epochs} épocas "
        f"({len(train_loader)} batches/época, batch_size={batch_size})...\n"
    )

    epoch_losses: list[float] = []
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_train_loss = 0.0
        total_epoch_samples = 0
        t_epoch_start = time.time()

        for batch in train_loader:
            sp = batch["scanpath"].to(device)
            dist = batch["distances"].to(device)
            dur = batch["durations"].to(device)
            sc = batch["scalars"].to(device)
            targets = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(sp, dist, dur, sc)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            bs = int(targets.size(0))
            running_train_loss += float(loss.detach().cpu().item()) * bs
            total_epoch_samples += bs

        avg_train_loss = running_train_loss / total_epoch_samples
        epoch_duration = time.time() - t_epoch_start
        epoch_losses.append(float(avg_train_loss))
        print(f"• Época [{epoch}/{num_epochs}] | Loss Médio de Treino: {avg_train_loss:.4f} | Tempo: {epoch_duration:.1f}s")

    model.eval()
    running_test_loss = 0.0
    total_test_samples = 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    t_eval_start = time.time()
    with torch.no_grad():
        for batch in test_loader:
            sp = batch["scanpath"].to(device)
            dist = batch["distances"].to(device)
            dur = batch["durations"].to(device)
            sc = batch["scalars"].to(device)
            targets = batch["label"].to(device)

            outputs = model(sp, dist, dur, sc)
            loss = criterion(outputs, targets)

            bs = int(targets.size(0))
            running_test_loss += float(loss.detach().cpu().item()) * bs
            total_test_samples += bs

            preds = torch.argmax(outputs, dim=1)
            preds_np = preds.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            all_preds.extend(preds_np.tolist())
            all_targets.extend(targets_np.tolist())

    avg_test_loss = running_test_loss / total_test_samples if total_test_samples > 0 else 0.0
    eval_duration = time.time() - t_eval_start

    print("\n" + "=" * 75)
    print(f"AVALIAÇÃO FINAL NA {num_epochs}ª ÉPOCA - PARTICIPANTE ID {test_pid}")
    print("=" * 75)
    print(f"• Loss de Teste : {avg_test_loss:.4f}")
    print(f"• Tempo de Teste: {eval_duration:.2f}s")
    print("-" * 75)

    metrics = compute_metrics(all_targets, all_preds, verbose=True)
    metrics["test_loss"] = float(avg_test_loss)
    metrics["epoch_losses"] = epoch_losses
    metrics["model"] = model
    return metrics


def run_loso_evaluation(
    dataset: BenabderrahmaneDataset,
    precomputed_cache: Optional[dict[tuple[int, str], dict[str, torch.Tensor]]] = None,
    num_epochs: int = 5,
    batch_size: int = 16,
    lr: float = 3e-4,
    max_folds: Optional[int] = None,
    checkpoint_file: Optional[str | Path] = "loso_benabderrahmane_results.json",
    device: Optional[torch.device | str] = None,
) -> dict[str, Any]:
    """Executa a validação cruzada Leave-One-Subject-Out (LOSO-CV) estrita sobre o dataset Cilia.

    Garante ausência total de vazamento de dados (data leakage): a cada fold, um único participante
    é isolado para teste, os pesos de classe são calculados apenas no treino e o modelo é reinicializado.

    Args:
        dataset: Instância de `BenabderrahmaneDataset`.
        precomputed_cache: Dicionário opcional de tensores pré-computados em RAM.
        num_epochs: Número de épocas de treinamento por fold.
        batch_size: Tamanho do lote (batch size).
        lr: Taxa de aprendizado do otimizador Adam.
        max_folds: Limite opcional de folds a executar (None executa todos os 57 participantes).
        checkpoint_file: Caminho opcional de arquivo JSON para salvar progresso após cada fold.
        device: Dispositivo PyTorch (`cuda` ou `cpu`).

    Returns:
        Dicionário consolidado contendo métricas globais e resumos individuais por fold.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    unique_pids = sorted(list({int(k[0]) for k in dataset.valid_keys}))
    total_participants = len(unique_pids)

    if max_folds is not None:
        folds_to_run = unique_pids[:max_folds]
        print(
            f"Iniciando LOSO-CV limitado aos primeiros {len(folds_to_run)} folds de teste "
            f"(total: {total_participants})...\n"
        )
    else:
        folds_to_run = unique_pids
        print(f"Iniciando LOSO-CV COMPLETO com todos os {total_participants} participantes (57 folds)...\n")

    all_y_true: list[int] = []
    all_y_pred: list[int] = []
    all_y_probs: list[list[float]] = []
    fold_summaries: list[dict[str, Any]] = []

    total_loso_start = time.time()

    for fold_idx, test_pid in enumerate(folds_to_run, start=1):
        fold_start = time.time()
        print("=" * 75)
        print(f"FOLD [{fold_idx}/{len(folds_to_run)}] - SUJEITO DE TESTE ID = {test_pid}")
        print("=" * 75)

        # 1. Particionamento estrito independente por participante
        train_keys = [k for k in dataset.valid_keys if int(k[0]) != test_pid]
        test_keys = [k for k in dataset.valid_keys if int(k[0]) == test_pid]

        test_labels = [int(CLASS_TO_LABEL[dataset.groups[k]["Class"].iloc[0]]) for k in test_keys]
        test_class_name = "TD" if test_labels[0] == 0 else "ASD"

        train_labels = [int(CLASS_TO_LABEL[dataset.groups[k]["Class"].iloc[0]]) for k in train_keys]
        class_weights, n_td, n_asd = compute_class_weights(train_labels, device=device)
        class_weights = class_weights.to(device)
        w0, w1 = float(class_weights[0].detach().cpu().item()), float(class_weights[1].detach().cpu().item())

        print(f"• Treino: {len(train_keys):,} amostras (TD: {n_td}, ASD: {n_asd}) | Pesos: [TD={w0:.3f}, ASD={w1:.3f}]")
        print(f"• Teste : {len(test_keys):,} amostras (Participante {test_pid} | Classe: {test_class_name})")

        # 2. Construção dos DataLoaders (drop_last=True no treino evita batch_size == 1 no BatchNorm)
        if precomputed_cache is not None:
            train_ds: torch.utils.data.Dataset = CachedTensorDataset([precomputed_cache[k] for k in train_keys])
            test_ds: torch.utils.data.Dataset = CachedTensorDataset([precomputed_cache[k] for k in test_keys])
        else:
            train_ds = BenabderrahmaneDataset(train_keys, dataset.groups, dataset.stim_map)
            test_ds = BenabderrahmaneDataset(test_keys, dataset.groups, dataset.stim_map)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # 3. Reinicialização completa de pesos do zero e envio ao device
        torch.manual_seed(42 + fold_idx)
        np.random.seed(42 + fold_idx)

        model = BenabderrahmaneNet()
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # 4. Treinamento do Fold
        for epoch in range(1, num_epochs + 1):
            model.train()
            running_loss = 0.0
            total_samples = 0
            t_ep_start = time.time()

            for batch in train_loader:
                sp = batch["scanpath"].to(device)
                dist = batch["distances"].to(device)
                dur = batch["durations"].to(device)
                sc = batch["scalars"].to(device)
                targets = batch["label"].to(device)

                optimizer.zero_grad()
                outputs = model(sp, dist, dur, sc)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                bs = int(targets.size(0))
                running_loss += float(loss.detach().cpu().item()) * bs
                total_samples += bs

            ep_loss = running_loss / total_samples
            ep_time = time.time() - t_ep_start
            print(f"  └─ Época [{epoch}/{num_epochs}] | Loss Treino: {ep_loss:.4f} | Tempo: {ep_time:.1f}s", flush=True)

        # 5. Avaliação no Participante de Teste
        model.eval()
        fold_preds: list[int] = []
        fold_probs: list[list[float]] = []
        fold_targets: list[int] = []
        running_test_loss = 0.0
        total_test_samples = 0

        with torch.no_grad():
            for batch in test_loader:
                sp = batch["scanpath"].to(device)
                dist = batch["distances"].to(device)
                dur = batch["durations"].to(device)
                sc = batch["scalars"].to(device)
                targets = batch["label"].to(device)

                outputs = model(sp, dist, dur, sc)
                loss = criterion(outputs, targets)
                probs = F.softmax(outputs, dim=1)

                bs = int(targets.size(0))
                running_test_loss += float(loss.detach().cpu().item()) * bs
                total_test_samples += bs

                preds_np = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                probs_np = probs.detach().cpu().numpy()
                targets_np = targets.detach().cpu().numpy()

                fold_preds.extend(preds_np.tolist())
                fold_probs.extend(probs_np.tolist())
                fold_targets.extend(targets_np.tolist())

        fold_test_loss = running_test_loss / total_test_samples if total_test_samples > 0 else 0.0
        fold_acc = accuracy_score(fold_targets, fold_preds)
        fold_duration = time.time() - fold_start

        correct_count = int(np.sum(np.asarray(fold_targets) == np.asarray(fold_preds)))
        print(
            f"  └─ Resumo Fold {fold_idx}: Loss Teste: {fold_test_loss:.4f} | "
            f"Acurácia: {fold_acc * 100:.1f}% ({correct_count}/{len(fold_targets)}) | "
            f"Duração: {fold_duration:.1f}s\n",
            flush=True,
        )

        all_y_true.extend(fold_targets)
        all_y_pred.extend(fold_preds)
        all_y_probs.extend(fold_probs)

        fold_summary = {
            "fold": fold_idx,
            "test_pid": test_pid,
            "class": test_class_name,
            "n_samples": len(fold_targets),
            "correct": correct_count,
            "accuracy": float(fold_acc),
            "test_loss": float(fold_test_loss),
            "duration_s": float(fold_duration),
        }
        fold_summaries.append(fold_summary)

        if checkpoint_file:
            try:
                ckpt_path = Path(checkpoint_file)
                ckpt_data = {
                    "folds_completed": fold_idx,
                    "total_folds": len(folds_to_run),
                    "fold_summaries": fold_summaries,
                    "partial_y_true": all_y_true,
                    "partial_y_pred": all_y_pred,
                }
                ckpt_path.write_text(json.dumps(ckpt_data, indent=2), encoding="utf-8")
            except Exception:
                pass

    total_time = time.time() - total_loso_start
    print(f"LOSO-CV concluído em {total_time:.1f}s ({total_time / 60:.2f} min).\n")

    metrics = compute_full_benchmark_metrics(all_y_true, all_y_pred, all_y_probs, verbose=True)
    metrics["fold_summaries"] = fold_summaries
    metrics["total_time_seconds"] = float(total_time)
    return metrics


# Alias padronizado para importação direta nos notebooks
run_loso_cv = run_loso_evaluation


def validate_forward_pass(
    dataset: BenabderrahmaneDataset,
    device: Optional[torch.device | str] = None,
    batch_size: int = 2,
    seed: int = 42,
) -> torch.Tensor:
    """Executa teste de sanidade do forward pass de `BenabderrahmaneNet` em um lote real.

    Args:
        dataset: Instância de `BenabderrahmaneDataset`.
        device: Dispositivo de execução (`cuda` ou `cpu`).
        batch_size: Tamanho do lote para o teste.
        seed: Semente aleatória para reprodutibilidade.

    Returns:
        Tensor de logits de saída com shape `(batch_size, 2)`.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    torch.manual_seed(seed)
    dataloader_sanity = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batch_sanity = next(iter(dataloader_sanity))

    model_sanity = BenabderrahmaneNet()
    model_sanity = model_sanity.to(device)
    model_sanity.eval()

    sp = batch_sanity["scanpath"].to(device)
    dist = batch_sanity["distances"].to(device)
    dur = batch_sanity["durations"].to(device)
    sc = batch_sanity["scalars"].to(device)

    t0 = time.time()
    with torch.no_grad():
        out_sanity = model_sanity(sp, dist, dur, sc)
    dt = time.time() - t0

    total_params = sum(p.numel() for p in model_sanity.parameters() if p.requires_grad)
    logits_np = out_sanity.detach().cpu().numpy()

    print("=" * 70)
    print(f"Dispositivo de execução : {device}")
    print(f"Shape de saída          : {tuple(out_sanity.shape)}")
    print(f"Logits gerados          :\n{logits_np}")
    print(f"Tempo do forward pass   : {dt * 1000:.2f} ms")
    print(f"Total de parâmetros     : {total_params:,} treináveis")
    print("=" * 70)

    assert out_sanity.shape == torch.Size([batch_size, 2]), "Shape de saída inválido!"
    assert not torch.isnan(out_sanity).any(), "NaN detectado nos logits!"
    print("✓ Teste de sanidade aprovado com sucesso no dispositivo configurado!")

    return out_sanity
