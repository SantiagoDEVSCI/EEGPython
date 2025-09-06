from __future__ import annotations
import numpy as np
import mne
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any
from sklearn.cluster import KMeans
from scipy.signal import find_peaks


@dataclass
class GFPConfig:
    min_distance_ms: float = 10.0
    prominence: float = 0.0


@dataclass
class MicrostatesConfig:
    k_range: Tuple[int, int] = (4, 7)
    k_final: int = 4
    n_init: int = 20
    max_iter: int = 300
    # Usar default_factory para objetos mutables/instancias
    gfp: GFPConfig = field(default_factory=GFPConfig)
    smoothing_enabled: bool = True
    min_duration_ms: float = 30.0
    ignore_polarity: bool = True


def compute_gfp(data: np.ndarray) -> np.ndarray:
    return np.std(data, axis=0, ddof=0)


def pick_gfp_peaks(gfp: np.ndarray, sfreq: float, min_distance_ms: float, prominence: float) -> np.ndarray:
    distance = int((min_distance_ms / 1000.0) * sfreq)
    distance = max(distance, 1)
    peaks, _ = find_peaks(gfp, distance=distance, prominence=prominence)
    return peaks


def normalize_maps(maps: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(maps, axis=0, keepdims=True) + 1e-15
    return maps / norms


def polarity_correlation(a: np.ndarray, b: np.ndarray, ignore_polarity: bool) -> np.ndarray:
    num = a.T @ b
    return np.abs(num) if ignore_polarity else num


def kmeans_maps(maps: np.ndarray, k: int, n_init: int, max_iter: int, random_state: int = 42):
    km = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=random_state, algorithm="lloyd")
    km.fit(maps.T)
    centers = km.cluster_centers_.T
    centers = normalize_maps(centers)
    return centers, km


def backfit(templates: np.ndarray, data: np.ndarray, ignore_polarity: bool = True):
    sims = polarity_correlation(templates, data, ignore_polarity)
    labels = np.argmax(sims, axis=0)
    gev_num = (np.max(sims, axis=0) ** 2)
    gev_den = np.sum(data ** 2, axis=0) + 1e-15
    gev = np.sum(gev_num) / np.sum(gev_den)
    return labels, gev


def temporal_smooth(labels: np.ndarray, sfreq: float, min_duration_ms: float) -> np.ndarray:
    min_len = int((min_duration_ms / 1000.0) * sfreq)
    if min_len <= 1:
        return labels
    x = labels.copy()
    start = 0
    for i in range(1, len(x) + 1):
        if i == len(x) or x[i] != x[start]:
            seg_len = i - start
            if seg_len < min_len:
                prev_lab = x[start - 1] if start > 0 else x[i] if i < len(x) else x[start]
                next_lab = x[i] if i < len(x) else prev_lab
                x[start:i] = prev_lab if prev_lab == next_lab else next_lab
            start = i
    return x


def evaluate_k_range(
    maps: np.ndarray,
    data: np.ndarray,
    k_min: int,
    k_max: int,
    n_init: int,
    max_iter: int,
    ignore_polarity: bool = True,
    random_state: int = 42,
):
    ks, gevs = [], []
    for k in range(k_min, k_max + 1):
        centers, _ = kmeans_maps(maps, k, n_init, max_iter, random_state)
        _, gev = backfit(centers, data, ignore_polarity)
        ks.append(k)
        gevs.append(gev)
    return np.array(ks), np.array(gevs)


def metrics_from_labels(labels: np.ndarray, sfreq: float, k: int) -> dict:
    cov = np.zeros(k)
    dur = np.zeros(k)
    occ = np.zeros(k)
    n = len(labels)
    for s in range(k):
        mask = labels == s
        cov[s] = mask.mean()
        # duración promedio de segmentos
        lengths = []
        i = 0
        while i < n:
            if labels[i] == s:
                j = i
                while j < n and labels[j] == s:
                    j += 1
                lengths.append(j - i)
                i = j
            else:
                i += 1
        dur[s] = (np.mean(lengths) / sfreq) if lengths else 0.0
        # ocurrencia por segundo
        transitions = np.sum((labels[1:] == s) & (labels[:-1] != s))
        occ[s] = transitions / (n / sfreq)
    return dict(coverage=cov, duration=dur, occurrence=occ)


def microstate_pipeline(raw: mne.io.Raw, cfg: MicrostatesConfig) -> Dict[str, Any]:
    data = raw.get_data(picks="eeg")
    gfp = compute_gfp(data)
    peaks = pick_gfp_peaks(gfp, raw.info["sfreq"], cfg.gfp.min_distance_ms, cfg.gfp.prominence)
    maps = normalize_maps(data[:, peaks])

    centers, _ = kmeans_maps(maps, cfg.k_final, cfg.n_init, cfg.max_iter)
    labels, gev = backfit(centers, data, cfg.ignore_polarity)

    if cfg.smoothing_enabled:
        labels = temporal_smooth(labels, raw.info["sfreq"], cfg.min_duration_ms)

    mets = metrics_from_labels(labels, raw.info["sfreq"], centers.shape[1])
    return dict(templates=centers, labels=labels, gfp=gfp, peaks=peaks, gev=float(gev), metrics=mets)