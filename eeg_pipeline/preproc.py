from __future__ import annotations
import os
import configparser
import mne
from mne import io
from dataclasses import dataclass
from typing import Optional, List, Any, Tuple

# --- Validación simple BrainVision (opcional) ---
from pathlib import Path

def validate_brainvision_triplet(vhdr_path: str):
    """
    Validación opcional: verifica existencia de .vhdr y .vmrk con mismo stem.
    No valida el binario porque MNE lo resuelve desde el .vhdr.
    """
    p = Path(vhdr_path)
    vmrk = p.with_suffix(".vmrk")
    missing = [str(x) for x in (p, vmrk) if not x.exists()]
    if missing:
        raise FileNotFoundError(f"Faltan archivos BrainVision: {missing}")

@dataclass
class PreprocConfig:
    l_freq: float = 1.0
    h_freq: float = 45.0
    filter_order: int = 8
    notch_enabled: bool = False
    notch_freq: int = 50
    resample_sfreq: float = 250.0
    montage_mode: str = "builtin"  # "builtin" | "file" | "none"
    montage_name: str = "standard_1020"
    montage_file: str | None = None
    bads_manual: Tuple[str, ...] = ()
    interpolate: bool = True
    ref_mode: str = "average"      # "average" | "linked_mastoids" | "single"
    ref_channels: Tuple[str, ...] = ()
    ica_enabled: bool = True
    ica_n_components: float | int = 0.99
    ica_random_state: int = 97
    ica_max_iter: Any = "auto"
    ica_auto_apply: bool = False
    detect_eog: bool = True



def load_brainvision(vhdr_path: str, preload: bool = True) -> mne.io.Raw:
    """
    Carga BrainVision con MNE, que maneja encabezados, rutas y codificaciones.
    """
    # Validación opcional (ver función arriba); puedes comentarla si quieres
    try:
        validate_brainvision_triplet(vhdr_path)
    except Exception:
        # Si quieres omitir la validación externa, comenta este bloque
        pass

    raw = mne.io.read_raw_brainvision(vhdr_path, preload=preload, verbose="ERROR")
    return raw

def set_montage_if_needed(raw: mne.io.Raw, mode: str, name: str, file_path: str | None) -> mne.io.Raw:
    raw = raw.copy()
    try:
        if mode == "builtin":
            montage = mne.channels.make_standard_montage(name)
            raw.set_montage(montage, match_case=False, on_missing="warn")
        elif mode == "file":
            if not file_path:
                raise ValueError("Debes especificar 'montage_file' cuando 'montage_mode' = 'file'")
            montage = mne.channels.read_custom_montage(file_path)
            raw.set_montage(montage, match_case=False, on_missing="warn")
        elif mode == "none":
            pass
        else:
            raise ValueError(f"Modo de montaje no reconocido: {mode}")
    except Exception as e:
        print(f"[WARN] No se pudo aplicar montaje ({mode}): {e}")
    return raw

def apply_notch(raw: mne.io.Raw, base_freq: int) -> mne.io.Raw:
    raw = raw.copy()
    nyq = raw.info["sfreq"] / 2.0
    freqs = [base_freq * k for k in range(1, 6) if base_freq * k < nyq]
    if freqs:
        raw.notch_filter(freqs=freqs)
    return raw

def bandpass_filter(raw: mne.io.Raw, l_freq: float, h_freq: float, order: int) -> mne.io.Raw:
    raw = raw.copy()
    raw.filter(l_freq, h_freq, method="iir", iir_params=dict(order=order, ftype="butter"))
    return raw

def resample(raw: mne.io.Raw, sfreq: float) -> mne.io.Raw:
    raw = raw.copy()
    raw.resample(sfreq)
    return raw

def mark_bad_channels(raw: mne.io.Raw, bads: Optional[List[str]]=None) -> mne.io.Raw:
    raw = raw.copy()
    if bads:
        raw.info["bads"] = list(dict.fromkeys((raw.info.get("bads") or []) + bads))
    return raw

def interpolate_bads(raw: mne.io.Raw) -> mne.io.Raw:
    raw = raw.copy()
    if raw.info.get("bads"):
        raw.interpolate_bads(reset_bads=True)
    return raw

def set_reference(raw: mne.io.Raw, mode: str="average", channels: Tuple[str, ...]=()) -> mne.io.Raw:
    raw = raw.copy()
    if mode == "average":
        raw.set_eeg_reference("average", projection=False)
    elif mode == "linked_mastoids":
        raw.set_eeg_reference(list(channels) if channels else ["M1", "M2"], projection=False)
    elif mode == "single":
        if not channels:
            raise ValueError("Debes indicar 'ref_channels' para referencia 'single'")
        raw.set_eeg_reference(list(channels), projection=False)
    else:
        raise ValueError(f"Modo de referencia no reconocido: {mode}")
    return raw

def run_ica(raw: mne.io.Raw, n_components: float|int=0.99, random_state: int=97, max_iter="auto"):
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=random_state, max_iter=max_iter)
    ica.fit(raw)
    return ica

def auto_find_eog(ica, raw):
    picks = []
    try:
        inds_eog, _ = ica.find_bads_eog(raw)
        picks += list(inds_eog)
    except Exception:
        pass
    return sorted(set(picks))

def apply_ica(raw: mne.io.Raw, ica, exclude: list[int]) -> mne.io.Raw:
    ica = ica.copy()
    ica.exclude = exclude
    return ica.apply(raw.copy())

def save_raw(raw: mne.io.Raw, out_fif: str):
    raw.save(out_fif, overwrite=True)

def pipeline(vhdr_path: str, cfg: PreprocConfig) -> dict:
    log = []
    raw = load_brainvision(vhdr_path, preload=True)
    log.append(f"Load BrainVision: {vhdr_path}")

    raw = set_montage_if_needed(raw, cfg.montage_mode, cfg.montage_name, cfg.montage_file)
    log.append(f"Montage: {cfg.montage_mode} ({cfg.montage_name or cfg.montage_file})")

    if cfg.notch_enabled:
        raw = apply_notch(raw, cfg.notch_freq)
        log.append(f"Notch: {cfg.notch_freq} Hz (+ armónicos)")

    raw = bandpass_filter(raw, cfg.l_freq, cfg.h_freq, cfg.filter_order)
    log.append(f"Bandpass: {cfg.l_freq}-{cfg.h_freq} Hz (order={cfg.filter_order})")

    raw = resample(raw, cfg.resample_sfreq)
    log.append(f"Resample: {cfg.resample_sfreq} Hz")

    if cfg.bads_manual:
        raw = mark_bad_channels(raw, list(cfg.bads_manual))
        log.append(f"Manual bads: {list(cfg.bads_manual)}")

    if cfg.interpolate and raw.info.get("bads"):
        raw = interpolate_bads(raw)
        log.append("Interpolate bads")

    raw = set_reference(raw, cfg.ref_mode, cfg.ref_channels)
    log.append(f"Reference: {cfg.ref_mode} ({list(cfg.ref_channels) if cfg.ref_channels else 'default'})")

    ica = None
    eog_suggest = []
    if cfg.ica_enabled:
        ica = run_ica(raw, cfg.ica_n_components, cfg.ica_random_state, cfg.ica_max_iter)
        log.append(f"ICA: n_components={cfg.ica_n_components}")
        if cfg.detect_eog:
            eog_suggest = auto_find_eog(ica, raw)
            if eog_suggest:
                log.append(f"ICA EOG suggested exclude: {eog_suggest}")
        if cfg.ica_auto_apply:
            raw = apply_ica(raw, ica, eog_suggest)
            log.append(f"ICA applied exclude={eog_suggest}")

    return dict(raw=raw, ica=ica, eog_suggest=eog_suggest, log=log)