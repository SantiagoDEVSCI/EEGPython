import streamlit as st
import yaml
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mne

from eeg_pipeline.preproc import (
    PreprocConfig, pipeline as preproc_pipeline, save_raw, apply_ica
)
from eeg_pipeline.microstates import (
    MicrostatesConfig, microstate_pipeline, compute_gfp, pick_gfp_peaks,
    normalize_maps, evaluate_k_range
)
from eeg_pipeline.reporting import save_metrics_csv, save_report_html

st.set_page_config(page_title="EEG Microstates Pipeline", page_icon="🧠", layout="wide")
st.title("🧠 EEG Preprocessing + Microestados (Docker)")
st.caption("Sube BrainVision (.vhdr con .eeg/.vmrk), preprocesa en MNE y calcula microestados.")

cfg_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
with open(cfg_path, "r", encoding="utf-8") as f:
    full_cfg = yaml.safe_load(f)
pcfg = full_cfg["preproc"]
mcfg = full_cfg["microstates"]

st.sidebar.header("Datos y Preprocesamiento")
vhdr_file = st.sidebar.file_uploader("Archivo .vhdr (con .eeg/.vmrk al lado)", type=["vhdr"])

st.sidebar.subheader("Montaje")
montage_mode = st.sidebar.selectbox("Modo de montaje", ["builtin", "file", "none"], index=0)
builtin_options = ["standard_1020","standard_1005","biosemi32","biosemi64","biosemi128","biosemi256",
                   "GSN-HydroCel-32","GSN-HydroCel-64_1.0","GSN-HydroCel-128","GSN-HydroCel-256",
                   "easycap-M10","easycap-M14"]
montage_name = None
montage_file_path = None
if montage_mode == "builtin":
    default_idx = builtin_options.index("standard_1020")
    montage_name = st.sidebar.selectbox("Builtin montage", builtin_options, index=default_idx)
elif montage_mode == "file":
    up = st.sidebar.file_uploader("Archivo montaje (.elc/.sfp/.bvef/.loc/.locs/.ced/.txt)", type=["elc","sfp","bvef","loc","locs","ced","txt"])
    if up is not None:
        import tempfile
        tmp_m_dir = tempfile.mkdtemp()
        montage_file_path = str(Path(tmp_m_dir) / up.name)
        with open(montage_file_path, "wb") as f:
            f.write(up.getbuffer())

st.sidebar.subheader("Filtro")
l_freq = st.sidebar.number_input("Pasaaltas (Hz)", value=float(pcfg["l_freq"]), step=0.1)
h_freq = st.sidebar.number_input("Pasabajas (Hz)", value=float(pcfg["h_freq"]), step=1.0)
order = st.sidebar.selectbox("Orden Butterworth", [4,6,8,10], index=[4,6,8,10].index(pcfg["filter_order"]))

st.sidebar.subheader("Notch")
use_notch = st.sidebar.checkbox("Aplicar Notch", value=False)
notch_freq = st.sidebar.selectbox("Frecuencia", [50, 60], index=0)

st.sidebar.subheader("Remuestreo")
sfreq = st.sidebar.selectbox("Frecuencia destino (Hz)", [125, 200, 250, 500], index=[125,200,250,500].index(pcfg["resample_sfreq"]))

st.sidebar.subheader("Referencia")
ref_mode = st.sidebar.selectbox("Modo", ["average", "linked_mastoids", "single"], index=0)
ref_channels_text = st.sidebar.text_input("Canales referencia (coma-separado)", value="M1,M2" if ref_mode!="average" else "")

st.sidebar.subheader("Canales malos")
bads_text = st.sidebar.text_input("Marcar malos (coma-separado)", value="")

st.sidebar.subheader("ICA")
ica_enabled = st.sidebar.checkbox("Calcular ICA", value=pcfg["ica"]["enabled"])
n_comp = st.sidebar.text_input("n_components (0.99 o entero)", value="0.99")
ica_auto_apply = st.sidebar.checkbox("Aplicar automáticamente (usa EOG sugeridos si hay)", value=False)

st.sidebar.markdown("---")
st.sidebar.header("Microestados")
k_final = st.sidebar.slider("k final", min_value=2, max_value=12, value=int(mcfg["k_final"]))
k_min = st.sidebar.number_input("k mínimo (para curva GEV)", value=int(mcfg["k_range"][0]), step=1, min_value=2)
k_max = st.sidebar.number_input("k máximo (para curva GEV)", value=int(mcfg["k_range"][1]), step=1, max_value=20)
min_dur = st.sidebar.number_input("Mín. duración (ms)", value=float(mcfg["smoothing"]["min_duration_ms"]), step=5.0)

auto_save = st.sidebar.checkbox("Guardar automáticamente al finalizar", value=False)

for key in ["raw","ica","eog_suggest","ms","preproc_log","temp_dir"]:
    if key not in st.session_state: st.session_state[key] = None

colL, colR = st.columns([2,1])

with colL:
    if vhdr_file and st.button("Cargar y Preprocesar"):
        import tempfile
        temp_dir = tempfile.mkdtemp()
        st.session_state.temp_dir = temp_dir
        vhdr_path = Path(temp_dir) / vhdr_file.name
        with open(vhdr_path, "wb") as f:
            f.write(vhdr_file.getbuffer())

        ref_channels = tuple([x.strip() for x in ref_channels_text.split(",") if x.strip()]) if ref_mode!="average" else ()
        bads = tuple([x.strip() for x in bads_text.split(",") if x.strip()])

        try:
            n_components_val = float(n_comp) if "." in n_comp else int(n_comp)
        except ValueError:
            n_components_val = 0.99

        cfg = PreprocConfig(
            l_freq=l_freq, h_freq=h_freq, filter_order=int(order),
            notch_enabled=use_notch, notch_freq=int(notch_freq),
            resample_sfreq=float(sfreq),
            ref_mode=ref_mode, ref_channels=ref_channels,
            bads_manual=bads, interpolate=True,
            ica_enabled=ica_enabled, ica_n_components=n_components_val,
            ica_random_state=97, ica_max_iter="auto",
            ica_auto_apply=ica_auto_apply, detect_eog=True,
            montage_mode=montage_mode,
            montage_name=montage_name or "standard_1020",
            montage_file=montage_file_path
        )

        with st.spinner("Ejecutando preprocesamiento..."):
            out = preproc_pipeline(str(vhdr_path), cfg)
            st.session_state.raw = out["raw"]
            st.session_state.ica = out["ica"]
            st.session_state.eog_suggest = out["eog_suggest"]
            st.session_state.preproc_log = out["log"]

        st.success("Preprocesamiento completo.")
        if st.session_state.eog_suggest:
            st.info(f"ICA sugiere excluir (EOG): {st.session_state.eog_suggest}")

    if st.session_state.raw is not None:
        raw = st.session_state.raw
        st.subheader("Datos")
        st.write(f"Canales: {raw.info['nchan']} | sfreq: {raw.info['sfreq']} Hz | Duración: {raw.times[-1]:.1f}s")

        expander = st.expander("Ver señales y PSD")
        with expander:
            if st.checkbox("PSD promedio", value=False):
                fig = raw.plot_psd(fmax=60, average=True, show=False)
                st.pyplot(fig)
            if st.checkbox("Scroll de señales (10 canales, 10s)", value=False):
                fig = raw.copy().pick("eeg").plot(duration=10, n_channels=10, show=False, scalings="auto")
                st.pyplot(fig)

        if st.session_state.ica is not None and not ica_auto_apply:
            st.markdown("### Revisión ICA")
            exclude_text = st.text_input("Componentes a excluir (coma-separado)", value=",".join(map(str, st.session_state.eog_suggest or [])))
            if st.button("Aplicar ICA con exclusión"):
                try:
                    exclude = [int(x) for x in exclude_text.split(",") if x.strip()!=""]
                except Exception:
                    exclude = []
                with st.spinner("Aplicando ICA..."):
                    raw_new = apply_ica(raw, st.session_state.ica, exclude)
                    st.session_state.raw = raw_new
                st.success(f"ICA aplicado. Excluidos: {exclude}")

        st.markdown("---")
        st.subheader("Microestados")

        if st.checkbox("Calcular curva GEV vs k (usa picos de GFP)", value=True):
            data = raw.get_data(picks="eeg")
            gfp = compute_gfp(data)
            peaks = pick_gfp_peaks(gfp, raw.info["sfreq"], min_distance_ms=10, prominence=0.0)
            maps = normalize_maps(data[:, peaks])
            import matplotlib.pyplot as plt
            ks, gevs = evaluate_k_range(maps, data, int(k_min), int(k_max), n_init=20, max_iter=300)
            fig, ax = plt.subplots(figsize=(5,3))
            ax.plot(ks, gevs, "-o")
            ax.set_xlabel("k"); ax.set_ylabel("GEV"); ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        if st.button("Calcular microestados (k final)"):
            mcfg_obj = MicrostatesConfig(
                k_range=(int(k_min), int(k_max)),
                k_final=int(k_final),
                n_init=20,
                max_iter=300,
                smoothing_enabled=True,
                min_duration_ms=float(min_dur),
                ignore_polarity=True
            )
            with st.spinner("Ejecutando pipeline de microestados..."):
                ms = microstate_pipeline(raw, mcfg_obj)
                st.session_state.ms = ms
            st.success(f"GEV: {st.session_state.ms['gev']:.4f}")

        if st.session_state.ms is not None:
            ms = st.session_state.ms
            k = ms["templates"].shape[1]
            st.write(f"k = {k} clases")

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 3))
            t = np.arange(len(ms["gfp"])) / raw.info["sfreq"]
            ax.plot(t, ms["gfp"], color="black")
            ax.plot(ms["peaks"]/raw.info["sfreq"], ms["gfp"][ms["peaks"]], "r.", ms=4)
            ax.set_xlabel("Tiempo (s)"); ax.set_ylabel("GFP")
            st.pyplot(fig)

            cov = ms["metrics"]["coverage"]; dur = ms["metrics"]["duration"]; occ = ms["metrics"]["occurrence"]
            st.table({
                "Clase": list(range(k)),
                "Coverage": [round(float(x),3) for x in cov],
                "Duración (s)": [round(float(x),3) for x in dur],
                "Ocurrencia (Hz)": [round(float(x),3) for x in occ]
            })

            st.markdown("Topografías:")
            import numpy as np
            ncols = min(4, k); nrows = int(np.ceil(k/ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
            axes = np.atleast_2d(axes)
            for i in range(k):
                ax = axes[i//ncols, i%ncols]
                mne.viz.plot_topomap(ms["templates"][:,i], raw.info, axes=ax, show=False)
                ax.set_title(f"Clase {i}")
            plt.tight_layout()
            st.pyplot(fig)

            out_name = st.text_input("Nombre base de salida", value="subject001")
            if st.button("Guardar resultados"):
                base = Path(__file__).resolve().parents[1]
                preproc_dir = base / "data" / "derivatives" / "preproc"
                ms_dir = base / "data" / "derivatives" / "microstates"
                reports_dir = base / "data" / "reports"
                preproc_dir.mkdir(parents=True, exist_ok=True)
                ms_dir.mkdir(parents=True, exist_ok=True)
                reports_dir.mkdir(parents=True, exist_ok=True)

                fif_out = preproc_dir / f"{out_name}_preproc.fif"
                save_raw(raw, str(fif_out))
                save_metrics_csv(out_name, ms["metrics"], str(ms_dir / f"{out_name}_metrics.csv"))
                save_report_html(out_name, ms["gev"], ms["metrics"], str(reports_dir / f"{out_name}_report.html"))
                st.success("Guardado completo (FIF, CSV, HTML).")
