# Dépendances : pip install pedalboard pyloudnorm soundfile numpy demucs torch
# Optionnel (glisser-déposer) : pip install tkinterdnd2-universal
# + FFmpeg installé et accessible dans le PATH (https://ffmpeg.org/download.html)

import os
import shutil
import subprocess
import threading
import warnings
import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from pedalboard import Pedalboard, Limiter

# Glisser-déposer optionnel. On teste le chargement réel de la DLL tkdnd
# sans créer/détruire de fenêtre (ce qui laissait un résidu "application
# has been destroyed" dans la console).
DND_AVAILABLE = False
try:
    import tkinterdnd2
    _dll_dir = os.path.join(os.path.dirname(tkinterdnd2.__file__), "tkdnd")
    DND_AVAILABLE = os.path.isdir(_dll_dir)
    from tkinterdnd2 import TkinterDnD, DND_FILES
except Exception:
    DND_AVAILABLE = False

warnings.filterwarnings("ignore", message="Possible clipped samples in output.")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_LUFS = -9.0
LIMITER_CEILING_DB = -0.3
DECLIP_THRESH = 0.9895
TAP_SUFFIX = "_TAP"
SUPPORTED_IN = (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aiff", ".aif",
                ".opus", ".wma", ".aac", ".alac", ".ape")

ENCODERS = {
    "flac": (["-c:a", "flac", "-sample_fmt", "s32", "-compression_level", "8"], "flac"),
    "mp3":  (["-c:a", "libmp3lame", "-b:a", "320k"], "mp3"),
}

# Correspondance : libellé affiché -> nom du stem Demucs
STEM_LABELS = {
    "Vocal":   "vocals",
    "Batterie": "drums",
    "Basse":   "bass",
    "Mélodie": "other",
}


def find_ffmpeg():
    """Retourne le chemin de ffmpeg ou None. Cherche PATH + emplacements Windows courants."""
    p = shutil.which("ffmpeg")
    if p:
        return p
    candidates = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        os.path.expanduser(r"~\ffmpeg\bin\ffmpeg.exe"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def is_already_processed(path):
    """Vrai si le nom de fichier porte déjà le marqueur TAP."""
    base = os.path.splitext(os.path.basename(path))[0]
    return base.endswith(TAP_SUFFIX)


def safe_name(name):
    """Nettoie un nom pour l'usage en nom de dossier (retire caractères interdits Windows)."""
    for ch in '<>:"/\\|?*':
        name = name.replace(ch, "_")
    return name.strip()


# ---------------------------------------------------------------------------
# DSP (mode loudness)
# ---------------------------------------------------------------------------
def declip(audio, thresh=DECLIP_THRESH):
    """
    Restauration des crêtes écrêtées par interpolation sur les runs saturés.
    Renvoie (audio_corrigé, nb_échantillons_corrigés).
    """
    out = audio.copy()
    n = out.shape[0]
    fixed = 0
    for ch in range(out.shape[1]):
        x = out[:, ch]
        clipped = np.abs(x) >= thresh
        if not clipped.any():
            continue
        edges = np.diff(clipped.astype(np.int8))
        starts = np.where(edges == 1)[0] + 1
        ends = np.where(edges == -1)[0] + 1
        if clipped[0]:
            starts = np.r_[0, starts]
        if clipped[-1]:
            ends = np.r_[ends, n]
        for s, e in zip(starts, ends):
            l = max(s - 2, 0)
            r = min(e + 1, n - 1)
            if r - l < 3:
                continue
            good = np.array([l, l + 1, r - 1, r])
            good = good[(good < s) | (good >= e)]
            good = good[(good >= 0) & (good < n)]
            if len(good) < 2:
                continue
            interp = np.interp(np.arange(s, e), good, x[good])
            sign = np.sign(x[s]) if x[s] != 0 else 1.0
            x[s:e] = interp * 1.02 if np.all(np.sign(interp) == sign) else interp
            fixed += e - s
        out[:, ch] = x
    return np.clip(out, -1.0, 1.0), fixed


def normalize_lufs(audio, sr, target_lufs=TARGET_LUFS):
    """Normalisation loudness intégrée (ITU-R BS.1770), gain calculé à la main."""
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    if loudness == float("-inf"):
        return audio
    gain_db = target_lufs - loudness
    return audio * (10.0 ** (gain_db / 20.0))


def limit(audio, sr, ceiling_db=LIMITER_CEILING_DB):
    """Limiteur crête lookahead pour garantir le plafond true-peak."""
    board = Pedalboard([Limiter(threshold_db=ceiling_db, release_ms=100.0)])
    return board(audio.T.astype(np.float32), sr).T


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class DJApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DJ Audio Processor V10")
        self.root.geometry("900x740")

        self.ffmpeg = find_ffmpeg()
        self.src = ""
        self.dst = ""
        self.found_files = []
        self.skipped_files = []
        self.dropped_files = []   # fichiers glissés-déposés directement
        self.out_fmt = tk.StringVar(value="flac")
        self.mode = tk.StringVar(value="loudness")   # "loudness" ou "stems"
        self.stem_model = tk.StringVar(value="htdemucs_ft")
        self.stop_flag = threading.Event()
        self.worker = None

        # Cases à cocher stems
        self.stem_vars = {label: tk.BooleanVar(value=(label == "Vocal"))
                          for label in STEM_LABELS}

        top = tk.Frame(root)
        top.pack(fill="x", padx=8, pady=6)
        tk.Button(top, text="1. DOSSIER MUSIQUE",
                  command=self.set_src, bg="#dddddd").pack(side="left", padx=4, fill="x", expand=True)
        tk.Button(top, text="2. DOSSIER SORTIE",
                  command=self.set_dst, bg="#dddddd").pack(side="left", padx=4, fill="x", expand=True)

        dnd_text = ("↓ Glissez un dossier OU des fichiers audio ici ↓" if DND_AVAILABLE
                    else "(glisser-déposer indisponible — utilisez le bouton 1)")
        self.drop_zone = tk.Label(root, text=dnd_text, relief="ridge", bd=2,
                                  bg="#eef3f7" if DND_AVAILABLE else "#f0f0f0",
                                  fg="#333333" if DND_AVAILABLE else "#999999", height=2)
        self.drop_zone.pack(fill="x", padx=8, pady=2)
        if DND_AVAILABLE:
            try:
                self.drop_zone.drop_target_register(DND_FILES)
                self.drop_zone.dnd_bind("<<Drop>>", self._on_drop)
            except Exception:
                self.drop_zone.config(text="(glisser-déposer indisponible)",
                                      bg="#f0f0f0", fg="#999999")

        # --- Choix du mode ---
        mode_frame = tk.LabelFrame(root, text="Mode de traitement")
        mode_frame.pack(fill="x", padx=8, pady=4)
        tk.Radiobutton(mode_frame, text="Traitement loudness (LUFS + limiteur → TAP)",
                       variable=self.mode, value="loudness",
                       command=self._on_mode_change).pack(anchor="w", padx=6)
        tk.Radiobutton(mode_frame, text="Séparation en stems (Demucs)",
                       variable=self.mode, value="stems",
                       command=self._on_mode_change).pack(anchor="w", padx=6)

        # --- Panneau loudness (format) ---
        self.fmt_frame = tk.LabelFrame(root, text="Format de sortie (loudness)")
        self.fmt_frame.pack(fill="x", padx=8, pady=2)
        tk.Radiobutton(self.fmt_frame, text="FLAC (24 bits, sans perte)",
                       variable=self.out_fmt, value="flac").pack(side="left", padx=10)
        tk.Radiobutton(self.fmt_frame, text="MP3 320 kbps",
                       variable=self.out_fmt, value="mp3").pack(side="left", padx=10)

        # --- Panneau stems (cases + modèle) ---
        self.stem_frame = tk.LabelFrame(root, text="Stems à extraire (sortie WAV brut)")
        for label in STEM_LABELS:
            tk.Checkbutton(self.stem_frame, text=label,
                           variable=self.stem_vars[label]).pack(side="left", padx=8)
        model_sub = tk.Frame(self.stem_frame)
        model_sub.pack(side="left", padx=20)
        tk.Label(model_sub, text="Modèle :").pack(side="left")
        tk.Radiobutton(model_sub, text="Qualité max (ft)",
                       variable=self.stem_model, value="htdemucs_ft").pack(side="left")
        tk.Radiobutton(model_sub, text="Rapide",
                       variable=self.stem_model, value="htdemucs").pack(side="left")

        self.recursive = tk.BooleanVar(value=True)
        tk.Checkbutton(root, text="Recherche récursive (inclure les sous-dossiers)",
                       variable=self.recursive, command=self._refresh_scan).pack(anchor="w", padx=8)

        prog_frame = tk.Frame(root)
        prog_frame.pack(fill="x", padx=8, pady=4)
        self.current_lbl = tk.Label(prog_frame, text="En attente…", anchor="w")
        self.current_lbl.pack(fill="x")
        self.file_bar = ttk.Progressbar(prog_frame, mode="indeterminate")
        self.file_bar.pack(fill="x", pady=2)
        self.global_bar = ttk.Progressbar(prog_frame, mode="determinate")
        self.global_bar.pack(fill="x", pady=2)

        self.btn_launch = tk.Button(root, text="LANCER LE TRAITEMENT",
                                    command=self.toggle, bg="green", fg="white",
                                    font=("Arial", 12, "bold"))
        self.btn_launch.pack(fill="x", padx=8, pady=6)

        cols = tk.Frame(root)
        cols.pack(fill="both", expand=True, padx=8, pady=4)

        left = tk.LabelFrame(cols, text="À traiter")
        left.pack(side="left", fill="both", expand=True, padx=(0, 4))
        self.queue_list = tk.Listbox(left, activestyle="none")
        self.queue_list.pack(fill="both", expand=True, side="left")
        ql_scroll = tk.Scrollbar(left, command=self.queue_list.yview)
        ql_scroll.pack(side="right", fill="y")
        self.queue_list.config(yscrollcommand=ql_scroll.set)

        right = tk.LabelFrame(cols, text="Terminés")
        right.pack(side="left", fill="both", expand=True, padx=(4, 0))
        self.done_list = tk.Listbox(right, activestyle="none")
        self.done_list.pack(fill="both", expand=True, side="left")
        dl_scroll = tk.Scrollbar(right, command=self.done_list.yview)
        dl_scroll.pack(side="right", fill="y")
        self.done_list.config(yscrollcommand=dl_scroll.set)

        self._on_mode_change()   # affiche le bon panneau au démarrage

        if self.ffmpeg:
            self.current_lbl.config(text=f"FFmpeg OK — {self.ffmpeg}")
        else:
            self.current_lbl.config(text="!!! FFmpeg INTROUVABLE — voir popup")
            messagebox.showwarning(
                "FFmpeg manquant",
                "FFmpeg est introuvable.\n\nTéléchargez-le sur ffmpeg.org, "
                "puis ajoutez le dossier 'bin' au PATH Windows.")

    def ui(self, fn):
        self.root.after(0, fn)

    def _on_mode_change(self):
        """Affiche le panneau correspondant au mode choisi."""
        if self.mode.get() == "stems":
            self.fmt_frame.pack_forget()
            self.stem_frame.pack(fill="x", padx=8, pady=2, after=self.drop_zone)
        else:
            self.stem_frame.pack_forget()
            self.fmt_frame.pack(fill="x", padx=8, pady=2, after=self.drop_zone)

    def _parse_drop_paths(self, data):
        """
        Découpe la chaîne renvoyée par tkdnd en une liste de chemins.
        Windows entoure d'accolades les chemins contenant des espaces :
        '{C:/mon dossier/a.flac} C:/b.flac {C:/c d.flac}'.
        """
        paths, i, n = [], 0, len(data)
        while i < n:
            if data[i] == "{":
                j = data.find("}", i)
                if j == -1:
                    paths.append(data[i + 1:].strip())
                    break
                paths.append(data[i + 1:j])
                i = j + 1
            elif data[i].isspace():
                i += 1
            else:
                j = i
                while j < n and not data[j].isspace():
                    j += 1
                paths.append(data[i:j])
                i = j
        return [os.path.normpath(p) for p in paths if p.strip()]

    def _on_drop(self, event):
        paths = self._parse_drop_paths(event.data.strip())
        if not paths:
            return

        dirs = [p for p in paths if os.path.isdir(p)]
        files = [p for p in paths if os.path.isfile(p)
                 and p.lower().endswith(SUPPORTED_IN)]

        # Cas 1 : on a déposé un (ou plusieurs) dossier(s) -> ancien comportement
        # (on prend le premier dossier comme source scannée).
        if dirs and not files:
            self.src = dirs[0]
            self.dropped_files = []
            self.current_lbl.config(text=f"Source (glissée) : {self.src}")
            self._refresh_scan()
            return

        # Cas 2 : on a déposé des fichiers audio -> on traite juste ceux-là.
        if files:
            self.dropped_files = files
            # Dossier source = celui du 1er fichier (utile pour l'affichage seulement)
            self.src = os.path.dirname(files[0])
            self._refresh_scan()
            return

        messagebox.showwarning(
            "Glisser-déposer",
            "Déposez un DOSSIER ou des fichiers audio pris en charge.")

    def set_src(self):
        d = filedialog.askdirectory(title="Choisir le dossier contenant la musique")
        if not d:
            return
        self.src = os.path.normpath(d)
        self.dropped_files = []   # on repart sur un scan de dossier
        self.current_lbl.config(text=f"Source : {self.src}")
        self._refresh_scan()

    def set_dst(self):
        d = filedialog.askdirectory(title="Choisir le dossier de sortie")
        if d:
            self.dst = os.path.normpath(d)
            self.current_lbl.config(text=f"Destination : {self.dst}")

    def _scan_files(self):
        to_process, skipped = [], []

        # Si des fichiers ont été déposés directement, on ne scanne pas de
        # dossier : on traite exactement cette sélection.
        if self.dropped_files:
            walker = ((os.path.dirname(p), os.path.basename(p))
                      for p in self.dropped_files)
        elif self.src:
            if self.recursive.get():
                walker = ((dp, fn) for dp, _, fns in os.walk(self.src) for fn in fns)
            else:
                walker = ((self.src, fn) for fn in os.listdir(self.src)
                          if os.path.isfile(os.path.join(self.src, fn)))
        else:
            return to_process, skipped

        for dirpath, name in walker:
            if not name.lower().endswith(SUPPORTED_IN):
                continue
            full = os.path.join(dirpath, name)
            # En mode stems, on ne filtre pas les TAP (on peut vouloir séparer un fichier traité)
            if self.mode.get() == "loudness" and is_already_processed(full):
                skipped.append(full)
            else:
                to_process.append(full)
        return sorted(to_process), sorted(skipped)

    def _refresh_scan(self):
        if not self.src and not self.dropped_files:
            return
        self.found_files, self.skipped_files = self._scan_files()
        self.queue_list.delete(0, "end")
        self.done_list.delete(0, "end")
        for p in self.found_files:
            self.queue_list.insert("end", os.path.basename(p))
        for p in self.skipped_files:
            self.queue_list.insert("end", f"[déjà traité] {os.path.basename(p)}")
            self.queue_list.itemconfig("end", fg="#999999")
        origine = "fichiers déposés" if self.dropped_files else "dossier"
        self.current_lbl.config(
            text=f"[{origine}] {len(self.found_files)} à traiter — "
                 f"{len(self.skipped_files)} ignoré(s)")

    def toggle(self):
        if self.worker and self.worker.is_alive():
            self.stop_flag.set()
            self.btn_launch.config(text="ARRÊT EN COURS…", state="disabled")
            self.current_lbl.config(text="Arrêt demandé — fin du fichier en cours…")
            return
        if not self.ffmpeg:
            messagebox.showerror("FFmpeg manquant", "FFmpeg n'est pas installé.")
            return
        if not self.src or not self.dst:
            messagebox.showwarning("Attention", "Veuillez choisir les deux dossiers.")
            return
        if not self.found_files:
            messagebox.showwarning("Attention", "Aucun fichier audio à traiter.")
            return
        if self.mode.get() == "stems":
            chosen = [STEM_LABELS[l] for l, v in self.stem_vars.items() if v.get()]
            if not chosen:
                messagebox.showwarning("Attention",
                                       "Cochez au moins un stem à extraire.")
                return
        self.stop_flag.clear()
        self.btn_launch.config(text="STOP", bg="#c0392b")
        self.global_bar.config(maximum=len(self.found_files), value=0)
        self.worker = threading.Thread(target=self._process_all, daemon=True)
        self.worker.start()

    # --- Traitement loudness ---
    def _process_loudness(self, src_p, tmp_in, tmp_out):
        subprocess.run([self.ffmpeg, "-y", "-i", src_p, "-c:a", "pcm_f32le", tmp_in],
                       check=True, capture_output=True)
        audio, sr = sf.read(tmp_in, dtype="float32", always_2d=True)

        # Mesures AVANT traitement
        n_total = audio.shape[0] * audio.shape[1]
        peak_before = float(np.max(np.abs(audio))) if n_total else 0.0
        meter = pyln.Meter(sr)
        lufs_before = meter.integrated_loudness(audio)

        # Déclippage (renvoie le nombre d'échantillons corrigés)
        audio, fixed = declip(audio)

        # Normalisation loudness — on calcule le gain nous-mêmes pour le rapport
        if lufs_before == float("-inf"):
            gain_db = 0.0
        else:
            gain_db = TARGET_LUFS - lufs_before
            audio = audio * (10.0 ** (gain_db / 20.0))

        audio = limit(audio, sr)
        audio = np.clip(audio, -1.0, 1.0)

        # Mesures APRÈS traitement
        lufs_after = meter.integrated_loudness(audio)
        peak_after = float(np.max(np.abs(audio))) if n_total else 0.0

        stats = {
            "clip_fixed": fixed,
            "clip_pct": (100.0 * fixed / n_total) if n_total else 0.0,
            "lufs_before": lufs_before,
            "lufs_after": lufs_after,
            "gain_db": gain_db,
            "peak_before_db": 20.0 * np.log10(peak_before) if peak_before > 0 else float("-inf"),
            "peak_after_db": 20.0 * np.log10(peak_after) if peak_after > 0 else float("-inf"),
        }

        sf.write(tmp_out, audio, sr, subtype="FLOAT")
        args, ext = ENCODERS[self.out_fmt.get()]
        base = os.path.splitext(os.path.basename(src_p))[0]
        out_name = f"{base}{TAP_SUFFIX}.{ext}"
        dst_p = os.path.join(self.dst, out_name)

        # On récupère la pochette du fichier d'origine (2e entrée), qu'on
        # réinjecte dans le fichier traité. -map 0:a = l'audio retraité,
        # -map 1:v = l'image du fichier source. Si la source n'a pas de
        # pochette, FFmpeg échoue → on refait un encodage simple sans image.
        has_cover = self._has_cover(src_p)
        if has_cover:
            if ext == "mp3":
                cover_args = ["-map", "0:a", "-map", "1:v", *args,
                              "-c:v", "copy", "-id3v2_version", "3",
                              "-metadata:s:v", "title=Album cover",
                              "-metadata:s:v", "comment=Cover (front)"]
            else:  # flac
                cover_args = ["-map", "0:a", "-map", "1:v", *args,
                              "-c:v", "copy", "-disposition:v", "attached_pic"]
            try:
                subprocess.run(
                    [self.ffmpeg, "-y", "-i", tmp_out, "-i", src_p, *cover_args, dst_p],
                    check=True, capture_output=True)
                return out_name, stats
            except subprocess.CalledProcessError:
                pass  # repli sans pochette

        subprocess.run([self.ffmpeg, "-y", "-i", tmp_out, *args, dst_p],
                       check=True, capture_output=True)
        return out_name, stats

    def _has_cover(self, src_p):
        """Vrai si le fichier source contient une pochette (flux vidéo/image)."""
        try:
            r = subprocess.run(
                [self.ffmpeg, "-i", src_p], capture_output=True, text=True)
            # FFmpeg écrit les infos sur stderr ; on cherche un flux Video
            return "Video:" in r.stderr
        except Exception:
            return False

    # --- Séparation stems ---
    def _process_stems(self, src_p, chosen_stems):
        """
        Sépare src_p avec Demucs sur GPU, ne garde que les stems choisis,
        les range dans un sous-dossier 'Titre [stem1+stem2]/' au format WAV.
        Import de demucs différé (au 1er appel) pour ne pas ralentir le démarrage.
        """
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        from demucs.audio import AudioFile, save_audio

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = get_model(self.stem_model.get())
        model.to(device)
        model.eval()

        # Chargement du fichier au bon format pour Demucs
        wav = AudioFile(src_p).read(streams=0, samplerate=model.samplerate,
                                    channels=model.audio_channels)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / (ref.std() + 1e-8)

        with torch.no_grad():
            sources = apply_model(model, wav[None], device=device,
                                  progress=False)[0]
        sources = sources * ref.std() + ref.mean()

        # Ordre des stems tel que renvoyé par le modèle
        source_names = model.sources   # ['drums', 'bass', 'other', 'vocals']

        base = os.path.splitext(os.path.basename(src_p))[0]
        # Nom de dossier : "Titre [vocals+bass]"
        tag = "+".join(chosen_stems)
        out_dir = os.path.join(self.dst, safe_name(f"{base} [{tag}]"))
        os.makedirs(out_dir, exist_ok=True)

        written = []
        for name, source in zip(source_names, sources):
            if name not in chosen_stems:
                continue
            out_path = os.path.join(out_dir, f"{safe_name(base)}_{name}.wav")
            save_audio(source.cpu(), out_path, samplerate=model.samplerate)
            written.append(f"{name}.wav")
        return os.path.basename(out_dir), written

    @staticmethod
    def _fmt_db(v):
        """Formate une valeur en dB, gère le -inf (silence)."""
        if v == float("-inf"):
            return "silence"
        return f"{v:+.1f} dB"

    def _format_stats(self, s):
        """Met en forme les mesures d'un fichier loudness pour le rapport."""
        out = []

        # Déclippage
        if s["clip_fixed"] > 0:
            out.append(f"      • Clipping     : CORRIGÉ — "
                       f"{s['clip_fixed']} échantillon(s) restauré(s) "
                       f"({s['clip_pct']:.3f} % du fichier)")
        else:
            out.append("      • Clipping     : aucun (fichier propre)")

        # Loudness avant/après + sens de l'ajustement
        lb = s["lufs_before"]
        la = s["lufs_after"]
        g = s["gain_db"]
        if lb == float("-inf"):
            out.append("      • Loudness     : non mesurable (silence)")
        else:
            if g > 0.05:
                sens = f"son MONTÉ de {g:+.1f} dB"
            elif g < -0.05:
                sens = f"son BAISSÉ de {g:.1f} dB"
            else:
                sens = "quasi inchangé (déjà au bon niveau)"
            out.append(f"      • Loudness     : {lb:.1f} → {la:.1f} LUFS "
                       f"(cible {TARGET_LUFS}) — {sens}")

        # Crête avant/après
        out.append(f"      • Crête        : {self._fmt_db(s['peak_before_db'])} → "
                   f"{self._fmt_db(s['peak_after_db'])} "
                   f"(plafond {LIMITER_CEILING_DB} dB)")
        return out

    def _write_report(self, results, stopped):
        stamp = datetime.datetime.now()
        fname = f"rapport_TAP_{stamp:%Y%m%d_%H%M%S}.txt"
        path = os.path.join(self.dst, fname)
        mode = "Séparation stems" if self.mode.get() == "stems" else "Loudness TAP"
        lines = [
            "=" * 60,
            " RAPPORT DE TRAITEMENT — DJ Audio Processor",
            "=" * 60,
            f"Date            : {stamp:%Y-%m-%d %H:%M:%S}",
            f"Mode            : {mode}",
            f"Dossier source  : {self.src}",
            f"Dossier sortie  : {self.dst}",
        ]
        if self.mode.get() == "stems":
            chosen = [l for l, v in self.stem_vars.items() if v.get()]
            lines.append(f"Stems extraits  : {', '.join(chosen)}")
            lines.append(f"Modèle Demucs   : {self.stem_model.get()}")
        else:
            lines.append(f"Format          : {self.out_fmt.get().upper()}")
            lines.append(f"Cible loudness  : {TARGET_LUFS} LUFS")
            lines.append(f"Plafond limiteur: {LIMITER_CEILING_DB} dBTP")
        lines.append(f"Statut          : {'ARRÊTÉ avant la fin' if stopped else 'Terminé'}")
        lines.append("-" * 60)
        ok = [r for r in results if r[1] == "ok"]
        err = [r for r in results if r[1] == "err"]
        lines.append(f"TRAITÉS AVEC SUCCÈS ({len(ok)}) :")
        for name, _, detail, stats in ok:
            lines.append(f"  ✓ {detail}")
            if stats:  # mode loudness uniquement
                lines.extend(self._format_stats(stats))
        if err:
            lines.append("")
            lines.append(f"ERREURS ({len(err)}) :")
            for name, _, detail, stats in err:
                lines.append(f"  ✗ {name} — {detail}")
        lines.append("=" * 60)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            return fname
        except Exception:
            return None

    def _process_all(self):
        os.makedirs(self.dst, exist_ok=True)
        is_stems = self.mode.get() == "stems"
        chosen_stems = [STEM_LABELS[l] for l, v in self.stem_vars.items() if v.get()]

        tmp_dir = os.path.join(self.dst, ".tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_in = os.path.join(tmp_dir, "__in.wav")
        tmp_out = os.path.join(tmp_dir, "__out.wav")

        total = len(self.found_files)
        ok, errors, stopped = 0, 0, False
        results = []

        self.ui(lambda: self.file_bar.start(12))
        if is_stems:
            self.ui(lambda: self.current_lbl.config(
                text="Chargement du modèle Demucs (1er lancement = téléchargement)…"))

        for i, src_p in enumerate(self.found_files):
            if self.stop_flag.is_set():
                stopped = True
                break
            name = os.path.basename(src_p)
            self.ui(lambda n=name, idx=i: (
                self.current_lbl.config(text=f"Traitement : {n}"),
                self.queue_list.itemconfig(idx, fg="orange") if idx < self.queue_list.size() else None
            ))
            try:
                if is_stems:
                    folder, written = self._process_stems(src_p, chosen_stems)
                    ok += 1
                    detail = f"{folder} ({', '.join(written)})"
                    results.append((name, "ok", detail, None))
                    self.ui(lambda d=folder: self.done_list.insert("end", f"✓ {d}/"))
                else:
                    out_name, stats = self._process_loudness(src_p, tmp_in, tmp_out)
                    ok += 1
                    results.append((name, "ok", out_name, stats))
                    self.ui(lambda o=out_name: self.done_list.insert("end", f"✓ {o}"))
                self.ui(lambda idx=i: self.queue_list.itemconfig(idx, fg="green")
                        if idx < self.queue_list.size() else None)
            except subprocess.CalledProcessError:
                errors += 1
                results.append((name, "err", "décodage FFmpeg (fichier corrompu ?)", None))
                self.ui(lambda n=name: self.done_list.insert("end", f"✗ {n} (erreur FFmpeg)"))
                self.ui(lambda idx=i: self.queue_list.itemconfig(idx, fg="red")
                        if idx < self.queue_list.size() else None)
            except Exception as e:
                errors += 1
                results.append((name, "err", str(e), None))
                self.ui(lambda n=name, err=e: self.done_list.insert("end", f"✗ {n} ({err})"))
                self.ui(lambda idx=i: self.queue_list.itemconfig(idx, fg="red")
                        if idx < self.queue_list.size() else None)
            finally:
                for f in (tmp_in, tmp_out):
                    if os.path.exists(f):
                        try:
                            os.remove(f)
                        except OSError:
                            pass
                self.ui(lambda v=i + 1: self.global_bar.config(value=v))

        if os.path.isdir(tmp_dir) and not os.listdir(tmp_dir):
            os.rmdir(tmp_dir)

        report_name = self._write_report(results, stopped) if results else None

        self.ui(lambda: self.file_bar.stop())
        if stopped:
            msg = f"ARRÊTÉ : {ok} traité(s), {errors} erreur(s), reste {total - ok - errors}"
        else:
            msg = f"TERMINÉ : {ok} réussi(s), {errors} erreur(s)"
        if report_name:
            msg += f" — rapport : {report_name}"
        self.ui(lambda m=msg: self.current_lbl.config(text=m))
        self.ui(lambda: self.btn_launch.config(
            state="normal", text="LANCER LE TRAITEMENT", bg="green"))


if __name__ == "__main__":
    if DND_AVAILABLE:
        try:
            root = TkinterDnD.Tk()
        except Exception:
            root = tk.Tk()
    else:
        root = tk.Tk()
    DJApp(root)
    root.mainloop()