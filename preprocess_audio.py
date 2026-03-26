"""
export_split.py
---------------
Genera patient_split.json con los IDs de paciente por fold (train/val/test).

Este script NO procesa audio ni aplica filtros — solo divide los pacientes
y exporta el JSON que luego usa data_splits_patched.py de PathToMyHeart
para forzar el mismo split.

Uso:
    python src/export_split.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ==============================================================================
# === CONFIGURACIÓN ===
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "processed"

# Debe coincidir exactamente con los valores usados en preprocess_audio.py
REQUIRED_SITES = {"AV", "PV", "TV", "MV"}
PRESENT_NAMES  = {"presente", "present", "with_murmur", "murmur", "yes", "sano_con_soplo"}
RANDOM_SEED    = 42

# ==============================================================================
# === MAIN ===
# ==============================================================================
if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DATA_DIR.exists():
        sys.exit(f"ERROR: No se encuentra {RAW_DATA_DIR}")

    # 1. Indexar pacientes únicos desde la estructura raw/
    all_files = []
    for valve in REQUIRED_SITES:
        valve_dir = RAW_DATA_DIR / valve
        if not valve_dir.exists():
            continue
        for status_dir in valve_dir.iterdir():
            if not status_dir.is_dir():
                continue
            for wav in status_dir.glob("*.wav"):
                pid = wav.stem.split('_')[0]
                all_files.append({'pid': pid, 'path': wav})

    df_files = pd.DataFrame(all_files)
    if df_files.empty:
        sys.exit("ERROR: No se encontraron archivos .wav en data/raw/")

    unique_patients = df_files['pid'].unique()
    print(f"Pacientes únicos encontrados: {len(unique_patients)}")

    # 2. Etiqueta por paciente (1 = murmur presente, 0 = ausente)
    patient_labels = []
    for pid in unique_patients:
        pid_paths = df_files[df_files['pid'] == pid]['path'].tolist()
        is_sick   = any(
            any(x in str(p).lower() for x in PRESENT_NAMES)
            for p in pid_paths
        )
        patient_labels.append(1 if is_sick else 0)

    # 3. Split idéntico al de preprocess_audio.py
    p_train, p_temp, l_train, l_temp = train_test_split(
        unique_patients, patient_labels,
        test_size=0.40, stratify=patient_labels, random_state=RANDOM_SEED
    )
    p_val, p_test, _, _ = train_test_split(
        p_temp, l_temp,
        test_size=0.625, stratify=l_temp, random_state=RANDOM_SEED
    )

    print(f"Split generado:")
    print(f"  Train: {len(p_train)} pacientes")
    print(f"  Val:   {len(p_val)} pacientes")
    print(f"  Test:  {len(p_test)} pacientes")

    # 4. Exportar JSON
    split_export = {
        'train': [str(p) for p in p_train],
        'val':   [str(p) for p in p_val],
        'test':  [str(p) for p in p_test],
    }
    out_path = OUTPUT_DIR / "patient_split.json"
    with open(out_path, 'w') as f:
        json.dump(split_export, f, indent=2)

    print(f"\n✅ Exportado en: {out_path}")