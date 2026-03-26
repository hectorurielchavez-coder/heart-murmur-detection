import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from tqdm import tqdm

from DataProcessing.find_and_load_patient_files import (
    find_patient_files,
    load_patient_data,
)
from DataProcessing.label_extraction import get_murmur, get_outcome
from DataProcessing.XGBoost_features.metadata import get_metadata


# ==============================================================================
# === PARCHE: SPLIT FORZADO DESDE JSON EXTERNO ===
# Reemplaza el train_test_split interno de PathToMyHeart por los IDs
# del proyecto propio (random_state=42, split por paciente, estratificado).
#
# Para desactivar el parche y usar el comportamiento original de PathToMyHeart,
# cambia FORCE_EXTERNAL_SPLIT a False.
# ==============================================================================
FORCE_EXTERNAL_SPLIT = True
EXTERNAL_SPLIT_JSON  = r"C:\Escuela\8\hackaton\Cinc\data\processed\patient_split.json"


def load_external_split(json_path: str):
    """Carga los IDs de paciente desde el JSON exportado por preprocess_audio.py."""
    with open(json_path, "r") as f:
        split = json.load(f)
    # Aseguramos que todos los IDs sean strings sin espacios
    return (
        [str(x).strip() for x in split["train"]],
        [str(x).strip() for x in split["val"]],
        [str(x).strip() for x in split["test"]],
    )


def stratified_test_vali_split(
    stratified_features: list,
    data_directory: str,
    stratified_directory: str,
    test_size: float,
    vali_size: float,
    random_states: list = [42],
    cv: bool = False,
    n_splits: int = 10,
    stratified_cv: bool = False,
):
    # Check if stratified_directory directory exists, otherwise create it.
    if not os.path.exists(stratified_directory):
        os.makedirs(stratified_directory)

    # Get metadata
    patient_files = find_patient_files(data_directory)
    num_patient_files = len(patient_files)
    murmur_classes = ["Present", "Unknown", "Absent"]
    num_murmur_classes = len(murmur_classes)
    outcome_classes = ["Abnormal", "Normal"]
    num_outcome_classes = len(outcome_classes)
    features = list()
    murmurs = list()
    outcomes = list()
    for i in tqdm(range(num_patient_files)):
        current_patient_data = load_patient_data(patient_files[i])
        current_features = get_metadata(current_patient_data)
        current_features = np.insert(
            current_features, 0, current_patient_data.split(" ")[0]
        )
        current_features = np.insert(
            current_features, 1, current_patient_data.split(" ")[2][:-3]
        )
        features.append(current_features)
        current_murmur = np.zeros(num_murmur_classes, dtype=int)
        murmur = get_murmur(current_patient_data)
        if murmur in murmur_classes:
            j = murmur_classes.index(murmur)
            current_murmur[j] = 1
        murmurs.append(current_murmur)
        current_outcome = np.zeros(num_outcome_classes, dtype=int)
        outcome = get_outcome(current_patient_data)
        if outcome in outcome_classes:
            j = outcome_classes.index(outcome)
            current_outcome[j] = 1
        outcomes.append(current_outcome)

    features = np.vstack(features)
    murmurs = np.vstack(murmurs)
    outcomes = np.vstack(outcomes)

    features_pd = pd.DataFrame(
        features,
        columns=["id", "hz", "age", "female", "male", "height", "weight", "is_pregnant"],
    )
    murmurs_pd = pd.DataFrame(murmurs, columns=murmur_classes)
    outcomes_pd = pd.DataFrame(outcomes, columns=outcome_classes)
    complete_pd = pd.concat([features_pd, murmurs_pd, outcomes_pd], axis=1)
    complete_pd["id"] = complete_pd["id"].astype(int).astype(str)
    complete_pd["stratify_column"] = (
        complete_pd[stratified_features].astype(str).agg("-".join, axis=1)
    )

    # ==========================================================================
    # PARCHE: si FORCE_EXTERNAL_SPLIT está activo, ignoramos train_test_split
    # y usamos directamente los IDs del JSON externo.
    # ==========================================================================
    if FORCE_EXTERNAL_SPLIT:
        print(f"\n⚡ PARCHE ACTIVO: usando split externo desde:\n   {EXTERNAL_SPLIT_JSON}\n")
        train_ids, val_ids, test_ids = load_external_split(EXTERNAL_SPLIT_JSON)

        # Verificar que los IDs del JSON existen en este dataset
        all_ids = set(complete_pd["id"].tolist())
        missing_train = [x for x in train_ids if x not in all_ids]
        missing_val   = [x for x in val_ids   if x not in all_ids]
        missing_test  = [x for x in test_ids  if x not in all_ids]
        if missing_train or missing_val or missing_test:
            print(f"⚠ IDs no encontrados en training_data:")
            print(f"  train: {missing_train[:5]}{'...' if len(missing_train)>5 else ''}")
            print(f"  val:   {missing_val[:5]}{'...' if len(missing_val)>5 else ''}")
            print(f"  test:  {missing_test[:5]}{'...' if len(missing_test)>5 else ''}")

        complete_pd_train = complete_pd[complete_pd["id"].isin(train_ids)]
        complete_pd_val   = complete_pd[complete_pd["id"].isin(val_ids)]
        complete_pd_test  = complete_pd[complete_pd["id"].isin(test_ids)]

        print(f"Split resultante:")
        print(f"  Train: {len(complete_pd_train)} pacientes")
        print(f"  Val:   {len(complete_pd_val)} pacientes")
        print(f"  Test:  {len(complete_pd_test)} pacientes")

        # Guardar
        cnum = "seed_42_external"
        save_folder = os.path.join(stratified_directory, "cv_False", cnum)
        os.makedirs(os.path.join(save_folder, "train_data"), exist_ok=True)
        os.makedirs(os.path.join(save_folder, "vali_data"),  exist_ok=True)
        os.makedirs(os.path.join(save_folder, "test_data"),  exist_ok=True)

        with open(os.path.join(save_folder, "split_details.txt"), "w") as f:
            f.write("Split forzado desde JSON externo (random_state=42).\n")
            f.write(f"Fuente: {EXTERNAL_SPLIT_JSON}\n")
            f.write(f"Train: {len(complete_pd_train)} | Val: {len(complete_pd_val)} | Test: {len(complete_pd_test)}\n")

        for pid in complete_pd_train["id"]:
            copy_files(data_directory, pid, os.path.join(save_folder, "train_data/"))
        for pid in complete_pd_val["id"]:
            copy_files(data_directory, pid, os.path.join(save_folder, "vali_data/"))
        for pid in complete_pd_test["id"]:
            copy_files(data_directory, pid, os.path.join(save_folder, "test_data/"))

        print(f"\n✅ Split guardado en: {save_folder}")
        return  # <-- salimos, no ejecutamos el código original

    # ==========================================================================
    # CÓDIGO ORIGINAL (sin cambios) — solo se ejecuta si FORCE_EXTERNAL_SPLIT=False
    # ==========================================================================
    complete_pd_train_list = list()
    complete_pd_val_list = list()
    complete_pd_test_list = list()
    cnums = list()
    if cv:
        if stratified_cv:
            print("Performing stratified cross-validation")
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            for i, (train_index, test_index) in enumerate(
                skf.split(complete_pd, complete_pd["stratify_column"])
            ):
                cnums.append(f"split_{i}")
                complete_pd_train, complete_pd_test = complete_pd.iloc[train_index], complete_pd.iloc[test_index]
                vali_split = vali_size / (1 - test_size)
                complete_pd_train, complete_pd_val = train_test_split(
                    complete_pd_train,
                    test_size=vali_split,
                    random_state=42,
                    stratify=complete_pd_train["stratify_column"],
                )
                complete_pd_train_list.append(complete_pd_train)
                complete_pd_val_list.append(complete_pd_val)
                complete_pd_test_list.append(complete_pd_test)
        else:
            print("Performing random cross-validation")
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            for i, (train_index, test_index) in enumerate(kf.split(complete_pd)):
                cnums.append(f"split_{i}")
                complete_pd_train, complete_pd_test = complete_pd.iloc[train_index], complete_pd.iloc[test_index]
                vali_split = vali_size / (1 - test_size)
                complete_pd_train, complete_pd_val = train_test_split(
                    complete_pd_train,
                    test_size=vali_split,
                    random_state=42,
                )
                complete_pd_train_list.append(complete_pd_train)
                complete_pd_val_list.append(complete_pd_val)
                complete_pd_test_list.append(complete_pd_test)
    else:
        print("Performing statified split")
        for random_state in random_states:
            cnums.append(f"seed_{random_state}")
            complete_pd_train, complete_pd_test = train_test_split(
                complete_pd,
                test_size=test_size,
                random_state=random_state,
                stratify=complete_pd["stratify_column"],
            )
            vali_split = vali_size / (1 - test_size)
            complete_pd_train, complete_pd_val = train_test_split(
                complete_pd_train,
                test_size=vali_split,
                random_state=random_state + 1,
                stratify=complete_pd_train["stratify_column"],
            )
            complete_pd_train_list.append(complete_pd_train)
            complete_pd_val_list.append(complete_pd_val)
            complete_pd_test_list.append(complete_pd_test)

    for cnum, complete_pd_train, complete_pd_val, complete_pd_test in zip(
        cnums, complete_pd_train_list, complete_pd_val_list, complete_pd_test_list
    ):
        print(f"Saving split {cnum} with cv {cv} from {len(cnums)} splits...")
        if cv:
            save_folder = os.path.join(stratified_directory, f"cv_{cv}_stratified_{stratified_cv}", cnum)
        else:
            save_folder = os.path.join(stratified_directory, f"cv_{cv}", cnum)
        os.makedirs(os.path.join(save_folder, "train_data"))
        os.makedirs(os.path.join(save_folder, "vali_data"))
        os.makedirs(os.path.join(save_folder, "test_data"))
        with open(os.path.join(save_folder, "split_details.txt"), "w") as text_file:
            text_file.write("This data split is stratified over the following features: \n")
            for feature in stratified_features:
                text_file.write(feature + ", ")
        for f in complete_pd_train["id"]:
            copy_files(data_directory, f, os.path.join(save_folder, "train_data/"))
        for f in complete_pd_val["id"]:
            copy_files(data_directory, f, os.path.join(save_folder, "vali_data/"))
        for f in complete_pd_test["id"]:
            copy_files(data_directory, f, os.path.join(save_folder, "test_data/"))


def copy_files(data_directory: str, ident: str, stratified_directory: str) -> None:
    files = os.listdir(data_directory)
    for f in files:
        if f.startswith(ident):
            _ = shutil.copy(os.path.join(data_directory, f), stratified_directory)


if __name__ == "__main__":

    print("---------------- Starting data_splits.py to split the data ----------------")

    parser = argparse.ArgumentParser(prog="StratifiedDataSplit")
    parser.add_argument(
        "--data_directory",
        type=str,
        default=r"C:\Escuela\8\hackaton\Cinc\data\training_data",
        help="Carpeta con los datos originales de PhysioNet.",
    )
    parser.add_argument(
        "--stratified_directory",
        type=str,
        default=r"C:\Escuela\8\hackaton\path\data\splits",
        help="Carpeta donde se guardarán los splits.",
    )
    parser.add_argument("--vali_size",     type=float, default=0.15)
    parser.add_argument("--test_size",     type=float, default=0.25)
    parser.add_argument("--cv",            type=bool,  default=False)
    parser.add_argument("--stratified_cv", type=bool,  default=False)
    args = parser.parse_args()

    stratified_features = ["Normal", "Abnormal", "Absent", "Present", "Unknown"]
    stratified_test_vali_split(stratified_features, **vars(args))