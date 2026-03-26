import os
import pickle

import numpy as np
import psutil
import torch
from tqdm import tqdm

from DataProcessing.find_and_load_patient_files import (
    find_patient_files,
    load_patient_data,
)
from DataProcessing.helper_code import get_num_locations, load_wav_file
from DataProcessing.label_extraction import get_murmur, get_outcome
from HumBugDB.LogMelSpecs.compute_LogMelSpecs import waveform_to_examples


def _ram():
    vm = psutil.virtual_memory()
    return f"RAM: {vm.used / 1024**3:.1f}/{vm.total / 1024**3:.1f} GB ({vm.percent}%)"


def calc_and_save_features(data_directory, spectrogram_directory, split, chunk_size=50):
    """Procesa pacientes en chunks y guarda a disco para evitar OOM de RAM."""
    murmur_classes = ["Present", "Unknown", "Absent"]
    outcome_classes = ["Abnormal", "Normal"]

    patient_files = find_patient_files(data_directory)
    num_patient_files = len(patient_files)
    print(f"[{split}] {num_patient_files} pacientes — {_ram()}")

    spec_chunks = []
    murmur_chunks = []
    outcome_chunks = []
    chunk_files = []

    for i in tqdm(range(num_patient_files), desc=f"Espectrogramas {split}"):
        current_patient_data = load_patient_data(patient_files[i])
        current_spectrograms = load_spectrograms(data_directory, current_patient_data)

        n_specs = sum(len(s) for s in current_spectrograms)

        current_murmur = np.zeros(len(murmur_classes), dtype=int)
        murmur = get_murmur(current_patient_data)
        if murmur in murmur_classes:
            current_murmur[murmur_classes.index(murmur)] = 1

        current_outcome = np.zeros(len(outcome_classes), dtype=int)
        outcome = get_outcome(current_patient_data)
        if outcome in outcome_classes:
            current_outcome[outcome_classes.index(outcome)] = 1

        for spec in current_spectrograms:
            for s in spec:
                spec_chunks.append(s)
        murmur_chunks.extend([current_murmur] * n_specs)
        outcome_chunks.extend([current_outcome] * n_specs)

        # Cada chunk_size pacientes, guardar a disco y limpiar RAM
        if (i + 1) % chunk_size == 0 or (i + 1) == num_patient_files:
            chunk_idx = len(chunk_files)
            chunk_path = os.path.join(spectrogram_directory, f"chunk_{split}_{chunk_idx}.pt")
            torch.save({
                "specs":    torch.cat(spec_chunks).unsqueeze(1),
                "murmurs":  torch.Tensor(np.array(murmur_chunks)),
                "outcomes": torch.Tensor(np.array(outcome_chunks)),
            }, chunk_path)
            chunk_files.append(chunk_path)
            spec_chunks = []
            murmur_chunks = []
            outcome_chunks = []
            print(f"[{split}] chunk {chunk_idx} guardado — {_ram()}")

    # Combinar chunks en tensores finales
    print(f"[{split}] Combinando {len(chunk_files)} chunks — {_ram()}")
    all_specs, all_murmurs, all_outcomes = [], [], []
    for chunk_path in chunk_files:
        chunk = torch.load(chunk_path)
        all_specs.append(chunk["specs"])
        all_murmurs.append(chunk["murmurs"])
        all_outcomes.append(chunk["outcomes"])
        os.remove(chunk_path)

    specs    = torch.cat(all_specs)
    murmurs  = torch.cat(all_murmurs)
    outcomes = torch.cat(all_outcomes)
    del all_specs, all_murmurs, all_outcomes

    print(f"[{split}] specs shape: {specs.shape} — {_ram()}")
    torch.save(specs,    os.path.join(spectrogram_directory, f"spec_{split}"))
    torch.save(murmurs,  os.path.join(spectrogram_directory, f"murmurs_{split}"))
    torch.save(outcomes, os.path.join(spectrogram_directory, f"outcomes_{split}"))
    del specs, murmurs, outcomes
    print(f"[{split}] guardado en disco y RAM liberada — {_ram()}")


def net_feature_loader(
    recalc_features, train_data_directory, test_data_directory, spectrogram_directory
):
    if not os.path.isdir(spectrogram_directory):
        os.makedirs(spectrogram_directory)

    if recalc_features:
        print(f"[net_feature_loader] Calculando TRAIN — {_ram()}")
        calc_and_save_features(train_data_directory, spectrogram_directory, "train")
        print(f"[net_feature_loader] Calculando TEST — {_ram()}")
        calc_and_save_features(test_data_directory, spectrogram_directory, "test")

    print(f"[net_feature_loader] Cargando desde disco — {_ram()}")
    spectrograms_train = torch.load(os.path.join(spectrogram_directory, "spec_train"))
    murmurs_train      = torch.load(os.path.join(spectrogram_directory, "murmurs_train"))
    outcomes_train     = torch.load(os.path.join(spectrogram_directory, "outcomes_train"))
    spectrograms_test  = torch.load(os.path.join(spectrogram_directory, "spec_test"))
    murmurs_test       = torch.load(os.path.join(spectrogram_directory, "murmurs_test"))
    outcomes_test      = torch.load(os.path.join(spectrogram_directory, "outcomes_test"))
    print(f"[net_feature_loader] Listo — {_ram()}")

    return (
        spectrograms_train,
        murmurs_train,
        outcomes_train,
        spectrograms_test,
        murmurs_test,
        outcomes_test,
    )


def patient_feature_loader(recalc_features, data_directory, output_directory):
    if recalc_features == "True":
        spectrograms, murmurs, outcomes = calc_patient_features(data_directory)
        with open(output_directory + "spectrograms", "wb") as fp:
            pickle.dump(spectrograms, fp)
        with open(output_directory + "murmurs", "wb") as fp:
            pickle.dump(murmurs, fp)
        with open(output_directory + "outcomes", "wb") as fp:
            pickle.dump(outcomes, fp)
    else:
        with open(output_directory + "spectrograms", "rb") as fp:
            spectrograms = pickle.load(fp)
        with open(output_directory + "murmurs", "rb") as fp:
            murmurs = pickle.load(fp)
        with open(output_directory + "outcomes", "rb") as fp:
            outcomes = pickle.load(fp)

    return spectrograms, murmurs, outcomes


def load_spectrograms(data_directory, data):
    num_locations = get_num_locations(data)
    recording_information = data.split("\n")[1 : num_locations + 1]

    mel_specs = list()
    for i in range(num_locations):
        entries = recording_information[i].split(" ")
        recording_file = entries[2]
        filename = os.path.join(data_directory, recording_file)
        recording, frequency = load_wav_file(filename)
        recording = recording / 32768
        mel_spec = waveform_to_examples(recording, frequency)
        mel_specs.append(mel_spec)
    return mel_specs


def load_spectrograms_yaseen(file_path):
    mel_specs = list()
    recording, frequency = load_wav_file(file_path)
    recording = recording / 32768
    mel_spec = waveform_to_examples(recording, frequency)
    mel_specs.append(mel_spec)
    return mel_specs


def list_wav_files(data_directory):
    wav_files = []
    subfolder_names = []
    for root, dirs, files in os.walk(data_directory):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
                subfolder_names.append(os.path.basename(root))
    return wav_files, subfolder_names


def calc_patient_features(data_directory):
    """Mantenida para compatibilidad con patient_feature_loader y dbres.py."""
    print(f"[calc_patient_features] Inicio — {_ram()}")

    if "yaseen" in data_directory:
        outcome_classes = [f.name for f in os.scandir(data_directory) if f.is_dir()]
        murmur_classes = outcome_classes
        num_murmur_classes = len(murmur_classes)
        num_outcome_classes = len(outcome_classes)
        patient_files, labels = list_wav_files(data_directory)
        num_patient_files = len(patient_files)
        spectrograms = list()
        murmurs = list()
        outcomes = list()
        for label, file_path in zip(labels, patient_files):
            current_outcome = np.zeros(num_outcome_classes, dtype=int)
            outcome = label
            if outcome in outcome_classes:
                j = outcome_classes.index(outcome)
                current_outcome[j] = 1
            outcomes.append(current_outcome)
            murmurs = outcomes
            current_spectrograms = load_spectrograms_yaseen(file_path)
            spectrograms.append(current_spectrograms)
    else:
        murmur_classes = ["Present", "Unknown", "Absent"]
        num_murmur_classes = len(murmur_classes)
        outcome_classes = ["Abnormal", "Normal"]
        num_outcome_classes = len(outcome_classes)
        patient_files = find_patient_files(data_directory)
        num_patient_files = len(patient_files)
        print(f"[calc_patient_features] {num_patient_files} pacientes")
        spectrograms = list()
        murmurs = list()
        outcomes = list()
        for i in tqdm(range(num_patient_files), desc="Cargando espectrogramas"):
            current_patient_data = load_patient_data(patient_files[i])
            current_spectrograms = load_spectrograms(data_directory, current_patient_data)
            spectrograms.append(current_spectrograms)
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
            if i % 100 == 0 and i > 0:
                print(f"[calc_patient_features] {i}/{num_patient_files} — {_ram()}")

    print(f"[calc_patient_features] Completo — {_ram()}")
    return spectrograms, murmurs, outcomes