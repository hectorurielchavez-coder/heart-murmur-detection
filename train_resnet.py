import argparse
import os
import torch

from Config import hyperparameters
from DataProcessing.net_feature_extractor import net_feature_loader
from HumBugDB.runTorch import ResnetDropoutFull as ResnetDropoutBinary
from HumBugDB.runTorch import ResnetFull as ResnetBinary
from HumBugDB.runTorch import train_model as train_model_binary
from HumBugDB.runTorchMultiClass import ResnetDropoutFull as ResnetDropoutMulti
from HumBugDB.runTorchMultiClass import ResnetFull as ResnetMulti
from HumBugDB.runTorchMultiClass import train_model as train_model_multi


def create_model(model_name, num_classes, bayesian):
    print(f"\n[DEBUG] create_model — clases: {num_classes}, bayesian: {bayesian}")
    if model_name == "resnet50":
        print("[DEBUG] Modelo: ResNet50 sin dropout")
        if num_classes == 2:
            model = ResnetBinary()
            training = train_model_binary
        else:
            model = ResnetMulti(num_classes)
            training = train_model_multi
    elif model_name == "resnet50dropout":
        print(f"[DEBUG] Modelo: ResNet50Dropout bayesian={bayesian}")
        if num_classes == 2:
            print("[DEBUG] 🔀 Rama BINARIA (runTorch.py)")
            model = ResnetDropoutBinary(dropout=hyperparameters.dropout, bayesian=bayesian)
            training = train_model_binary
        else:
            print("[DEBUG] 🔀 Rama MULTICLASE (runTorchMultiClass.py)")
            model = ResnetDropoutMulti(n_classes=num_classes, bayesian=bayesian)
            training = train_model_multi
    else:
        raise NotImplementedError("Only implemented resnet50 and resnet50dropout")

    print("[DEBUG] ✅ Modelo instanciado en RAM correctamente.")
    return model, training


def make_binary_labels(murmur_path, classes_name, out_path):
    """
    Genera etiquetas binarias desde los murmurs originales (3 clases)
    y las guarda en disco — sin mantener tensores grandes en RAM.
    """
    print(f"[DEBUG] Generando etiquetas binarias para '{classes_name}'...")
    murmurs = torch.load(murmur_path, map_location="cpu", weights_only=True)
    knowledge = torch.zeros((murmurs.shape[0], 2))

    if classes_name == "murmur_binary":
        for i in range(len(murmurs)):
            if torch.argmax(murmurs[i]) == 0 or torch.argmax(murmurs[i]) == 1:
                knowledge[i, 0] = 1
            else:
                knowledge[i, 1] = 1
    elif classes_name == "binary_present":
        for i in range(len(murmurs)):
            if torch.argmax(murmurs[i]) == 1 or torch.argmax(murmurs[i]) == 2:
                knowledge[i, 0] = 1
            else:
                knowledge[i, 1] = 1
    elif classes_name == "binary_unknown":
        for i in range(len(murmurs)):
            if torch.argmax(murmurs[i]) == 0 or torch.argmax(murmurs[i]) == 2:
                knowledge[i, 1] = 1
            else:
                knowledge[i, 0] = 1

    torch.save(knowledge, out_path)
    print(f"[DEBUG] ✅ Etiquetas guardadas en: {out_path}")
    del murmurs, knowledge


def run_model_training(
    recalc_features,
    train_data_directory,
    vali_data_directory,
    spectrogram_directory,
    model_name,
    model_label,
    model_dir,
    classes_name,
    bayesian,
    weights,
):
    print(f"\n[DEBUG] >>> Iniciando run_model_training — objetivo: {classes_name} <<<")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[DEBUG] Dispositivo: {device}")

    # ── Calcular/cargar espectrogramas ───────────────────────────────────────
    # net_feature_loader guarda los tensores en disco y retorna None
    # Los datos NO se cargan en RAM aquí — se pasan como paths al DataLoader
    net_feature_loader(
        recalc_features,
        train_data_directory,
        vali_data_directory,
        spectrogram_directory,
    )

    # ── Paths a disco ────────────────────────────────────────────────────────
    spec_train    = os.path.join(spectrogram_directory, "spec_train")
    spec_test     = os.path.join(spectrogram_directory, "spec_test")
    murmur_train  = os.path.join(spectrogram_directory, "murmurs_train")
    murmur_test   = os.path.join(spectrogram_directory, "murmurs_test")
    outcome_train = os.path.join(spectrogram_directory, "outcomes_train")
    outcome_test  = os.path.join(spectrogram_directory, "outcomes_test")

    print(f"[DEBUG] Archivos en disco verificados:")
    for p in [spec_train, spec_test, murmur_train, murmur_test]:
        print(f"  {'✅' if os.path.exists(p) else '❌'} {p}")

    # ── Asignar paths de etiquetas según tarea ───────────────────────────────
    print(f"\n[DEBUG] Asignando etiquetas para: {classes_name}")
    sampler = None

    if classes_name == "murmur":
        y_train_path = murmur_train
        y_test_path  = murmur_test
        num_classes  = 3

    elif classes_name == "outcome_binary":
        y_train_path = outcome_train
        y_test_path  = outcome_test
        num_classes  = 2

    elif classes_name in ("murmur_binary", "binary_present", "binary_unknown"):
        y_train_path = os.path.join(spectrogram_directory, f"y_train_{classes_name}.pt")
        y_test_path  = os.path.join(spectrogram_directory, f"y_test_{classes_name}.pt")
        make_binary_labels(murmur_train, classes_name, y_train_path)
        make_binary_labels(murmur_test,  classes_name, y_test_path)
        num_classes  = 2
        sampler      = True if classes_name == "binary_unknown" else None

    else:
        raise ValueError(f"classes_name '{classes_name}' no reconocido.")

    print(f"[DEBUG] y_train_path: {y_train_path}")
    print(f"[DEBUG] y_test_path:  {y_test_path}")

    # ── Crear modelo ─────────────────────────────────────────────────────────
    print("\n[DEBUG] Llamando a create_model...")
    model, training = create_model(model_name, num_classes, bayesian)

    # ── Entrenar ─────────────────────────────────────────────────────────────
    kwargs = dict(
        clas_weight=weights,
        x_val=spec_test,
        y_val=y_test_path,
        model=model,
        model_name=model_label,
        model_dir=model_dir,
    )
    if sampler:
        kwargs["sampler"] = sampler

    print(f"\n[DEBUG] 🚀 Entrando a training() para {classes_name}...")
    model = training(spec_train, y_train_path, **kwargs)

    # ── Liberar VRAM ─────────────────────────────────────────────────────────
    del model
    torch.cuda.empty_cache()
    print(f"[DEBUG] ✅ Entrenamiento de '{classes_name}' completo. VRAM liberada.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="TrainResNet")
    parser.add_argument("--recalc_features", action="store_true")
    parser.add_argument("--no-recalc_features", dest="recalc_features", action="store_false")
    parser.set_defaults(recalc_features=True)
    parser.add_argument("--train_data_directory", type=str, default="data/stratified_data/train_data")
    parser.add_argument("--vali_data_directory",  type=str, default="data/stratified_data/vali_data")
    parser.add_argument("--spectrogram_directory", type=str, default="data/spectrograms")
    parser.add_argument("--model_name", type=str, choices=["resnet50", "resnet50dropout"], default="resnet50dropout")
    parser.add_argument("--model_label", type=str, default="ResNetDropout")
    parser.add_argument("--model_dir",   type=str, default="data/models")
    parser.add_argument(
        "--classes_name", type=str,
        choices=["murmur", "outcome_binary", "murmur_binary", "binary_present", "binary_unknown"],
        default="murmur",
    )
    parser.add_argument("--disable-bayesian", dest="bayesian", action="store_false", default=True)
    parser.add_argument("--weights_str", type=str, default=None)

    args = parser.parse_args()

    weights = None
    if args.weights_str:
        weights = [int(x) for x in args.weights_str.split(",")]
    vars(args).popitem()

    print("---------------- Starting train_resnet.py ----------------")
    print(f"---------------- Datos desde: {args.train_data_directory}")

    run_model_training(**vars(args), weights=weights)