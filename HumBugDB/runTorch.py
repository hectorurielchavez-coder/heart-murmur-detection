import os

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm

from Config import hyperparameters
from HumBugDB.ResNetDropoutSource import resnet50dropout
from HumBugDB.ResNetSource import resnet50


class ResnetFull(nn.Module):
    def __init__(self):
        super(ResnetFull, self).__init__()
        self.resnet = resnet50(pretrained=hyperparameters.pretrained)
        self.n_channels = 3
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc1 = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.resnet(x).squeeze(-1).squeeze(-1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x


class ResnetDropoutFull(nn.Module):
    def __init__(self, dropout=0.2, bayesian=True):
        super(ResnetDropoutFull, self).__init__()
        self.dropout = dropout
        self.bayesian = bayesian
        self.resnet = resnet50dropout(
            pretrained=hyperparameters.pretrained, dropout_p=self.dropout, bayesian=bayesian
        )
        self.n_channels = 3
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc1 = nn.Linear(2048, 1)

    def forward(self, x):
        training = True if self.bayesian else self.training
        x = self.resnet(x).squeeze(-1).squeeze(-1)
        x = self.fc1(F.dropout(x, p=self.dropout, training=training))
        x = torch.sigmoid(x)
        return x


class DiskDataset(Dataset):
    """Lee specs y labels desde disco — no acumula todo en RAM."""
    def __init__(self, spec_path, label_path):
        print(f"[DEBUG] Mapeando dataset desde disco:")
        print(f"[DEBUG]   specs:  {spec_path}")
        print(f"[DEBUG]   labels: {label_path}")
        self.specs  = torch.load(spec_path,  map_location="cpu", weights_only=True)
        self.labels = torch.load(label_path, map_location="cpu", weights_only=True).float()
        print(f"[DEBUG] Dataset mapeado — {len(self.labels)} muestras, shape: {self.specs.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.specs[idx].float(), self.labels[idx]


def make_weights_for_balanced_classes(dataset, nclasses):
    count = [0] * nclasses
    labels = dataset.labels if isinstance(dataset, DiskDataset) else [item[1] for item in dataset]
    for label in labels:
        count[torch.argmax(label)] += 1
    N = float(sum(count))
    weight_per_class = [N / c if c > 0 else 0.0 for c in count]
    return [weight_per_class[torch.argmax(label)] for label in labels]


def build_dataloader(
    x_train, y_train, x_val=None, y_val=None, shuffle=True, sampler=None
):
    print("[DEBUG] Construyendo DataLoaders...")

    if isinstance(x_train, str):
        train_dataset = DiskDataset(x_train, y_train)
    else:
        train_dataset = TensorDataset(
            x_train.clone().detach().float(),
            y_train.clone().detach().float()
        )

    if sampler is None:
        train_loader = DataLoader(
            train_dataset, batch_size=hyperparameters.batch_size,
            shuffle=shuffle, num_workers=0, pin_memory=False
        )
    else:
        weights = torch.DoubleTensor(make_weights_for_balanced_classes(train_dataset, 2))
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        train_loader = DataLoader(
            train_dataset, batch_size=hyperparameters.batch_size,
            sampler=sampler, num_workers=0, pin_memory=False
        )

    if x_val is not None:
        if isinstance(x_val, str):
            val_dataset = DiskDataset(x_val, y_val)
        else:
            val_dataset = TensorDataset(
                x_val.clone().detach().float(),
                y_val.clone().detach().float()
            )
        val_loader = DataLoader(
            val_dataset, batch_size=hyperparameters.batch_size,
            shuffle=False, num_workers=0, pin_memory=False
        )
        print("[DEBUG] DataLoaders listos (train + val).")
        return train_loader, val_loader

    print("[DEBUG] DataLoaders listos (solo train).")
    return train_loader


def train_model(
    x_train,
    y_train,
    clas_weight=None,
    x_val=None,
    y_val=None,
    model=ResnetDropoutFull(),
    model_name="test",
    model_dir="models",
    sampler=None,
):
    print("\n[DEBUG] >>> Iniciando train_model() <<<")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    if x_val is not None:
        train_loader, val_loader = build_dataloader(x_train, y_train, x_val, y_val, sampler=sampler)
    else:
        train_loader = build_dataloader(x_train, y_train, sampler=sampler)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[DEBUG] Dispositivo: {device}")

    if torch.cuda.device_count() > 1:
        print("[DEBUG] Usando data parallel")
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    model = model.to(device)
    print("[DEBUG] Modelo transferido a GPU.")

    criterion = nn.BCELoss()
    optimiser = optim.Adam(model.parameters(), lr=hyperparameters.lr)

    best_val_acc   = -np.inf
    best_train_acc = -np.inf
    e_saved = None
    output_string_to_save = ""
    overrun_counter = 0

    print("[DEBUG] Iniciando loop de epocas...")
    for e in range(hyperparameters.epochs):
        start_time = time.time()
        train_loss = 0.0
        model.train()
        all_y, all_y_pred = [], []

        print(f"[DEBUG] Epoca {e} — esperando primer batch...")
        for batch_i, inputs in enumerate(train_loader):
            if batch_i == 0:
                print(f"[DEBUG] Batch 0 cargado — shape: {inputs[0].shape}. Forward pass...")

            x = inputs[0].repeat(1, 3, 1, 1).to(device)
            y = torch.argmax(inputs[1], dim=1, keepdim=True).float().to(device)

            optimiser.zero_grad()
            y_pred = model(x)

            if clas_weight is not None:
                criterion.weight = (clas_weight[1] - clas_weight[0]) * y + clas_weight[0]
                loss = criterion.forward(y_pred, y)
            else:
                loss = criterion(y_pred, y)

            loss.backward()
            optimiser.step()

            train_loss += loss.item()
            all_y.append(y.cpu().detach())
            all_y_pred.append(y_pred.cpu().detach())

            if batch_i == 0:
                print("[DEBUG] Batch 0 procesado con exito!")

            del x, y

        train_loss_avg = train_loss / len(train_loader)
        all_y      = torch.cat(all_y)
        all_y_pred = torch.cat(all_y_pred)
        train_metric = balanced_accuracy_score(
            all_y.numpy(), (all_y_pred.numpy() > 0.5).astype(float)
        )

        if x_val is not None:
            val_loss, val_metric = test_model(model, val_loader, clas_weight, criterion, device=device)
            acc_metric      = val_metric
            best_acc_metric = best_val_acc
        else:
            val_loss, val_metric = None, None
            acc_metric      = train_metric
            best_acc_metric = best_train_acc

        if acc_metric > best_acc_metric:
            checkpoint_name = f"model_{model_name}.pth"
            e_saved = e
            torch.save(model.state_dict(), os.path.join(model_dir, checkpoint_name))
            print(f"[DEBUG] Modelo guardado -> {os.path.join(model_dir, checkpoint_name)}")
            best_train_acc = train_metric
            if x_val is not None:
                best_val_acc = val_metric
            overrun_counter = -1

        overrun_counter += 1

        if x_val is not None:
            output_string = (
                "Epoch: %d, Train Loss: %.8f, Train Acc: %.8f, Val Loss: %.8f, "
                "Val Acc: %.8f, overrun_counter %i"
                % (e, train_loss_avg, train_metric, val_loss, val_metric, overrun_counter)
            )
        else:
            output_string = (
                "Epoch: %d, Train Loss: %.8f, Train Acc: %.8f, overrun_counter %i"
                % (e, train_loss_avg, train_metric, overrun_counter)
            )
        print(output_string)
        output_string_to_save += output_string + "\n"
        print(f"[DEBUG] Epoca {e} completada en {round((time.time()-start_time)/60, 4)} min.")

        if overrun_counter > hyperparameters.max_overrun:
            print(f"[DEBUG] Early stopping en epoca {e} (overrun={overrun_counter})")
            break

    if e_saved is not None:
        checkpoint_name = f"model_{model_name}_final.pth"
        torch.save(model.state_dict(), os.path.join(model_dir, checkpoint_name))
        print(f"[DEBUG] Modelo final guardado -> {os.path.join(model_dir, checkpoint_name)}")

    output_string_to_save += f"Best epoch: {e_saved}\n"
    with open(os.path.join(model_dir, f"output_{model_name}.txt"), "w") as f:
        f.write(output_string_to_save)

    return model


def test_model(model, test_loader, clas_weight, criterion, device=None):
    with torch.no_grad():
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        test_loss = 0.0
        model.eval()
        all_y, all_y_pred = [], []

        for inputs in test_loader:
            x = inputs[0].repeat(1, 3, 1, 1).to(device)
            y = torch.argmax(inputs[1], dim=1, keepdim=True).float().to(device)

            y_pred = model(x)

            if clas_weight is not None:
                criterion.weight = (clas_weight[1] - clas_weight[0]) * y + clas_weight[0]
                loss = criterion.forward(y_pred, y)
            else:
                loss = criterion(y_pred, y)

            test_loss += loss.item()
            all_y.append(y.cpu().detach())
            all_y_pred.append(y_pred.cpu().detach())
            del x, y, y_pred

        all_y      = torch.cat(all_y)
        all_y_pred = torch.cat(all_y_pred)
        test_metric = balanced_accuracy_score(
            all_y.numpy(), (all_y_pred.numpy() > 0.5).astype(float)
        )

    return test_loss / len(test_loader), test_metric


def load_model(filepath, model=ResnetDropoutFull()):
    print(f"[DEBUG] Cargando modelo desde: {filepath}")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("[DEBUG] Usando data parallel")
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model = model.to(device)

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    elif torch.backends.mps.is_available():
        map_location = lambda storage, loc: storage.mps()
    else:
        map_location = torch.device("cpu")

    model.load_state_dict(torch.load(filepath, map_location=map_location))
    print("[DEBUG] Modelo cargado exitosamente.")
    return model