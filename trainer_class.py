import torch
from torch import optim
import json 
import os
import torch.fx as fx
from datetime import datetime
import yaml
import time
from safetensors.torch import save_file
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, r2_score
import torch.nn as nn
import numpy as np
import pickle
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
def extract_layer_weights(model, layer_keyword="weight"):
    weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad and layer_keyword in name:
            weights[name] = param.detach().cpu().numpy().ravel()
    return weights
class Trainer:
    def __init__(self, 
                 model, 
                 criterion, 
                 global_optimizer=None,
                 personal_optimizer = None, 
                 scheduler_g=None,
                 scheduler_p=None,
                 train_loader=None, 
                 val_loader=None,   
                 changes=None,
                 dataset_name=None,
                 run_name="my_experiment",
                 save = 1,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        
        """
        optimizer_groups: dict with keys like {"transformer": optimizer1, "cnn": optimizer2}
        schedulers: dict with keys matching optimizer_groups (optional)
        """
        self.model = model
        self.best_val_loss = float('inf')
        self.best_train_loss = float('inf')
        self.criterion = criterion
        
        self.global_params = []
        self.persnalization_params = []
        for name, param in model.shared_layer.named_parameters():
            self.global_params.append(param)
        for name, param in model.personal_layer.named_parameters():
            self.persnalization_params.append(param)



        self.global_optimizer = optim.NAdam(self.global_params, lr=1e-4) if global_optimizer is None  else global_optimizer
        self.personal_optimizer = optim.NAdam(self.persnalization_params, lr=1e-3) if personal_optimizer is None else personal_optimizer
        self.scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(self.global_optimizer, mode='min', factor=0.5, patience=2) if scheduler_g is None  else scheduler_g
        self.scheduler_p = optim.lr_scheduler.ReduceLROnPlateau(self.personal_optimizer, mode='min', factor=0.5, patience=2) if scheduler_p is None  else scheduler_p
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_train_losses = []
        self.batch_train_accs = []
        self.epoch_train_losses = []
        self.epoch_train_accs = []
        self.epoch_val_losses = []
        self.epoch_val_accs = []

        self.batch_time = []
        self.epoch_time = []
        self.save = save
        
        if self.save > 0: 
            self._create_run_folder(folder_name=run_name)
            self.best_model_path = os.path.join(self.run_folder, "best_model.pt")
            self._save_run_yaml(run_name, dataset_name)
            self._save_diff(changes)
            example_x = torch.randint(0, 1000, (1, 128))  # FIXED shape

            gm = self.trace_model_fx(self.model, example_x)

            with open(os.path.join(self.run_folder,"model_fx_graph.txt"), "w") as f:
                f.write(str(gm.graph))

    def trace_model_fx(self, model, example_input):
        """
        Safely trace a PyTorch model using FX
        WITHOUT modifying model code.
        """

        model = model.eval()

        gm = fx.symbolic_trace(
            model,
            concrete_args={"x": example_input}
        )

        return gm

    def _save_diff(self, diff):

        with open(os.path.join(self.run_folder, "diff.json"), "w") as f:
            json.dump(diff, f, indent=4)


    def _save_run_yaml(self, run_name,dataset_name):
        run_config = {
            "run": {
                "name": run_name,
                "timestamp": datetime.now().isoformat(),
                "device": str(self.device),
                
            },
            "training": {
                "batch_size": self.train_loader.batch_size if self.train_loader else None,
                "num_train_batches": len(self.train_loader) if self.train_loader else None,
                "num_val_batches": len(self.val_loader) if self.val_loader else None,
                "dataset_name": dataset_name,

            },
            "optimizers": {
                "global": {
                    "type": self.global_optimizer.__class__.__name__,
                    "lr": self.global_optimizer.param_groups[0]["lr"],
                    "betas": self.global_optimizer.param_groups[0].get("betas"),
                    "weight_decay": self.global_optimizer.param_groups[0].get("weight_decay")
                },
                "personal": {
                    "type": self.personal_optimizer.__class__.__name__,
                    "lr": self.personal_optimizer.param_groups[0]["lr"],
                    "betas": self.personal_optimizer.param_groups[0].get("betas"),
                    "weight_decay": self.personal_optimizer.param_groups[0].get("weight_decay")
                }
            },
            "schedulers": {
                "global": {
                    "type": self.scheduler_g.__class__.__name__,
                    "factor": self.scheduler_g.factor,
                    "patience": self.scheduler_g.patience
                },
                "personal": {
                    "type": self.scheduler_p.__class__.__name__,
                    "factor": self.scheduler_p.factor,
                    "patience": self.scheduler_p.patience
                }
            }
        }

        with open(os.path.join(self.run_folder, "run.yaml"), "w") as f:
            yaml.safe_dump(run_config, f, sort_keys=False)


    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True


    def train(self, epochs_list=[4,3,3], early_stopping=True, frac=1.0, val_frac=1.0, log=0, best_val_acc = -float("inf"),acc_patience = 5, acc_wait = 0, min_delta = 1e-4, skip_phase=True):
        train_length = len(self.train_loader)
        self.weight_history = {}
        for phase, epochs in enumerate(epochs_list):
            initial_time=time.perf_counter()
            for epoch in range(epochs):
                et0 = time.perf_counter()
                if early_stopping==True and et0-initial_time > 60 * epochs:
                    print('early stoping due to excedding time limit')
                    break
                if not skip_phase:
                    if phase == 0:
                        for param in self.model.shared_layer.parameters():
                            param.requires_grad = True
                        for param in self.model.personal_layer.parameters():
                            param.requires_grad = True
                        active_optims = [self.global_optimizer, self.personal_optimizer]
                        active_scheds = [self.scheduler_g, self.scheduler_p]
                        phase_name = "global+personal"
                    elif phase == 1:
                        # Train CNN/LSTM/FC — freeze Transformer
                        for param in self.model.shared_layer.parameters():
                            param.requires_grad = True
                        for param in self.model.personal_layer.parameters():
                            param.requires_grad = False
                        active_optims = [self.global_optimizer]
                        active_scheds = [self.scheduler_g]
                        phase_name = "global"
                    else:
                        for param in self.model.shared_layer.parameters():
                            param.requires_grad = False
                        for param in self.model.personal_layer.parameters():
                            param.requires_grad = True
                        active_optims = [self.personal_optimizer]
                        active_scheds = [self.scheduler_p]
                        phase_name = "personal"
                else:
                    phase=0
                    active_optims = [self.global_optimizer, self.personal_optimizer]
                    active_scheds = [self.scheduler_g, self.scheduler_p]
                    phase_name = "global+personal"
                self.model.train()
                train_loss, correct_train, total_train = 0, 0, 1
                max_batches = int(train_length * (frac)) +1 
                running_train_loss = 0
                for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
                    if batch_idx >= max_batches:
                        break
                    
                    bt0 = time.perf_counter()
                    batch_x, batch_y = batch_x.to(self.device, non_blocking=True), batch_y.to(self.device, non_blocking=True).float().unsqueeze(1)
                    for opt in active_optims:
                        opt.zero_grad()

                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()

                    for opt in active_optims:
                        opt.step()




                    # === Metrics ===
                    if log > 1 and (batch_idx + 1) % (40/log) == 0:
                        batch_loss = loss.item()
                        preds = (outputs >= 0.5).float()
                        batch_acc = (preds == batch_y).float().mean().item()
                        print(f"\rEpoch {epoch+1}/{epochs}: Training {phase_name} | "
                            f"Batch {batch_idx+1}/{max_batches} | "
                            f"Loss: {batch_loss:.4f}, Acc: {batch_acc:.4f}", end='')

                    
                        batch_time = time.perf_counter() - bt0

                        self.batch_time.append(batch_time)


                        self.batch_train_losses.append(batch_loss)
                        self.batch_train_accs.append(batch_acc)
                        running_train_loss += batch_loss
                        total_train += 1
                avg_train_loss = running_train_loss / total_train



                if log > 2:
                    print(f'\r total training batch size {max_batches}'.ljust(100), end='')
                    with torch.no_grad():
                        for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
                            if batch_idx >= max_batches:
                                break
                            batch_x, batch_y = batch_x.to(self.device, non_blocking=True), batch_y.to(self.device, non_blocking=True).float().unsqueeze(1)
                            outputs = self.model(batch_x)
                            loss = self.criterion(outputs, batch_y)

                            batch_loss = loss.item()
                            probs = torch.sigmoid(outputs)
                            preds = (probs >= 0.5).float()
                            batch_acc = (preds == batch_y).float().mean().item()

                            train_loss += batch_loss * batch_x.size(0)
                            correct_train += (preds == batch_y).sum().item()
                            total_train += batch_x.size(0)

                        avg_train_loss = train_loss / total_train
                        train_acc = correct_train / total_train
                        self.epoch_train_accs.append(train_acc)
                self.epoch_train_losses.append(avg_train_loss)

                # === Validation ===
                if self.val_loader is not None:
                    avg_val_loss, val_acc = self.evaluate(val_frac)


                    if log > 0:
                        print(f"\rEpoch {epoch+1}/{epochs} Training {phase_name}| "
                            f"Train Loss: {avg_train_loss:.4f},"+" Train Acc: {train_acc:.4f} | " if log > 2 else " | "
                            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
                    self.epoch_val_losses.append(avg_val_loss)
                    self.epoch_val_accs.append(val_acc)

                else:
                    if log > 0:
                        print(f"\rEpoch {epoch+1}/{epochs} Training {phase_name}| "
                            f"Train Loss: {avg_train_loss:.4f},"+" Train Acc: {train_acc:.4f}"" Train Acc: {train_acc:.4f} | " if log > 2 else "")

                if self.val_loader is not None:
                    for sched in active_scheds:
                        sched.step(round(avg_val_loss, 3))

                epoch_time = time.perf_counter() - et0
                self.epoch_time.append(epoch_time)

                if self.save > 0 and self.val_loader is not None and avg_val_loss is not None:
                    if avg_val_loss < self.best_val_loss:
                        self.best_val_loss = avg_val_loss

                        # 1️⃣ Save best model
                        save_file(
                            self.model.state_dict(),
                            os.path.join(self.run_folder, "best_loss_model.safetensors")
                        )

                        # 2️⃣ Store weight distributions for this epoch
                        layer_weights = extract_layer_weights(self.model)
                        for layer_name, w in layer_weights.items():
                            self.weight_history.setdefault(layer_name, {})
                            self.weight_history[layer_name][epoch] = w

                        if log > 1:
                            print(
                                f" Best Loss Model Saved! "
                                f"Val Loss = {avg_val_loss:.4f} | "
                                f"Weights captured  epoch {epoch+1}"
                            )
                # === Accuracy-based Early Stopping ===
                if early_stopping and self.val_loader is not None:

                    if val_acc > best_val_acc + min_delta:
                        best_val_acc = val_acc
                        acc_wait = 0
                    else:
                        acc_wait += 1
                        if log > 1:
                            print(f"No Val Acc improvement ({acc_wait}/{acc_patience})")
                        if acc_wait >= acc_patience:
                            print(
                                f"Early stopping due to validation accuracy plateau. "
                                f"Best Val Acc: {best_val_acc:.4f}"
                            )
                            break

        if self.save > 0:
            self.save_training_data(frac,val_frac)



    def evaluate(self, frac=1.0):
        self.model.eval()
        val_loss, correct_val, total_val = 0, 0, 0
        max_batches = max(int(len(self.val_loader) * (frac)), 0)+1
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y) in enumerate(self.val_loader):
                if batch_idx >= max_batches:
                    break
                batch_x, batch_y = batch_x.to(self.device, non_blocking=True), batch_y.to(self.device, non_blocking=True).float().unsqueeze(1)
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                avg_batch_loss = loss.item()
                val_loss += avg_batch_loss * batch_x.size(0)
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                correct_val += (preds == batch_y).sum().item()
                total_val += batch_x.size(0)
        avg_val_loss = val_loss / total_val
        val_acc = correct_val / total_val
        return avg_val_loss, val_acc

    def test(self, test_loader, th=0.5):
        self.model.eval()
        all_preds = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device).float().unsqueeze(1)

                outputs = self.model(batch_x)
                probs = outputs.cpu().numpy().flatten()
                preds = (outputs >= th).float().cpu().numpy().flatten()
                targets = batch_y.cpu().numpy().flatten()

                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(targets)

        # Convert to arrays
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Metrics
        try:
            roc = roc_auc_score(all_targets, all_probs)
        except:
            roc = None

        metrics = {
            "threshold":float(th),
            "accuracy": float(accuracy_score(all_targets, all_preds)),
            "precision": float(precision_score(all_targets, all_preds, zero_division=0)),
            "recall": float(recall_score(all_targets, all_preds, zero_division=0)),
            "f1": float(f1_score(all_targets, all_preds, zero_division=0)),
            "roc_auc": None if roc is None else float(roc),
            "r2_score": float(r2_score(all_targets, all_probs)),
            "classification_report": classification_report(all_targets, all_preds, output_dict=True),
            "confusion_matrix": confusion_matrix(all_targets, all_preds).tolist()
        }

        # Save to JSON
        if self.save > 0:
            save_path = os.path.join(self.run_folder, "test_metrics.json")
            with open(save_path, "w") as f:
                json.dump(metrics, f, indent=4)
            print(f"✅ Test metrics saved to: {save_path}")
        return metrics
    
    def tune_threshold(self, to_tune="acc"):
        """
        Tune decision threshold based on selected metric.
        
        to_tune: one of ["acc", "prec", "rec", "f1"]
        """
        self.model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x).squeeze(-1)
                all_logits.append(logits.cpu())
                all_labels.append(y.cpu())

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)
        probs = torch.sigmoid(logits)

        thresholds = np.linspace(0.1, 0.9, 801)
        best_t = 0.5
        best_score = -1.0

        for t in thresholds:
            preds = (probs >= t).int()

            if to_tune == "acc":
                score = accuracy_score(labels, preds)

            elif to_tune == "prec":
                score = precision_score(labels, preds, zero_division=0)

            elif to_tune == "rec":
                score = recall_score(labels, preds)

            elif to_tune == "f1":
                score = f1_score(labels, preds)

            else:
                raise ValueError(
                    "to_tune must be one of ['acc', 'prec', 'rec', 'f1']"
                )

            if score > best_score:
                best_score = score
                best_t = t

        print(f"Best threshold ({to_tune}): {best_t:.3f}")
        print(f"Best {to_tune}: {best_score:.6f}")

        return best_t


    def _create_run_folder(self, folder_name="run"):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.main_dir = "training_runs"
        os.makedirs(self.main_dir, exist_ok=True)

        self.run_folder = os.path.join(self.main_dir, f"{timestamp}_{folder_name}")
        os.makedirs(self.run_folder, exist_ok=True)

        print("Run folder created at:", self.run_folder)

    def save_training_data(self, frac, val_frac):
        data = {
            "training_dataset_size": frac,
            "validation_dataset_size": val_frac,
            "batch_train_losses": self.batch_train_losses,
            "batch_train_accs": self.batch_train_accs,
            "epoch_train_losses": self.epoch_train_losses,
            "epoch_train_accs": self.epoch_train_accs,
            "epoch_val_losses": self.epoch_val_losses,
            "epoch_val_accs": self.epoch_val_accs,
            "batch_times": self.batch_time,
            "epoch_times": self.epoch_time
        }
        with open(os.path.join(self.run_folder, "weight_history.pkl"), "wb") as f:
            pickle.dump(self.weight_history, f)

        # Save as JSON
        with open(os.path.join(self.run_folder, "logs.json"), "w") as f:
            json.dump(data, f, indent=4)

    

