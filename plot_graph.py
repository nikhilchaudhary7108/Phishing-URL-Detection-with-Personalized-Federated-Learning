import os
import json
import matplotlib.pyplot as plt
import pickle
import numpy as np
from collections import defaultdict

def merge_layer_history(weight_history, merge_depth=2):
    """
    weight_history:
      layer_name -> {epoch -> np.array}

    Returns:
      merged_layer -> {epoch -> np.array}
    """

    merged = defaultdict(lambda: defaultdict(list))

    for layer_name, epoch_data in weight_history.items():
        merged_name = ".".join(layer_name.split(".")[:merge_depth])

        for epoch, weights in epoch_data.items():
            merged[merged_name][epoch].append(weights)

    # concatenate weights per epoch
    final_merged = {}
    for layer, epoch_dict in merged.items():
        final_merged[layer] = {
            epoch: np.concatenate(w_list)
            for epoch, w_list in epoch_dict.items()
        }

    return final_merged

def plot_run(run_folder):
    """
    Loads logs.json from the run folder
    and plots all types of graphs with try-except protection.
    """
    json_path = os.path.join(run_folder, "logs.json")
    
    if not os.path.exists(json_path):
        print("❌ logs.json not found in:", run_folder)
        return
    
    # Load logs
    with open(json_path, "r") as f:
        logs = json.load(f)

    batch_train_losses = logs.get("batch_train_losses", [])
    batch_train_accs = logs.get("batch_train_accs", [])
    epoch_train_losses = logs.get("epoch_train_losses", [])
    epoch_train_accs = logs.get("epoch_train_accs", [])
    epoch_val_losses = logs.get("epoch_val_losses", [])
    epoch_val_accs = logs.get("epoch_val_accs", [])
    batch_times = logs.get("batch_times", [])
    epoch_times = logs.get("epoch_times", [])

    # Make /graphs folder inside run folder
    graphs_folder = os.path.join(run_folder, "graphs")
    os.makedirs(graphs_folder, exist_ok=True)

    # ---- Safe Save Function ----
    def safe_plot(x, y, title, xlabel, ylabel, filename):
        try:
            if len(x) == 0 or len(y) == 0:
                print(f"⚠ Skipped {filename} — missing data")
                return
            plt.figure()
            plt.plot(x, y)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_folder, filename))
            plt.close()
            print(f"   ✔ Saved {filename}")
        except Exception as e:
            print(f"❌ Error saving {filename}:", e)

    print("\n📊 Generating graphs...\n")

    # ============================
    # 1. BATCH-WISE GRAPHS
    # ============================

    safe_plot(
        list(range(len(batch_train_losses))),
        batch_train_losses,
        "Batch-wise Training Loss",
        "Batch Index",
        "Loss",
        "batch_loss_vs_batch.png"
    )

    safe_plot(
        list(range(len(batch_train_accs))),
        batch_train_accs,
        "Batch-wise Training Accuracy",
        "Batch Index",
        "Accuracy",
        "batch_acc_vs_batch.png"
    )

    # Cumulative time
    batch_cum_time = []
    total = 0
    for t in batch_times:
        total += t
        batch_cum_time.append(total)

    safe_plot(
        batch_cum_time,
        batch_train_losses,
        "Batch-wise Loss vs Time",
        "Time (seconds)",
        "Loss",
        "batch_loss_vs_time.png"
    )

    safe_plot(
        batch_cum_time,
        batch_train_accs,
        "Batch-wise Accuracy vs Time",
        "Time (seconds)",
        "Accuracy",
        "batch_acc_vs_time.png"
    )

    # ============================
    # 2. EPOCH-WISE GRAPHS
    # ============================

    safe_plot(
        list(range(len(epoch_train_losses))),
        epoch_train_losses,
        "Epoch-wise Train Loss",
        "Epoch",
        "Loss",
        "epoch_train_loss_vs_epoch.png"
    )

    safe_plot(
        list(range(len(epoch_train_accs))),
        epoch_train_accs,
        "Epoch-wise Train Accuracy",
        "Epoch",
        "Accuracy",
        "epoch_train_acc_vs_epoch.png"
    )

    safe_plot(
        list(range(len(epoch_val_losses))),
        epoch_val_losses,
        "Epoch-wise Validation Loss",
        "Epoch",
        "Loss",
        "epoch_val_loss_vs_epoch.png"
    )

    safe_plot(
        list(range(len(epoch_val_accs))),
        epoch_val_accs,
        "Epoch-wise Validation Accuracy",
        "Epoch",
        "Accuracy",
        "epoch_val_acc_vs_epoch.png"
    )

    # ---- Epoch time cumulative ----
    epoch_cum_time = []
    total = 0
    for t in epoch_times:
        total += t
        epoch_cum_time.append(total)

    safe_plot(
        epoch_cum_time,
        epoch_train_losses,
        "Train Loss vs Time (Epoch-wise)",
        "Time (seconds)",
        "Loss",
        "epoch_train_loss_vs_time.png"
    )

    safe_plot(
        epoch_cum_time,
        epoch_train_accs,
        "Train Accuracy vs Time (Epoch-wise)",
        "Time (seconds)",
        "Accuracy",
        "epoch_train_acc_vs_time.png"
    )

    safe_plot(
        epoch_cum_time,
        epoch_val_losses,
        "Val Loss vs Time (Epoch-wise)",
        "Time (seconds)",
        "Loss",
        "epoch_val_loss_vs_time.png"
    )

    safe_plot(
        epoch_cum_time,
        epoch_val_accs,
        "Val Accuracy vs Time (Epoch-wise)",
        "Time (seconds)",
        "Accuracy",
        "epoch_val_acc_vs_time.png"
    )

    print("\n✅ All available graphs saved in:", graphs_folder)



        # ============================
    # 3. MERGED LAYER WEIGHT UPDATES
    # ============================

    weight_history_path = os.path.join(run_folder, "weight_history.pkl")
    layer_updates_folder = os.path.join(run_folder, "layer_updates")
    os.makedirs(layer_updates_folder, exist_ok=True)

    MERGE_DEPTH = 2   # 🔥 change this if needed

    if not os.path.exists(weight_history_path):
        print("⚠ weight_history.pkl not found — skipping layer box plots")
        return

    try:
        with open(weight_history_path, "rb") as f:
            weight_history = pickle.load(f)
    except Exception as e:
        print("❌ Failed to load weight_history.pkl:", e)
        return

    # 🔥 MERGE LAYERS HERE
    merged_history = merge_layer_history(
        weight_history,
        merge_depth=MERGE_DEPTH
    )

    print(f"\n📦 Generating merged layer box plots (depth={MERGE_DEPTH})...\n")

    for layer_name, epoch_data in merged_history.items():
        try:
            epochs = sorted(epoch_data.keys())
            if len(epochs) < 2:
                print(f"⚠ Skipped {layer_name} — not enough epochs")
                continue

            data = [epoch_data[e] for e in epochs]

            plt.figure(figsize=(max(6, len(epochs)), 5))
            plt.boxplot(data, labels=epochs, showmeans=True)

            plt.title(
                f"Weight Distribution Over Epochs\n"
                f"{layer_name}  (merged depth={MERGE_DEPTH})"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Weight Value")
            plt.grid(True, axis="y")
            plt.tight_layout()

            safe_layer_name = layer_name.replace(".", "_").replace("/", "_")
            filename = f"{safe_layer_name}_merged_boxplot.png"

            plt.savefig(os.path.join(layer_updates_folder, filename))
            plt.close()

            print(f"   ✔ Saved {filename}")

        except Exception as e:
            print(f"❌ Error plotting {layer_name}:", e)

    print("\n✅ Merged layer weight box plots saved in:", layer_updates_folder)
