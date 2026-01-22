import copy
import torch
import torch.nn as nn
from model import *
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from trainer_class import Trainer
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def make_tensor_dataset(df):
    url_tensor = torch.stack(list(df["encode"]))
    labels_tensor = torch.tensor(df["label"].astype(np.float32).values, dtype=torch.long)
    return TensorDataset(url_tensor, labels_tensor)
def make_clients(global_model, train_dataset,val_dataset, dataset_name, num_clients=5, alpha=0.5, total_data=1, batch_size=256, max_train_samples_per_client=50000, max_val_samples_per_client=8000):
    import numpy as np
    # using alpha to produce unidentical splits high alpha more identical
    client_fractions = np.random.dirichlet([alpha] * num_clients) * total_data
    total_train_len = int(len(train_dataset) * total_data)
    total_val_len   = int(len(val_dataset) * total_data)
    train_start = 0
    val_start = 0
    client_models = []
    for i in range(num_clients):
        train_size = int(client_fractions[i] * total_train_len)
        val_size   = int(client_fractions[i] * total_val_len)

        train_size = min(train_size, max_train_samples_per_client)
        val_size   = min(val_size, max_val_samples_per_client)

        train_end = min(train_start + train_size, len(train_dataset))
        val_end   = min(val_start + val_size, len(val_dataset))

        train_slice = train_dataset.iloc[train_start:train_end]
        val_slice   = val_dataset.iloc[val_start:val_end]

        train_start = train_end
        val_start   = val_end

        client_model = URLBinaryCNN_bestmodel(vocab_size=97, maxlen=128).to(device)                     # fresh instance
        client_model.shared_layer.load_state_dict(global_model.shared_layer.state_dict())
        train_set = make_tensor_dataset(train_slice)
        val_set   = make_tensor_dataset(val_slice)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        client_models.append({'model':client_model, 'train_loader':train_loader,'val_loader':val_loader, 'val_acc':0, 'dataset':dataset_name})
    return client_models, client_fractions


def quick_fine_tune(model, train_loader, personal_optimizer,frac=1, epochs=1):

    criterion = nn.BCEWithLogitsLoss()

    # ▶▶ added save=0
    trainer = Trainer(
        model,
        criterion=criterion,
        train_loader=train_loader,
        personal_optimizer=personal_optimizer,
        save=0
    )

    trainer.train(epochs_list=[0,0,epochs], frac=frac, log=0)


def train_client(client_devices, epoch=1):
    print("training client models: ")
    client_acc = []
    client_losses = []
    for i, client_device in enumerate(client_devices):
        print('\tclient ',i,'dataset', client_device['dataset'][8], end=' ')
        criterion = nn.BCEWithLogitsLoss()
        lr_p = 0.001
        lr_g = 0.0005

        personal_optimizer = torch.optim.NAdam(client_device['model'].personal_layer.parameters(), lr=lr_p, weight_decay=lr_p/10)
        global_optimizer = torch.optim.NAdam(client_device['model'].shared_layer.parameters(), lr=lr_g, weight_decay=lr_g/10)

        personal_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            personal_optimizer, mode='min', factor=0.5, patience=1
        )
        global_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            global_optimizer, mode='min', factor=0.5, patience=1
        )

        # ▶▶ added save=0
        trainer = Trainer(
            client_device['model'],
            criterion=criterion,
            train_loader=client_device['train_loader'],
            val_loader=client_device['val_loader'],
            personal_optimizer=personal_optimizer,
            global_optimizer=global_optimizer,
            scheduler_g=global_scheduler,
            scheduler_p=personal_scheduler,
            save=0
        )

        trainer.train(epochs_list=[epoch,0,0],log=0)
        #print("epoch accuracies:-", trainer.epoch_val_accs)
        client_losses.append(trainer.epoch_val_losses)
        client_acc.append(trainer.epoch_val_accs)
    print()
    return client_losses, client_acc


def fed_meta_learing(global_model, client_devices,train_loader, global_personal_optimizer, meta_lr=0.1, quick_tune_epoch = 2):
    total_samples = sum(len(client_device['train_loader'].dataset) for client_device in client_devices)
    trim_ratio = 0.2  # 10% trimming
    num_clients = len(client_devices)
    k = int(trim_ratio * num_clients)

    weights = [
        len(client_device['train_loader'].dataset) / total_samples
        for client_device in client_devices
    ]
    global_state = {
        **global_model.shared_layer.state_dict()
    }

    avg_state = copy.deepcopy(global_state)
    '''
    for key in avg_state.keys():

        # 1. Collect all client tensors for this parameter
        client_tensors = []
        for i, client_device in enumerate(client_devices):
            client_state = client_device['model'].shared_layer.state_dict()
            client_tensors.append(client_state[key])

        # Shape: [num_clients, *param_shape]
        stacked = torch.stack(client_tensors, dim=0)

        # 2. Sort across client dimension
        sorted_tensor, _ = torch.sort(stacked, dim=0)

        # 3. Trim extremes
        if k > 0:
            trimmed_tensor = sorted_tensor[k:-k]
        else:
            trimmed_tensor = sorted_tensor

        # 4. Mean of remaining values
        avg_state[key] = torch.mean(trimmed_tensor, dim=0)
    '''
    for key in avg_state.keys():
        avg_state[key] = torch.zeros_like(avg_state[key])
        for client_device, w in zip(client_devices, weights):
            client_state = {
                **client_device['model'].shared_layer.state_dict(),
            #**client_model.personal_layer.state_dict()
            }
            avg_state[key] += client_state[key] * w

    new_state = {}
    for key in global_state.keys():
        new_state[key] = global_state[key] + meta_lr * (avg_state[key] - global_state[key])


    global_model.shared_layer.load_state_dict({
        k: v for k, v in new_state.items() if k in global_model.shared_layer.state_dict()
    })
    quick_fine_tune(global_model,train_loader,global_personal_optimizer,  0.1, epochs=quick_tune_epoch)

def soft_update(client, global_model, alpha=0.3):
    for c_p, g_p in zip(client.shared_layer.parameters(), global_model.shared_layer.parameters()):
        c_p.data = c_p.data + alpha * (g_p.data - c_p.data)

def update_clients(client_devices, global_model, alpha=0.3, quick_tune_epoch = 3):
    #client_acc_avg = sum([client_device['val_acc'] for client_device in client_devices])/len(client_devices)
    #client_std = sum([(client_device['val_acc']-client_acc_avg)**2 for client_device in client_devices])/len(client_devices)**(1/2)
    for i, client_device in enumerate(client_devices):
        client_device['model'].shared_layer.load_state_dict(global_model.shared_layer.state_dict())
        soft_update(client=client_device['model'], global_model=global_model, alpha=alpha)

        personal_optimizer = torch.optim.NAdam(client_device['model'].personal_layer.parameters(), lr=0.001, weight_decay=0.0001)
        #client.personal_layer.load_state_dict(global_model.shared_layer.state_dict())
        quick_fine_tune(client_device['model'], client_device['train_loader'], personal_optimizer,epochs=quick_tune_epoch)
        pass
def evaluate_client(client_devices):
    avg_val_losses, val_accs = [], []
    for i, client_device in enumerate(client_devices):
        criterion = nn.BCEWithLogitsLoss()
        trainer = Trainer(client_device['model'], criterion=criterion, train_loader=client_device['train_loader'], val_loader=client_device['val_loader'],save=0)
        avg_val_loss, val_acc = trainer.evaluate()
        avg_val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        client_device['val_acc'] = val_acc
    return avg_val_losses, val_accs