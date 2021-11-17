from configs.init import TaskConfig
from scripts.base_train import train_epoch, validation
from scripts.melspec import LogMelspec
import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch


def train(
        model, opt,
        melspec_train, melspec_val,
        train_loader, val_loader,
        history, config, weight_path, device
):
    for n in range(TaskConfig.num_epochs):
        print(type(model))
        print(type(train_loader))
        print(type(melspec_train))
        print(type(device))
        train_epoch(model, opt, train_loader, melspec_train, device)

        au_fa_fr = validation(model, val_loader,
                              melspec_val, device)
        history['val_metric'].append(au_fa_fr)

        clear_output()
        plt.plot(history['val_metric'])
        plt.ylabel('Metric')
        plt.xlabel('Epoch')
        plt.grid()
        plt.show()

        print('END OF EPOCH', n)

    torch.save(model.state_dict(), weight_path)
