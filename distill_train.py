from scripts.base_config import TaskConfig
from scripts.dark_knowledge_train import train_epoch, validation
from IPython.display import clear_output
import torch


def train(
        teacher_model, student_model, opt,
        melspec_train, melspec_val,
        train_loader, val_loader,
        history, config, weight_path,
        device, wandb_session
):
    for n in range(TaskConfig.num_epochs):
        train_epoch(teacher_model, student_model, opt, train_loader, melspec_train, device, config)

        au_fa_fr = validation(teacher_model, student_model, val_loader,
                              melspec_val, device, config)
        history['val_metric'].append(au_fa_fr)

        wandb_session.log({"val_metric": au_fa_fr})

        clear_output()

        print('END OF EPOCH', n)

    torch.save(student_model.state_dict(), weight_path)
