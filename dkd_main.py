from scripts.dataset import SpeechCommandDataset
from scripts.dkd_config import TaskConfig
from scripts.base_config import TaskConfig as TeacherConfig
import torch
from scripts.augmentation import AugsCreation
from scripts.collator import get_sampler, Collator
from torch.utils.data import DataLoader
from scripts.melspec import LogMelspec
from scripts.base_model import CRNN
from collections import defaultdict
from dkd_train import train
import sys
import wandb
import os
import torch.quantization
torch.manual_seed(0)


def main_worker(weight_path):
    config_class = TaskConfig

    print("initialize dataset")
    dataset = SpeechCommandDataset(
        path2dir='speech_commands', keywords=config_class.keyword
    )

    indexes = torch.randperm(len(dataset))
    train_indexes = indexes[:int(len(dataset) * 0.8)]
    val_indexes = indexes[int(len(dataset) * 0.8):]

    train_df = dataset.csv.iloc[train_indexes].reset_index(drop=True)
    val_df = dataset.csv.iloc[val_indexes].reset_index(drop=True)

    train_set = SpeechCommandDataset(csv=train_df, transform=AugsCreation())
    val_set = SpeechCommandDataset(csv=val_df)

    print("initialize sampler")

    train_sampler = get_sampler(train_set.csv['label'].values)

    print("initialize dataloader")

    train_loader = DataLoader(train_set, batch_size=config_class.batch_size,
                              shuffle=False, collate_fn=Collator(),
                              sampler=train_sampler,
                              num_workers=2, pin_memory=True)

    val_loader = DataLoader(val_set, batch_size=config_class.batch_size,
                            shuffle=False, collate_fn=Collator(),
                            num_workers=2, pin_memory=True)

    print("initialize melspec")

    melspec_train = LogMelspec(is_train=True, config=config_class)
    melspec_val = LogMelspec(is_train=False, config=config_class)

    print("initialize model student")

    history = defaultdict(list)
    config = config_class()
    model_student = CRNN(config).to(config.device)
    if config.quantize_model:
        model_student = torch.quantization.quantize_dynamic(model_student, {torch.nn.Linear}, dtype=torch.qint8)

    print("initialize model teacher")

    config_teacher = TeacherConfig()
    model_teacher = CRNN(config_teacher)
    model_teacher.load_state_dict(torch.load(config.teacher_model_path))
    model_teacher = model_teacher.to(config.device)
    model_teacher.eval()

    opt = torch.optim.Adam(
        model_student.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    print(model_student)
    print("number of parameters for student:", sum([p.numel() for p in model_student.parameters()]))
    print(model_teacher)
    print("number of parameters for teacher:", sum([p.numel() for p in model_teacher.parameters()]))

    print("set wandb")
    os.environ["WANDB_API_KEY"] = config.wandb_api
    wandb_session = wandb.init(project="kws", entity="nd0761")
    wandb.config = config.__dict__

    print("start train section")
    wandb_session.watch(model_student)

    train(
        model_teacher, model_student, opt,
        melspec_train, melspec_val,
        train_loader, val_loader,
        history, config, weight_path,
        device=config.device, wandb_session=wandb_session
    )
    wandb_session.finish()

    print("save model")
    return 0


if __name__ == "__main__":
    main_worker(sys.argv[1])
