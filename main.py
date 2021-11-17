from scripts.dataset import SpeechCommandDataset
from configs.init import TaskConfig
import torch
from scripts.augmentation import AugsCreation
from scripts.collator import get_sampler, Collator
from torch.utils.data import DataLoader
from scripts.melspec import LogMelspec
from scripts.base_model import CRNN
from collections import defaultdict
from train import train
import sys


def main_worker(weight_path):
    print("initialize dataset")
    dataset = SpeechCommandDataset(
        path2dir='speech_commands', keywords=TaskConfig.keyword
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

    train_loader = DataLoader(train_set, batch_size=TaskConfig.batch_size,
                              shuffle=False, collate_fn=Collator(),
                              sampler=train_sampler,
                              num_workers=2, pin_memory=True)

    val_loader = DataLoader(val_set, batch_size=TaskConfig.batch_size,
                            shuffle=False, collate_fn=Collator(),
                            num_workers=2, pin_memory=True)

    print("initialize melspec")

    melspec_train = LogMelspec(is_train=True, config=TaskConfig)
    melspec_val = LogMelspec(is_train=False, config=TaskConfig)

    print("initialize model")

    history = defaultdict(list)
    config = TaskConfig()
    model = CRNN(config).to(config.device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    print(model)
    print("number of parameters:", sum([p.numel() for p in model.parameters()]))

    print("start train section")

    train(
        model, opt,
        melspec_train, melspec_val,
        train_loader, val_loader,
        history, config, weight_path, device=config.device
    )

    print("save model")
    return 0


if __name__ == "__main__":
    main_worker(sys.argv[1])
