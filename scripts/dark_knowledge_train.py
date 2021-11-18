from tqdm import tqdm
import torch
import torch.nn.functional as F
from scripts.metrics import count_FA_FR, get_au_fa_fr


def dkd_loss(alpha, temp, Q_st, Q_te, labels):
    Q_st = F.log_softmax(Q_st / temp, dim=-1)
    Q_te = F.log_softmax(Q_te / temp, dim=-1)
    st_te_cross_entropy = F.cross_entropy(Q_st, Q_te) * alpha * temp ** 2
    st_la_cross_entropy = F.cross_entropy(Q_st, labels) * (1 - alpha)
    return st_te_cross_entropy + st_la_cross_entropy


def train_epoch(
        teacher_model, student_model, opt,
        loader, log_melspec, device,
        config
):
    teacher_model.train()
    student_model.train()
    for i, (batch, labels) in tqdm(enumerate(loader), total=len(loader)):
        batch, labels = batch.to(device), labels.to(device)
        batch = log_melspec(batch)

        opt.zero_grad()

        # run model # with autocast():
        with torch.no_grad():
            Q_te = teacher_model(batch)

        Q_st = student_model(batch)
        # we need probabilities so we use softmax & CE separately
        probs = F.softmax(Q_st, dim=-1)
        # loss = F.cross_entropy(logits, labels)
        loss = dkd_loss(config.dkd_alpha, config.dkd_temperature,
                        Q_st, Q_te, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), 5)

        opt.step()

        # logging
        argmax_probs = torch.argmax(probs, dim=-1)
        FA, FR = count_FA_FR(argmax_probs, labels)
        acc = torch.sum(argmax_probs == labels) / torch.numel(argmax_probs)

    return acc

@torch.no_grad()
def validation(
        teacher_model, student_model,
        loader, log_melspec, device, config
):
    teacher_model.eval()
    student_model.eval()

    val_losses, accs, FAs, FRs = [], [], [], []
    all_probs, all_labels = [], []
    for i, (batch, labels) in tqdm(enumerate(loader)):
        batch, labels = batch.to(device), labels.to(device)
        batch = log_melspec(batch)

        Q_te = teacher_model(batch)

        Q_st = student_model(batch)
        # we need probabilities so we use softmax & CE separately
        probs = F.softmax(Q_st, dim=-1)
        # loss = F.cross_entropy(logits, labels)
        loss = dkd_loss(config.dkd_alpha, config.dkd_temperature,
                        Q_st, Q_te, labels)

        # logging
        argmax_probs = torch.argmax(probs, dim=-1)
        all_probs.append(probs[:, 1].cpu())
        all_labels.append(labels.cpu())
        val_losses.append(loss.item())
        accs.append(
            torch.sum(argmax_probs == labels).item() /  # ???
            torch.numel(argmax_probs)
        )
        FA, FR = count_FA_FR(argmax_probs, labels)
        FAs.append(FA)
        FRs.append(FR)

    # area under FA/FR curve for whole loader
    au_fa_fr = get_au_fa_fr(torch.cat(all_probs, dim=0).cpu(), all_labels)
    return au_fa_fr