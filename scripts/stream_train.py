from tqdm import tqdm
import torch
import torch.nn.functional as F
from scripts.metrics import count_FA_FR, get_au_fa_fr


class StreamingKws:
    def __init__(self, max_window_length=40, streaming_step_size=3):
        self.max_window_length = max_window_length
        self.streaming_step_size = streaming_step_size

    @torch.no_grad()
    def validation(self, model, x):
        model.eval()
        probs = []
        hidden = None

        for ids in range(0, x.shape[1], self.streaming_step_size):
            if ids + self.max_window_length > x.shape[1]:
                break
            output, hidden = model(x[:, ids:ids + self.max_window_length].unsqueeze(0), hidden, True)

            # we need probabilities so we use softmax & CE separately
            probs.append(F.softmax(output, dim=-1)[..., 1].unsqueeze(-1))

        probs = torch.cat(probs, dim=-1)
        return probs
