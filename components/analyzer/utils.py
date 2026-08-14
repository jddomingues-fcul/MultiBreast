import torch


def compute_toks_entropy(probs: torch.Tensor, normalize: bool = True) -> float:
    # probs: log softmaxed tensor of shape (vocab_size,)
    try:
        entropy = -torch.sum(probs * torch.exp(probs)).item()

        if normalize:
            vocab_size = probs.shape[0]
            entropy = entropy / torch.log(torch.tensor(vocab_size)).item()
    except Exception as e:
        print(f"Error computing entropy: {e}. Returning 0.0")
        entropy = 0.0

    return max(0.0, entropy)
