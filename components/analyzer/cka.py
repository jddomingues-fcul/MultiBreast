import torch


@torch.no_grad()
def hsic(
    model_a_sim: torch.Tensor, model_b_sim: torch.Tensor, centering_matrix: torch.Tensor
) -> torch.Tensor:
    assert model_a_sim.shape == model_b_sim.shape, (
        "Kernel matrices must have the same shape"
    )

    n = model_a_sim.size(0)
    if n <= 1:
        print("HSIC is not defined for n <= 1. Returning 0.")
        return torch.tensor(0.0, device=model_a_sim.device)

    denom = (model_a_sim.size(0) - 1) ** 2
    return (
        torch.trace(model_a_sim @ centering_matrix @ model_b_sim @ centering_matrix)
        / denom
    )


@torch.no_grad()
def linear_kernel(model_features: torch.Tensor) -> torch.Tensor:
    return model_features @ model_features.T


@torch.no_grad()
def create_centering_matrix(
    n: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if n <= 1:
        print("Centering matrix is not defined for n <= 1. Returning identity matrix.")
        return torch.ones((n, n), device=device)

    one_n = torch.ones((n, n), device=device) / n
    h = torch.eye(n, device=device) - one_n
    return h


@torch.no_grad()
def cka(
    model_a_features: torch.Tensor, model_b_features: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    n = model_a_features.size(0)
    if n <= 1:
        print("CKA is not defined for n <= 1. Returning 0.")
        return torch.tensor(0.0, device=model_a_features.device)

    model_a_sim, model_b_sim = (
        linear_kernel(model_a_features),
        linear_kernel(model_b_features),
    )
    centering_matrix = create_centering_matrix(
        model_a_sim.size(0), model_a_sim.device, model_a_sim.dtype
    )
    hsic_kl = hsic(model_a_sim, model_b_sim, centering_matrix)
    hsic_kk = hsic(model_a_sim, model_a_sim, centering_matrix)
    hsic_ll = hsic(model_b_sim, model_b_sim, centering_matrix)

    denom = (hsic_kk * hsic_ll) ** 0.5

    if denom.abs() < eps or not torch.isfinite(denom):
        print("Denominator in CKA is too small or not finite. Returning 0.")
        out = torch.tensor(0.0, device=model_a_features.device)
    else:
        out = hsic_kl / (denom + eps)

    return torch.clamp(out, 0.0, 1.0)


@torch.no_grad()
def get_cka_matrix(model_a_features: torch.Tensor, model_b_features: torch.Tensor):
    # model_a_features: (n_layers_a, n_samples, n_features_a)
    # model_b_features: (n_layers_b, n_samples, n_features_b)
    # out: (n_layers_a, n_layers_b) CKA matrix

    n_layers_a = model_a_features.shape[0]
    n_layers_b = model_b_features.shape[0]
    cka_matrix = torch.zeros(
        (n_layers_a, n_layers_b),
        device=model_a_features.device,
        dtype=model_a_features.dtype,
    )

    for i in range(n_layers_a):
        for j in range(n_layers_b):
            X, Y = model_a_features[i], model_b_features[j]
            cka_temp = cka(X, Y)
            cka_matrix[i, j] = cka_temp
    return cka_matrix
