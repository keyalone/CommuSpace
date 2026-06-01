

import numpy as np
import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import non_negative_parafac, non_negative_parafac_hals
from tensorly.cp_tensor import cp_normalize
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def _resolve_device() -> torch.device:
    """固定返回 CPU 设备"""
    return torch.device("cpu")


def _normalize_columns(mat: torch.Tensor) -> torch.Tensor:
    """对每一列进行 L2 归一化"""
    return mat / (mat.norm(dim=0, keepdim=True) + 1e-10)


def _normalize_rows_l1(mat: torch.Tensor) -> torch.Tensor:
    """对每一行进行 L1 归一化"""
    return mat / (mat.sum(dim=1, keepdim=True) + 1e-10)


def unfold_numpy(tensor: np.ndarray, mode: int) -> np.ndarray:
    """展平张量（NumPy）"""
    return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)


def reconstruct_cp(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, lambda_: torch.Tensor) -> torch.Tensor:
    """用 CP 分解参数重建张量（固定使用 CPU）"""
    return torch.einsum('ir,jr,kr,r->ijk', A, B, C, lambda_)




def tensor_decom_ini(tensor_values,
                     decom_type='non_negative_cp',
                     rank=20,
                     tol=1e-4,
                     n_iter_max=200,
                     l1_reg=0.0,
                     return_errors=False,
                     verbose=False):
   
    # 设置为 NumPy 后端
    tl.set_backend('numpy')

    if isinstance(tensor_values, torch.Tensor):
        tensor_values = tensor_values.cpu().numpy()
    elif not isinstance(tensor_values, np.ndarray):
        raise TypeError("Input must be NumPy ndarray or PyTorch tensor.")

    if len(tensor_values.shape) != 3:
        raise ValueError(f"Only 3D tensors supported. Got shape: {tensor_values.shape}")

    common_params = {
        'rank': rank,
        'init': 'random',
        'tol': tol,
        'n_iter_max': n_iter_max,
        'verbose': verbose,
        'random_state': 2025,
        'svd': 'numpy_svd',
        'normalize_factors': True,
        'return_errors': return_errors
    }

    if decom_type == 'non_negative_cp':
        if return_errors:
            factors, errors = non_negative_parafac(tensor_values, **common_params)
        else:
            factors = non_negative_parafac(tensor_values, **common_params)
    elif decom_type == 'non_negative_sparsity':
        common_params['sparsity_coefficients'] = l1_reg
        if return_errors:
            factors, errors = non_negative_parafac_hals(tensor_values, **common_params)
        else:
            factors = non_negative_parafac_hals(tensor_values, **common_params)
    else:
        raise ValueError(f"Unsupported decomposition type: {decom_type}")

    weights = np.array(factors.weights)
    U = np.array(factors.factors[0])
    V = np.array(factors.factors[1])
    W = np.array(factors.factors[2])

    if return_errors:
        errors = np.array([np.array(e) for e in errors])
        return weights, U, V, W, errors
    else:
        return weights, U, V, W

def train_cp_decomposition(X, rank, nmf_trials=1, gamma=None, gamma_scaling=1,
                           lr=1e-3, epochs=10000, tol=1e-6):
    
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    elif isinstance(X, torch.Tensor):
        X = X.float()
    else:
        raise TypeError("Input X must be NumPy ndarray or Torch tensor.")

    print(f"\n[Initialization] try NMF {nmf_trials} times | platform: CPU")

    # 用 tensor_decom_ini 初始化
    lambda_init, A_init, B_init, C_init, _ = tensor_decom_ini(X.numpy(), rank=rank, return_errors=True)
    
    A = nn.Parameter(torch.tensor(A_init, dtype=torch.float32))
    B = nn.Parameter(torch.tensor(B_init, dtype=torch.float32))
    C = nn.Parameter(torch.tensor(C_init, dtype=torch.float32))
    lambda_ = torch.tensor(lambda_init, dtype=torch.float32)  # ← 不再作为 nn.Parameter
    

    # 自动估计 gamma 和 lambda_l1
    if gamma is None:
        with torch.no_grad():
            X_hat_tmp = reconstruct_cp(A, B, C, lambda_)
            
            loss_rec_0 = torch.norm(X - X_hat_tmp) ** 2
            A_T_A_0 = A.T @ A
            loss_orth_0 = torch.norm(A_T_A_0 - torch.diag(torch.diagonal(A_T_A_0))) ** 2


            if gamma is None:
                gamma = (loss_rec_0 / (loss_orth_0 + 1e-8)).item() * gamma_scaling
                print(f"[Auto γ] gamma = {gamma:.4f}")
            

    gamma = float(gamma)

    optimizer = torch.optim.Adam([A, B, C], lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=100, threshold=0.01, verbose=True
    # )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=100, threshold=0.01
    )

    loss_history, loss_rec_all, fit_history = [], [], []
    prev_loss, fit_old = float('inf'), float('inf')

    for epoch in range(epochs):
        optimizer.zero_grad()
        A_nonneg = torch.clamp(A, min=1e-8)
        X_hat = reconstruct_cp(A_nonneg, B, C, lambda_)
        # pdb.set_trace()
        loss_rec = torch.norm(X - X_hat) ** 2
        A_T_A = A_nonneg.T @ A_nonneg
        loss_orth = torch.norm(A_T_A - torch.diag(torch.diagonal(A_T_A))) ** 2
        
        loss = loss_rec + gamma * loss_orth 
        loss.backward()
        optimizer.step()

        with torch.no_grad():
           
            A.data = torch.clamp(A.data, min=1e-8)
            B.data = torch.clamp(B.data, min=1e-8)
            C.data = torch.clamp(C.data, min=1e-8)
            
            # lambda_.data = torch.clamp(lambda_.data, min=1e-8)
            weights_np, factors_np = cp_normalize((
                lambda_.cpu().numpy(),
                [A.cpu().numpy(), B.cpu().numpy(), C.cpu().numpy()]
            ))
            
            lambda_ = torch.tensor(weights_np, dtype=torch.float32, device=A.device)
            A.data = torch.tensor(factors_np[0], dtype=torch.float32, device=A.device)
            B.data = torch.tensor(factors_np[1], dtype=torch.float32, device=A.device)
            C.data = torch.tensor(factors_np[2], dtype=torch.float32, device=A.device)
            
            

        scheduler.step(loss.item())

        fit = 1 - torch.sqrt(loss_rec) / torch.norm(X)
        loss_rec_all.append(loss_rec.item())
        loss_history.append(loss.item())
        fit_history.append(fit.item())

        if abs(prev_loss - loss.item()) < tol:
            print(f"[Early Stop] epoch={epoch+1} | Δloss={abs(prev_loss - loss.item()):.2e} | Δfit={abs(fit - fit_old):.2e}")
            break

        prev_loss, fit_old = loss.item(), fit.item()

        if epoch % 10 == 0 or epoch == epochs - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | "
                  f"Fit: {fit.item():.4f} | LR: {current_lr:.2e} "
                  f"(Rec: {loss_rec.item():.4f}, Orth: {loss_orth.item():.4f})")

    A_final = torch.clamp(A, min=0)
    return A_final, B.detach(), C.detach(), lambda_.detach(), loss_history, loss_rec_all, fit_history

# --------------------------
# Torch utils
# ---------------------------
def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")




@torch.no_grad()
def cp_normalize_torch(w: torch.Tensor, factors: list[torch.Tensor], eps: float = 1e-12):
    """
    TensorLy cp_normalize 的 Torch 版本（近似同等行为）：
    - 对每个 factor 的每一列做 L2 归一化
    - 把各 factor 列范数乘到权重 w 上
    """
    w = w.clone()
    new_factors = []
    for F in factors:
        # norms: (R,)
        norms = torch.linalg.norm(F, ord=2, dim=0).clamp_min(eps)
        w = w * norms
        new_factors.append(F / norms)
    return w, new_factors


# ---------------------------
# 你已有的初始化函数（保持不变）
# - tensor_decom_ini(...) 返回 numpy: lambda_init, A_init, B_init, C_init, (errors)
# ---------------------------
# 请确保你已经定义了 tensor_decom_ini


def train_cp_decomposition_gpu(
    X,
    rank: int,
    nmf_trials: int = 1,
    gamma: float | None = None,
    gamma_scaling: float = 1.0,
    lr: float = 1e-3,
    epochs: int = 5000,
    tol: float = 1e-6,
    prefer_cuda: bool = True,
    use_amp: bool = True,
    clamp_eps: float = 1e-8,
):
    """
    GPU 版本：
    - 初始化仍用 tensor_decom_ini (CPU/Numpy)
    - 训练全程 torch on GPU
    - 用 cp_normalize_torch 替代 cp_normalize(numpy)
    """

    device = get_device(prefer_cuda=prefer_cuda)
    print(f"[Device] {device}")

    # ---- X -> torch ----
    if isinstance(X, np.ndarray):
        X_t = torch.from_numpy(X).float()
    elif isinstance(X, torch.Tensor):
        X_t = X.float()
    else:
        raise TypeError("Input X must be NumPy ndarray or Torch tensor.")
    X_t = X_t.to(device)

    # ---- Init (CPU/Numpy) ----
    print(f"\n[Initialization] tensorly NMF init on CPU | rank={rank}")
    lam_init, A_init, B_init, C_init, _ = tensor_decom_ini(
        X_t.detach().cpu().numpy(), rank=rank, return_errors=True
    )

    # ---- Params on device ----
    A = nn.Parameter(torch.tensor(A_init, dtype=torch.float32, device=device))
    B = nn.Parameter(torch.tensor(B_init, dtype=torch.float32, device=device))
    C = nn.Parameter(torch.tensor(C_init, dtype=torch.float32, device=device))

    # 权重不作为 Parameter（和你原逻辑一致）
    w = torch.tensor(lam_init, dtype=torch.float32, device=device)

    # ---- Auto gamma ----
    if gamma is None:
        with torch.no_grad():
            A0 = A.clamp_min(clamp_eps)
            B0 = B.clamp_min(clamp_eps)
            C0 = C.clamp_min(clamp_eps)
            X_hat0 = reconstruct_cp(A0, B0, C0, w)
            loss_rec0 = torch.norm(X_t - X_hat0) ** 2
            ATA0 = A0.T @ A0
            loss_orth0 = torch.norm(ATA0 - torch.diag(torch.diagonal(ATA0))) ** 2
            gamma = (loss_rec0 / (loss_orth0 + 1e-8)).item() * gamma_scaling
        print(f"[Auto γ] gamma = {gamma:.4f}")
    gamma = float(gamma)

    # ---- Optimizer / Scheduler ----
    optimizer = torch.optim.Adam([A, B, C], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=100, threshold=0.01
    )

    # ---- AMP ----
    use_amp = bool(use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    loss_history, loss_rec_all, fit_history = [], [], []
    prev_loss, fit_old = float("inf"), float("inf")

    X_norm = torch.norm(X_t)

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)

        # ---- Forward ----
        with torch.cuda.amp.autocast(enabled=use_amp):
            A_nonneg = A.clamp_min(clamp_eps)
            B_nonneg = B.clamp_min(clamp_eps)
            C_nonneg = C.clamp_min(clamp_eps)

            X_hat = reconstruct_cp(A_nonneg, B_nonneg, C_nonneg, w)

            loss_rec = torch.norm(X_t - X_hat) ** 2
            ATA = A_nonneg.T @ A_nonneg
            loss_orth = torch.norm(ATA - torch.diag(torch.diagonal(ATA))) ** 2

            loss = loss_rec + gamma * loss_orth

        # ---- Backward/Step ----
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # ---- Projection + Normalize (torch only) ----
        with torch.no_grad():
            A.clamp_(min=clamp_eps)
            B.clamp_(min=clamp_eps)
            C.clamp_(min=clamp_eps)

            # Torch CP normalize: update w and normalize factors' columns
            w_new, (A_new, B_new, C_new) = cp_normalize_torch(w, [A, B, C])
            w = w_new
            A.copy_(A_new)
            B.copy_(B_new)
            C.copy_(C_new)

        scheduler.step(loss.item())

        # ---- Metrics ----
        fit = 1.0 - torch.sqrt(loss_rec) / (X_norm + 1e-12)
        loss_rec_all.append(loss_rec.item())
        loss_history.append(loss.item())
        fit_history.append(fit.item())

        # ---- Early stop ----
        if abs(prev_loss - loss.item()) < tol:
            print(
                f"[Early Stop] epoch={epoch+1} | Δloss={abs(prev_loss - loss.item()):.2e} "
                f"| Δfit={abs(fit.item() - fit_old):.2e}"
            )
            break
        prev_loss, fit_old = loss.item(), fit.item()

        if epoch % 50 == 0 or epoch == epochs - 1:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Fit: {fit.item():.4f} "
                f"| LR: {current_lr:.2e} (Rec: {loss_rec.item():.4f}, Orth: {loss_orth.item():.4f})"
            )

    A_final = A.clamp_min(0.0)
    return (
        A_final.detach(),
        B.detach(),
        C.detach(),
        w.detach(),
        loss_history,
        loss_rec_all,
        fit_history,
        device,
    )


# -------------------------
# Helpers (standalone)
# -------------------------



class DenseCellTensor:
    """Store X as dense numpy (N,K,T) and provide batch fetch + per-cell energy."""
    def __init__(self, X_dense: np.ndarray):
        X = np.asarray(X_dense)
        if X.ndim != 3:
            raise ValueError(f"X must be (N,K,T), got {X.shape}")
        self.X = X.astype(np.float32, copy=False)
        self.N, self.K, self.T = self.X.shape

    def cell_energy(self) -> np.ndarray:
        # e_i = ||X_i||_F^2
        X2 = self.X.astype(np.float64) ** 2
        return X2.sum(axis=(1, 2))

    def get_batch_dense(self, idx_np: np.ndarray) -> np.ndarray:
        idx_np = np.asarray(idx_np, dtype=np.int64)
        return self.X[idx_np]


def build_importance_p(energy: np.ndarray, alpha: float = 0.05, p_min: float = 1e-12) -> torch.Tensor:
    """
    p(i) ∝ (1-alpha) * energy_i/sum(energy) + alpha * 1/N
    returns torch tensor on CPU
    """
    e = np.asarray(energy, dtype=np.float64)
    e = np.maximum(e, 0.0)
    s = e.sum()
    N = e.shape[0]
    if s <= 0:
        p = np.full(N, 1.0 / N, dtype=np.float64)
    else:
        p_e = e / s
        p = (1.0 - alpha) * p_e + alpha * (1.0 / N)
    p = np.maximum(p, p_min)
    p = p / p.sum()
    return torch.tensor(p, dtype=torch.float32, device="cpu")


def offdiag_fro_norm_sq(G: torch.Tensor) -> torch.Tensor:
    """||G - diag(diag(G))||_F^2 = sum_{r!=s} G_rs^2"""
    diag = torch.diagonal(G)
    G_off = G - torch.diag(diag)
    return (G_off * G_off).sum()


@torch.no_grad()
def estimate_X_norm_from_samples(
    X_store: DenseCellTensor,
    p_cpu: torch.Tensor,
    S: int = 2000,
    invp_cap: float = 1e6,
) -> float:
    """
    Estimate ||X||_F using importance sampling:
      E[ ||X_i||^2 / p(i) ] ≈ (1/S) sum_s ||X_{i_s}||^2 / p(i_s)
      => ||X||_F ≈ sqrt( above )
    """
    p = p_cpu.clamp_min(1e-12)
    p = p / p.sum()
    idx = torch.multinomial(p, S, replacement=True)  # CPU
    idx_np = idx.numpy()
    Xb = torch.from_numpy(X_store.get_batch_dense(idx_np))  # CPU float32
    inv_p = (1.0 / p[idx]).clamp_max(invp_cap)
    est = ((Xb * Xb).sum(dim=(1, 2)) * inv_p).mean().item()
    return float(np.sqrt(max(est, 0.0)) + 1e-12)


def calibrate_phi_from_samples(
    X_store: DenseCellTensor,
    p_cpu: torch.Tensor,               # (N,) CPU
    A0: np.ndarray, B0: np.ndarray, C0: np.ndarray, w0: np.ndarray,
    S: int,
    gamma_scaling: float,
    epsilon_fixed: float,
    batch_for_calib: int = 256,
    prefer_cuda: bool = True,
    use_amp: bool = True,
    clamp_eps: float = 1e-8,
    invp_cap: float = 1e6,
):
    """
    Calibrate phi (zeta) using samples:
      Rec0_hat  = (1/S) Σ_s ||X_i - Xhat_i||^2 / p(i)
      G0_hat    = (1/S) Σ_s (a_i a_i^T) / p(i)   (via Aw^T Aw)
      Orth0_hat = ||offdiag(G0_hat)||_F^2
      phi       = gamma_scaling * Rec0_hat / (Orth0_hat + epsilon_fixed)
    """
    device = get_device(prefer_cuda)
    use_amp = bool(use_amp and device.type == "cuda")

    # sanitize p
    if not torch.isfinite(p_cpu).all():
        raise ValueError("p_cpu has NaN/Inf")
    p = p_cpu.clamp_min(1e-12)
    p = p / p.sum()

    # check init arrays finite
    for name, arr in [("A0", A0), ("B0", B0), ("C0", C0), ("w0", w0)]:
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN/Inf")

    # move B,C,w to device
    B = torch.as_tensor(B0, dtype=torch.float32, device=device).clamp_min(clamp_eps)
    C = torch.as_tensor(C0, dtype=torch.float32, device=device).clamp_min(clamp_eps)
    w = torch.as_tensor(w0, dtype=torch.float32, device=device).clamp_min(clamp_eps)

    R = A0.shape[1]
    rec_sum = 0.0
    G_sum = torch.zeros((R, R), device=device, dtype=torch.float32)

    remaining = int(S)
    while remaining > 0:
        bsz = min(batch_for_calib, remaining)
        remaining -= bsz

        idx_cpu = torch.multinomial(p, bsz, replacement=True)
        idx_np = idx_cpu.numpy()

        Xb_np = X_store.get_batch_dense(idx_np)  # float32
        Xb = torch.from_numpy(Xb_np).to(device, non_blocking=True)

        Ab = torch.as_tensor(A0[idx_np], dtype=torch.float32, device=device).clamp_min(clamp_eps)

        p_i = p[idx_cpu].to(device, non_blocking=True)
        inv_p = (1.0 / p_i).clamp_max(invp_cap)

        # Rec term can use AMP
        with torch.cuda.amp.autocast(enabled=use_amp):
            Xhat = torch.einsum("br,kr,tr,r->bkt", Ab, B, C, w)
            err = ((Xb - Xhat) ** 2).sum(dim=(1, 2))
            rec_sum += float((err * inv_p).sum().item())

        # G term MUST be outside autocast, float32 matmul (prevents Inf)
        sqrt_inv_p = torch.sqrt(inv_p).to(dtype=torch.float32)
        Aw = (Ab.to(dtype=torch.float32) * sqrt_inv_p[:, None])
        G_sum += (Aw.T @ Aw)

    Rec0_hat = rec_sum / float(S)
    G0_hat = G_sum / float(S)
    Orth0_hat = float(offdiag_fro_norm_sq(G0_hat).item())

    phi = float(gamma_scaling * Rec0_hat / (Orth0_hat + float(epsilon_fixed)))
    return phi, Rec0_hat, Orth0_hat, G0_hat.detach().cpu()


# -------------------------
# Main training function (fixed: differentiable orth loss)
# -------------------------
def train_cp_decomposition_gpu_batch(
    X_numpy_dense: np.ndarray,
    rank: int,
    nmf_trials: int = 1,
    gamma_scaling: float = 1.0,
    epsilon_fixed: float = 1e-12,
    alpha: float = 0.05,
    beta: float = 0.99,
    steps: int = 20000,
    batch_cells: int = 256,
    lr: float = 1e-3,
    tol: float = 1e-6,                 # early stop threshold on EMA loss change (checked every log_every)
    prefer_cuda: bool = True,
    use_amp: bool = True,
    clamp_eps: float = 1e-8,
    orth_every: int = 1,               # update G_ema every k steps (monitoring)
    normalize_every: int = 200,
    calib_S: int = 2000,
    calib_batch: int = 256,
    log_every: int = 100,
    invp_cap: float = 1e6,
    orth_mode: str = "batch_I",        # "batch_I" or "batch_offdiag"
):
    """
    Batch CP training with importance sampling.
    Key fix vs previous version:
      - Orth loss is differentiable w.r.t. Ab (batch A), not using G_ema directly.

    Orth modes:
      - "batch_I":      loss_orth = || (Ãᵀ Ã) - I ||_F^2   (columns normalized)
      - "batch_offdiag":loss_orth = || offdiag(Ãᵀ Ã) ||_F^2
    """
    device = get_device(prefer_cuda)
    use_amp = bool(use_amp and device.type == "cuda")
    print(f"[Device] {device} | AMP={use_amp}")

    X_store = DenseCellTensor(X_numpy_dense)
    N, K, T = X_store.N, X_store.K, X_store.T

    # ---- 1) Init via tensor_decom_ini using a subset of cells ----
    init_cells = N
    init_idx = np.random.choice(N, size=init_cells, replace=False)
    X_init_np = X_store.get_batch_dense(init_idx)
    print(f"[Init] small dense for init: {X_init_np.shape}")

    lam_init, A_init_small, B_init, C_init, _ = tensor_decom_ini(
        X_init_np, rank=rank, return_errors=True
    )

    rng = np.random.default_rng(2025)
    A_init = rng.random((N, rank), dtype=np.float32) * 0.1
    A_init[init_idx] = A_init_small.astype(np.float32)
    w_init = lam_init.astype(np.float32)

    # ---- 2) Importance sampling distribution p(i) on CPU ----
    energy = X_store.cell_energy()
    p_cpu = build_importance_p(energy, alpha=alpha, p_min=1e-12)

    # ---- 3) Calibrate phi BEFORE training (epsilon_fixed fixed) ----
    phi, Rec0_hat, Orth0_hat, G0_hat_cpu = calibrate_phi_from_samples(
        X_store=X_store,
        p_cpu=p_cpu,
        A0=A_init, B0=B_init, C0=C_init, w0=w_init,
        S=calib_S,
        gamma_scaling=gamma_scaling,
        epsilon_fixed=epsilon_fixed,
        batch_for_calib=calib_batch,
        prefer_cuda=prefer_cuda,
        use_amp=use_amp,
        clamp_eps=clamp_eps,
        invp_cap=invp_cap,
    )
    print(f"[Calibrate] Rec0_hat={Rec0_hat:.4e} | Orth0_hat={Orth0_hat:.4e} | eps={epsilon_fixed:.1e}")
    print(f"[Calibrate] phi(zeta)={phi:.6g}")

    # ---- 4) Estimate ||X||_F (for fit reporting) ----
    X_norm_est = estimate_X_norm_from_samples(X_store, p_cpu, S=min(2000, N), invp_cap=invp_cap)
    print(f"[Init] estimated ||X||_F ≈ {X_norm_est:.4e}")

    # ---- 5) Parameters ----
    A_emb = nn.Embedding(N, rank, sparse=True).to(device)
    with torch.no_grad():
        A_emb.weight.copy_(torch.as_tensor(A_init, dtype=torch.float32, device=device))
        A_emb.weight.clamp_(min=clamp_eps)

    B = nn.Parameter(torch.as_tensor(B_init, dtype=torch.float32, device=device).clamp_min(clamp_eps))
    C = nn.Parameter(torch.as_tensor(C_init, dtype=torch.float32, device=device).clamp_min(clamp_eps))
    w = nn.Parameter(torch.as_tensor(w_init, dtype=torch.float32, device=device).clamp_min(clamp_eps))

    opt_A = torch.optim.SparseAdam([A_emb.weight], lr=lr)
    opt_BCw = torch.optim.Adam([B, C, w], lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ---- 6) Monitoring state: EMA Gram (NOT used for gradient) ----
    G_ema = torch.as_tensor(G0_hat_cpu.numpy(), dtype=torch.float32, device=device)

    # ---- 7) Training loop ----
    loss_history, fit_history = [], []

    ema_loss = None
    ema_beta = 0.98
    prev_ema_at_check = None
    prev_fit_at_check = None

    p_for_sampling = p_cpu.clamp_min(1e-12)
    p_for_sampling = p_for_sampling / p_for_sampling.sum()

    eye = torch.eye(rank, device=device, dtype=torch.float32)

    for step in range(1, steps + 1):
        idx_cpu = torch.multinomial(p_for_sampling, batch_cells, replacement=True)
        idx_np = idx_cpu.numpy()

        Xb_np = X_store.get_batch_dense(idx_np)
        Xb = torch.from_numpy(Xb_np).to(device, non_blocking=True)

        idx = idx_cpu.to(device, non_blocking=True)
        p_i = p_for_sampling[idx_cpu].to(device, non_blocking=True)
        inv_p = (1.0 / p_i).clamp_max(invp_cap)

        opt_A.zero_grad(set_to_none=True)
        opt_BCw.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            Ab = A_emb(idx).clamp_min(clamp_eps)     # (B,R)
            Bp = B.clamp_min(clamp_eps)
            Cp = C.clamp_min(clamp_eps)
            wp = w.clamp_min(clamp_eps)

            Xhat = torch.einsum("br,kr,tr,r->bkt", Ab, Bp, Cp, wp)

            err = ((Xb - Xhat) ** 2).sum(dim=(1, 2))       # (B,)
            loss_rec = (err * inv_p).mean()

            # ✅ Differentiable orth loss from current batch A
            # column-normalize Ab (per-rank component)
            Abn = Ab / (Ab.norm(dim=0, keepdim=True) + 1e-12)  # (B,R)
            Gb = Abn.T @ Abn  # (R,R)

            if orth_mode == "batch_offdiag":
                loss_orth = offdiag_fro_norm_sq(Gb)
            else:
                loss_orth = ((Gb - eye.to(Gb.dtype)) ** 2).sum()

            loss = loss_rec + float(phi) * loss_orth

        scaler.scale(loss).backward()
        scaler.step(opt_A)
        scaler.step(opt_BCw)
        scaler.update()

        with torch.no_grad():
            A_emb.weight.clamp_(min=clamp_eps)
            B.clamp_(min=clamp_eps)
            C.clamp_(min=clamp_eps)
            w.clamp_(min=clamp_eps)

            # EMA Gram monitoring (outside autocast, float32 matmul; prevents Inf)
            if orth_every > 0 and (step % orth_every) == 0:
                Ab2 = A_emb(idx).clamp_min(clamp_eps).float()
                sqrt_inv_p = torch.sqrt(inv_p).float()
                Aw = Ab2 * sqrt_inv_p[:, None]
                G_batch = (Aw.T @ Aw) / float(batch_cells)
                G_ema.mul_(beta).add_((1.0 - beta) * G_batch)

            # Optional CP normalize (torch version)
            if normalize_every > 0 and (step % normalize_every) == 0:
                w_new, (A_w, B_w, C_w) = cp_normalize_torch(w, [A_emb.weight, B, C])
                w.copy_(w_new)
                A_emb.weight.copy_(A_w)
                B.copy_(B_w)
                C.copy_(C_w)
                A_emb.weight.clamp_(min=clamp_eps)
                B.clamp_(min=clamp_eps)
                C.clamp_(min=clamp_eps)
                w.clamp_(min=clamp_eps)

        cur_loss = float(loss.item())
        loss_history.append(cur_loss)

        ema_loss = cur_loss if ema_loss is None else (ema_beta * ema_loss + (1 - ema_beta) * cur_loss)

        fit = 1.0 - float(torch.sqrt(loss_rec.detach().float()).item() / X_norm_est)
        fit_history.append(float(fit))

        if step % log_every == 0 or step == 1:
            # also show batch Gram offdiag as diagnostic
            gb_off = float(offdiag_fro_norm_sq(Gb.detach().float()).item())
            print(
                f"Step {step}/{steps} | loss={cur_loss:.4e} | ema={ema_loss:.4e} "
                f"| rec={loss_rec.item():.4e} | orth={loss_orth.item():.4e} "
                f"| fit≈{fit:.4f} | phi={phi:.3g} | offdiag(Gb)={gb_off:.3g}"
            )

        if tol > 0 and step % log_every == 0:
            if prev_ema_at_check is not None:
                dloss = abs(prev_ema_at_check - ema_loss)
                dfit = abs((prev_fit_at_check - fit) if prev_fit_at_check is not None else 0.0)
                if dloss < tol:
                    print(f"[Early Stop] step={step} | Δema_loss={dloss:.2e} | Δfit={dfit:.2e}")
                    break
            prev_ema_at_check = ema_loss
            prev_fit_at_check = fit

    return {
        "A": A_emb.weight.detach().cpu().numpy(),
        "B": B.detach().cpu().numpy(),
        "C": C.detach().cpu().numpy(),
        "w": w.detach().cpu().numpy(),
        "phi": phi,
        "p": p_cpu.numpy(),
        "G_ema": G_ema.detach().cpu().numpy(),      # monitoring
        "loss_history": loss_history,
        "fit_history": fit_history,
        "Rec0_hat": Rec0_hat,
        "Orth0_hat": Orth0_hat,
        "X_norm_est": X_norm_est,
        "device": str(device),
    }







def elbow_selection(X, rank_min=2, rank_max=20, gamma=None, gamma_scaling = 0.1,
                    lr=1e-3, epochs=10000, tol=1e-6):

    rank_list = np.arange(rank_min, rank_max + 1)
    fit_ranks = []
    loss_ranks = []

    for r in rank_list:
        print(f'Dealing with factor {r}')
        _, _, _, _, _, loss_rec, fit_all = train_cp_decomposition(X, r, gamma=gamma, gamma_scaling=gamma_scaling,
                                       lr=lr, epochs=epochs, tol=tol)
        fit_ranks.append(fit_all[-1])
        loss_ranks.append(loss_rec[-1])

    fit_ranks = np.array(fit_ranks)
    loss_ranks = np.array(loss_ranks)

    return rank_list, fit_ranks, loss_ranks




from kneed import KneeLocator
import numpy as np
import matplotlib.pyplot as plt

def select_optimal_rank_option(x, y, 
                        min_points=3,
                        min_slope_drop_ratio=0.2):
    """
    使用两段式线性拟合寻找“收益开始变慢”的分割点。

    Parameters
    ----------
    x : array-like
        rank_list
    y : array-like
        对应的性能值
    min_points : int
        每一段至少包含的点数
    min_slope_drop_ratio : float
        后段斜率必须比前段下降的最小比例
        例如 0.2 表示至少下降 20%

    Returns
    -------
    result : dict or None
        {
            'rank': 最优rank,
            'cut_index': 分割点index,
            'slope_before': 前段斜率,
            'slope_after': 后段斜率,
            'slope_drop_ratio': 斜率下降比例
        }
        如果没有找到合适分割点，返回 None
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        raise ValueError("x and y must have same length")

    def fit_line(xseg, yseg):
        a, b = np.polyfit(xseg, yseg, 1)
        yhat = a * xseg + b
        sse = np.sum((yseg - yhat) ** 2)
        return sse, a

    best = None

    # 遍历可能的分割点
    for cut in range(min_points, len(x) - min_points):
        sse1, a1 = fit_line(x[:cut+1], y[:cut+1])
        sse2, a2 = fit_line(x[cut:], y[cut:])
        total_sse = sse1 + sse2

        # 必须是后段斜率更小
        if a2 < a1:
            slope_drop_ratio = (a1 - a2) / abs(a1)

            # 斜率下降必须达到一定比例
            if slope_drop_ratio >= min_slope_drop_ratio:
                if best is None or total_sse < best['total_sse']:
                    best = {
                        'rank': x[cut],
                        'cut_index': cut,
                        'slope_before': a1,
                        'slope_after': a2,
                        'slope_drop_ratio': slope_drop_ratio,
                        'total_sse': total_sse
                    }

    if best is not None:
        best.pop('total_sse')  # 不对外暴露内部值
        return best
    else:
        return None


# import pdb
def find_knee_kneed(rank_list, fit_ranks, plot=True, save_fig=None):
    """
    Use KneeLocator to find the optimal knee point (rank), with monotonic adjustment if needed.

    Args:
        rank_list (array-like): List of ranks (x-axis).
        fit_ranks (array-like): Scores corresponding to each rank (y-axis).
        plot (bool): Whether to show plot.
        save_fig (str or None): Path to save figure.

    Returns:
        knee (int or None): Knee point rank, or None if not found.
    """
    rank_list = np.array(rank_list, dtype=int)
    fit_ranks = np.array(fit_ranks, dtype=float)

    # -------- 判断是否递增 --------
    is_monotonic = np.all(np.diff(fit_ranks) >= 0)
    fit_ranks_processed = fit_ranks if is_monotonic else np.maximum.accumulate(fit_ranks)
    method = "Original" if is_monotonic else "Monotonic Adjusted"

    # -------- 使用 KneeLocator --------
    # pdb.set_trace()
    kneedle = KneeLocator(
        x=rank_list,
        y=fit_ranks_processed,
        curve='concave',
        direction='increasing'
    )
    knee = kneedle.knee
    
    if knee is None:
        knee_opt = select_optimal_rank_option(rank_list, fit_ranks_processed)
        knee = knee_opt['rank']
    
    knee_y = kneedle.knee_y if knee is not None else None

    # -------- 绘图 --------
    if plot:
        plt.figure(figsize=(10, 5))
        
        plt.plot(rank_list, fit_ranks, 'o-', label='Original fit_ranks')
        if not is_monotonic:
            plt.plot(rank_list, fit_ranks_processed, 's--', label='Monotonic fit_ranks')
        if knee is not None:
            plt.axvline(knee, color='red', linestyle='--', label=f'Knee: {knee}')
            plt.scatter(knee, knee_y, color='red', zorder=10)
        
        plt.xlabel('Rank')
        plt.ylabel('Fit Score')
        plt.title(f'Knee Detection using KneeLocator ({method})')
        plt.legend()
        plt.grid(True)
        plt.xticks(np.arange(rank_list.min(), rank_list.max() + 1, 5))

        if save_fig is not None:
            plt.savefig(save_fig)
        plt.show()

    return knee


# =========================
# device
# =========================

# =========================
# blocks frob norm
# =========================
def _frob2_blocks_numpy3d(X_np: np.ndarray, block: int = 50_000) -> float:
    I = X_np.shape[0]
    s = 0.0
    for i0 in range(0, I, block):
        i1 = min(i0 + block, I)
        Xi = X_np[i0:i1].astype(np.float32, copy=False)
        s += float(np.sum(Xi * Xi))
    return s


@torch.no_grad()
def _spectral_norm_sym(H: torch.Tensor, n_iter: int = 20) -> float:
    """Largest eigenvalue of symmetric PSD (R small)."""
    R = H.shape[0]
    v = torch.randn(R, device=H.device, dtype=H.dtype)
    v = v / (torch.linalg.norm(v) + 1e-12)
    for _ in range(n_iter):
        v = H @ v
        v = v / (torch.linalg.norm(v) + 1e-12)
    return max(float((v @ (H @ v)).item()), 1e-12)


# =========================
# projections / normalization helpers
# =========================
@torch.no_grad()
def _clamp_nonneg_(M: torch.Tensor, clip01: bool = False):
    if clip01:
        M.clamp_(0.0, 1.0)
    else:
        M.clamp_(min=0.0)


@torch.no_grad()
def _col_l2_norm(M: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.linalg.norm(M, dim=0) + eps


@torch.no_grad()
def _absorb_and_normalize_BC_(B: torch.Tensor, C: torch.Tensor, lam: torch.Tensor,
                             clip01: bool = False, eps: float = 1e-12):
    """
    Rule 2: absorb BEFORE normalize.
      1) clamp B,C
      2) nB,nC
      3) lam <- lam * nB * nC
      4) B <- B/nB, C <- C/nC  (col L2=1)
    """
    _clamp_nonneg_(B, clip01=clip01)
    _clamp_nonneg_(C, clip01=clip01)

    nB = _col_l2_norm(B, eps)
    nC = _col_l2_norm(C, eps)

    lam.mul_(nB * nC).clamp_(min=0.0)
    B.div_(nB)
    C.div_(nC)


@torch.no_grad()
def _final_absorb_and_normalize_A_(A: torch.Tensor, lam: torch.Tensor,
                                  clip01: bool = False, eps: float = 1e-12):
    """
    Final step (your request):
      normalize A cols to L2=1 and absorb into lam
    """
    _clamp_nonneg_(A, clip01=clip01)
    nA = _col_l2_norm(A, eps)
    lam.mul_(nA).clamp_(min=0.0)
    A.div_(nA)


# =========================
# subset init via tensorly NNCP-HALS
# =========================
def _estimate_BC_on_subset(
    X_np: np.ndarray,
    rank: int,
    sample_ratio: float = 0.10,
    random_state: int = 2025,
    n_iter_max: int = 200,
    tol: float = 1e-4,
    clip01: bool = False,
):
    I, J, K = X_np.shape
    rng = np.random.default_rng(random_state)
    s = max(1, int(I * sample_ratio))
    idx = np.sort(rng.choice(I, size=s, replace=False))
    Xs_np = X_np[idx]  # (s,J,K)

    tl.set_backend("numpy")
    fac = non_negative_parafac_hals(
        Xs_np,
        rank=rank,
        init="random",
        n_iter_max=n_iter_max,
        tol=tol,
        normalize_factors=True,
        random_state=random_state,
        verbose=True,
    )

    lam = torch.tensor(np.array(fac.weights), dtype=torch.float32).clamp(min=0.0)
    B = torch.tensor(np.array(fac.factors[1]), dtype=torch.float32)
    C = torch.tensor(np.array(fac.factors[2]), dtype=torch.float32)

    # enforce our convention: B,C nonneg (optionally box) then absorb+L2
    _absorb_and_normalize_BC_(B, C, lam, clip01=clip01)

    return lam, B, C


# =========================
# A update: blocked PGD for rec = (1/2)||X - [lam;A,B,C]||^2
# =========================
@torch.no_grad()
def _update_A_blocks_pgd_half(
    X_np: np.ndarray,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    lam: torch.Tensor,
    block_rows: int = 50_000,
    pgd_iters: int = 8,
    eta: float = 1e-8,
    clip01: bool = False,
):
    """
    rec = 1/2||X - sum_r lam[r] * a_r∘b_r∘c_r||^2

    ∇A rec = A H - M
      H = (B^T B) * (C^T C) * (lam lam^T)
      M[i,r] = sum_{j,k} X[i,j,k] * B[j,r] * C[k,r] * lam[r]

    step = 1 / ||H||2
    """
    device = A.device
    I, J, K = X_np.shape
    R = A.shape[1]

    BtB = B.T @ B
    CtC = C.T @ C
    LamMat = lam[:, None] * lam[None, :]
    H = BtB * CtC * LamMat + eta * torch.eye(R, device=device, dtype=torch.float32)

    L = _spectral_norm_sym(H)
    step = 1.0 / L

    for i0 in range(0, I, block_rows):
        i1 = min(i0 + block_rows, I)
        Xblk = torch.from_numpy(
            np.ascontiguousarray(X_np[i0:i1]).astype(np.float32, copy=False)
        ).to(device)

        # Mblk[i,r] = sum_{j,k} X[i,j,k] B[j,r] C[k,r] lam[r]
        Mblk = torch.einsum("ijk,jr,kr,r->ir", Xblk, B, C, lam)  # (rows,R)

        Ablk = A[i0:i1].clone()
        for _ in range(pgd_iters):
            G = Ablk @ H - Mblk
            Ablk = Ablk - step * G
            # Rule 1 & 3: only clamp nonneg (no L2, no absorb)
            _clamp_nonneg_(Ablk, clip01=clip01)

        A[i0:i1] = Ablk

    return A


# =========================
# MTTKRP for B,C (blocked over i)
# =========================
@torch.no_grad()
def _mttkrp_BC_blocks(
    X_np: np.ndarray,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    lam: torch.Tensor,
    block_rows: int = 50_000,
):
    """
    MB[j,r] = sum_{i,k} X[i,j,k] * A[i,r] * C[k,r] * lam[r]
    MC[k,r] = sum_{i,j} X[i,j,k] * A[i,r] * B[j,r] * lam[r]
    """
    device = A.device
    I, J, K = X_np.shape
    R = A.shape[1]

    MB = torch.zeros(J, R, device=device, dtype=torch.float32)
    MC = torch.zeros(K, R, device=device, dtype=torch.float32)

    for i0 in range(0, I, block_rows):
        i1 = min(i0 + block_rows, I)
        Xblk = torch.from_numpy(
            np.ascontiguousarray(X_np[i0:i1]).astype(np.float32, copy=False)
        ).to(device)
        Ablk = A[i0:i1]

        # MB
        T = torch.einsum("ijk,kr->ijr", Xblk, C)            # (rows,J,R)
        MB += torch.einsum("ijr,ir,r->jr", T, Ablk, lam)    # sum_i

        # MC
        U = torch.einsum("ijk,jr->ikr", Xblk, B)            # (rows,K,R)
        MC += torch.einsum("ikr,ir,r->kr", U, Ablk, lam)

    return MB, MC


# =========================
# HALS update (for nonneg NNLS)
# =========================
@torch.no_grad()
def _hals_update_factor(U: torch.Tensor, M: torch.Tensor, H: torch.Tensor,
                        n_inner: int = 2, eps: float = 1e-12):
    """
    min_{U>=0} 1/2||M - U H||_F^2
    HALS:
      u_r <- max(0, u_r + (m_r - (U H)_r)/H_rr)
    """
    R = U.shape[1]
    for _ in range(n_inner):
        UH = U @ H
        for r in range(R):
            denom = float(H[r, r].item()) + eps
            U[:, r] = (U[:, r] + (M[:, r] - UH[:, r]) / denom).clamp(min=0.0)
            UH = U @ H
    return U


# =========================
# full rec loss (1/2||X-Xhat||^2) via blocks
# =========================
@torch.no_grad()
def _full_loss_rec_blocks_half(
    X_np: np.ndarray,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    lam: torch.Tensor,
    block_rows: int = 50_000,
):
    device = A.device
    I, J, K = X_np.shape
    normX2 = _frob2_blocks_numpy3d(X_np, block=block_rows)

    inner = 0.0
    for i0 in range(0, I, block_rows):
        i1 = min(i0 + block_rows, I)
        Xblk = torch.from_numpy(
            np.ascontiguousarray(X_np[i0:i1]).astype(np.float32, copy=False)
        ).to(device)
        Mblk = torch.einsum("ijk,jr,kr,r->ir", Xblk, B, C, lam)  # (rows,R)
        inner += float((Mblk * A[i0:i1]).sum().item())

    AtA = (A.T @ A).cpu()
    BtB = (B.T @ B).cpu()
    CtC = (C.T @ C).cpu()
    LamMat = (lam[:, None] * lam[None, :]).cpu()

    quad = float((AtA * BtB * CtC * LamMat).sum().item())
    rec = 0.5 * (normX2 - 2.0 * inner + quad)
    return float(rec)


# =========================
# Main training
# =========================
def train_cp_decomposition_large_cells(
    X,
    rank: int,
    sample_ratio: float = 0.10,
    subset_iters: int = 200,
    subset_tol: float = 1e-4,
    block_rows: int = 50_000,
    epochs: int = 10,
    A_inner: int = 2,
    A_pgd_iters: int = 8,
    hals_inner: int = 2,
    gamma: float = 0.1,           # your fixed gamma multiplier
    warmup_epochs: int = 3,       # warmup only rec, then compute zeta once and fix
    orth_every: int = 2,
    orth_lr: float = 1e-2,
    eta: float = 1e-8,
    tol: float = 1e-6,
    clip01: bool = False,         # keep False for correctness first
    verbose: bool = True,
    random_state: int = 2025,
):
    """
    Objective:
      L = (1/2)||X - [lam;A,B,C]||^2 + (zeta/2)||Off(A^T A)||^2

    Constraints (training):
      - A: >=0 (optional <=1), NO column L2 normalization during training.
      - B,C: >=0 (optional <=1), column L2=1, with scale absorbed into lam each epoch.

    zeta strategy:
      - warmup_epochs: zeta=0
      - at ep==warmup_epochs: zeta_fixed = gamma * (rec_avg / orth_avg)
      - then fixed, apply orth step every orth_every epochs
    """
    device = _resolve_device()
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    if isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().numpy().astype(np.float32, copy=False)
    elif isinstance(X, np.ndarray):
        X_np = X.astype(np.float32, copy=False)
    else:
        raise TypeError("X must be torch.Tensor or np.ndarray")

    I, J, K = X_np.shape
    R = rank

    if verbose:
        print(f"[Stage 0] subset init | sample_ratio={sample_ratio:.2%} | rank={R}")

    lam, B, C = _estimate_BC_on_subset(
        X_np, rank=R, sample_ratio=sample_ratio,
        random_state=random_state, n_iter_max=subset_iters, tol=subset_tol,
        clip01=clip01
    )
    lam = lam.to(device)
    B = B.to(device)
    C = C.to(device)

    # init A (small positive)
    A = torch.rand(I, R, device=device, dtype=torch.float32) * 1e-3
    _clamp_nonneg_(A, clip01=clip01)

    normX = np.sqrt(_frob2_blocks_numpy3d(X_np, block=block_rows) + 1e-12)

    loss_rec_hist = []
    loss_total_hist = []
    fit_hist = []

    prev_total = float("inf")
    zeta_fixed: Optional[float] = None

    for ep in range(1, epochs + 1):

        # =====================
        # 1) Update A (blocked PGD) -- NO L2 normalization
        # =====================
        for _ in range(A_inner):
            A = _update_A_blocks_pgd_half(
                X_np, A, B, C, lam,
                block_rows=block_rows,
                pgd_iters=A_pgd_iters,
                eta=eta,
                clip01=clip01,
            )

        # =====================
        # 2) Update B,C (MTTKRP + HALS)
        # =====================
        AtA = A.T @ A
        LamMat = lam[:, None] * lam[None, :]

        # ---- update B ----
        MB, _ = _mttkrp_BC_blocks(X_np, A, B, C, lam, block_rows=block_rows)
        CtC = C.T @ C
        HB = AtA * CtC * LamMat + eta * torch.eye(R, device=device, dtype=torch.float32)
        _hals_update_factor(B, MB, HB, n_inner=hals_inner)
        # absorb+normalize BC happens after C update (or you can do after each; we do after both)

        # ---- update C using UPDATED B ----
        _, MC = _mttkrp_BC_blocks(X_np, A, B, C, lam, block_rows=block_rows)
        BtB = B.T @ B
        HC = AtA * BtB * LamMat + eta * torch.eye(R, device=device, dtype=torch.float32)
        _hals_update_factor(C, MC, HC, n_inner=hals_inner)

        # ---- Rule 2: absorb BC scale into lam, then normalize BC ----
        _absorb_and_normalize_BC_(B, C, lam, clip01=clip01)

        # =====================
        # 3) compute rec / orth
        # =====================
        rec = _full_loss_rec_blocks_half(X_np, A, B, C, lam, block_rows=block_rows)

        AtA_now = A.T @ A
        off = AtA_now - torch.diag(torch.diagonal(AtA_now))
        orth = float(torch.sum(off * off).item())  # ||Off(A^T A)||^2

        # =====================
        # 4) compute zeta once (after warmup) and fix
        # =====================
        if (zeta_fixed is None) and (ep == warmup_epochs):
            rec_avg = rec / float(I)
            orth_avg = orth / float(R * (R - 1) + 1e-12)
            zeta_fixed = float(gamma) * float(rec_avg) / (float(orth_avg) + 1e-12)
            if verbose:
                print(f"[Zeta fixed] gamma={gamma:.3g} | zeta_fixed={zeta_fixed:.3e} "
                      f"(rec_avg={rec_avg:.3e}, orth_avg={orth_avg:.3e})")

        # =====================
        # 5) orth step (delayed + periodic)
        # =====================
        if (zeta_fixed is not None) and (ep > warmup_epochs) and (ep % orth_every == 0):
            # grad of (zeta/2)||Off(A^T A)||^2 is: 2*zeta * A*Off(A^T A)
            grad_orth = 2.0 * float(zeta_fixed) * (A @ off)
            step = float(orth_lr) / (float(torch.linalg.norm(AtA_now).item()) + 1e-12)
            A = A - step * grad_orth
            _clamp_nonneg_(A, clip01=clip01)  # still no L2

        # =====================
        # 6) total / fit / early stop
        # =====================
        if zeta_fixed is None:
            total = float(rec)
            zeta_show = 0.0
        else:
            total = float(rec) + 0.5 * float(zeta_fixed) * orth
            zeta_show = float(zeta_fixed)

        # rec = 1/2||X-Xhat||^2  => ||X-Xhat|| = sqrt(2*rec)
        fit = 1.0 - np.sqrt(max(2.0 * rec, 0.0)) / (normX + 1e-12)

        loss_rec_hist.append(float(rec))
        loss_total_hist.append(float(total))
        fit_hist.append(float(fit))

        if verbose:
            orth_rel = orth / (float(torch.sum(AtA_now * AtA_now).item()) + 1e-12)
            print(f"[Epoch {ep}/{epochs}] zeta={zeta_show:.2e} | rec={rec:.6e} | "
                  f"orth_rel={orth_rel:.3e} | fit={fit:.6f}")

        if abs(prev_total - total) < tol:
            if verbose:
                print(f"[Early Stop] Δloss={abs(prev_total-total):.2e} < tol={tol}")
            break
        prev_total = total

    # =====================
    # FINAL: normalize A columns and absorb into lambda (your request)
    # =====================
    _final_absorb_and_normalize_A_(A, lam, clip01=clip01)

    return A.detach(), B.detach(), C.detach(), lam.detach(), loss_total_hist, loss_rec_hist, fit_hist, zeta_fixed


# =========================
# elbow selection (kept consistent)
# =========================
def elbow_selection_large(
    X,
    rank_min: int = 2,
    rank_max: int = 20,

    # subset init
    sample_ratio_knee: float = 0.10,
    subset_iters: int = 200,
    subset_tol: float = 1e-4,

    # training
    epochs: int = 10,
    block_rows: int = 50_000,
    A_inner: int = 2,
    A_pgd_iters: int = 8,
    hals_inner: int = 2,

    # orth / zeta
    gamma: float = 0.1,
    warmup_epochs: int = 3,
    orth_every: int = 2,
    orth_lr: float = 1e-2,

    # numerics
    eta: float = 1e-8,
    tol: float = 1e-6,
    clip01: bool = False,

    # restarts
    n_restarts: int = 1,
    base_seed: int = 2025,

    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      rank_list,
      fit_mean_by_rank,
      loss_rec_mean_by_rank,
      zeta_fixed_mean_by_rank  (NaN if zeta not computed in a run)

    Notes:
      - Uses your current train_cp_decomposition_large_cells signature/returns:
        A,B,C,lam, loss_total_hist, loss_rec_hist, fit_hist, zeta_fixed
      - For stability, you can set n_restarts>1.
    """
    rank_list = np.arange(rank_min, rank_max + 1, dtype=int)

    fit_mean = []
    loss_rec_mean = []
    zeta_mean = []

    for r in rank_list:
        if verbose:
            print(f"[Elbow] rank={r} | restarts={n_restarts}")

        fit_runs = []
        loss_runs = []
        zeta_runs = []

        for t in range(n_restarts):
            seed = int(base_seed + 1000 * r + t)  # deterministic but rank-dependent

            _, _, _, _, _, loss_rec_hist, fit_hist, zeta_fixed = train_cp_decomposition_large_cells(
                X=X,
                rank=int(r),
                sample_ratio=float(sample_ratio_knee),
                subset_iters=int(subset_iters),
                subset_tol=float(subset_tol),
                block_rows=int(block_rows),
                epochs=int(epochs),
                A_inner=int(A_inner),
                A_pgd_iters=int(A_pgd_iters),
                hals_inner=int(hals_inner),
                gamma=float(gamma),
                warmup_epochs=int(warmup_epochs),
                orth_every=int(orth_every),
                orth_lr=float(orth_lr),
                eta=float(eta),
                tol=float(tol),
                clip01=bool(clip01),
                verbose=False,
                random_state=seed,
            )

            fit_runs.append(float(fit_hist[-1]))
            loss_runs.append(float(loss_rec_hist[-1]))
            zeta_runs.append(float(zeta_fixed) if zeta_fixed is not None else np.nan)

        fit_mean.append(float(np.nanmean(fit_runs)))
        loss_rec_mean.append(float(np.nanmean(loss_runs)))
        zeta_mean.append(float(np.nanmean(zeta_runs)))

        if verbose:
            print(f"  -> fit={fit_mean[-1]:.6f} | loss_rec={loss_rec_mean[-1]:.6e} | zeta={zeta_mean[-1]:.3e}")

    return (
        rank_list,
        np.asarray(fit_mean, dtype=float),
        np.asarray(loss_rec_mean, dtype=float),
        # np.asarray(zeta_mean, dtype=float),
    )

