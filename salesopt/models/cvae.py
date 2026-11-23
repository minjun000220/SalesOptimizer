"""조건부 VAE 수요 생성기 (논문의 cDGM, 데이터가 많을 때).

    생성기 G(x, p, η):  η = (z, ε),  z ~ N(0, I),  ε ~ N(0, 1)
        (μ, log σ) = Decoder(x, log p, z)
        y = μ + σ·ε,   D = exp(y) − 1

논문과 다른 점 (안전 완판용으로 바꾼 부분)
  1. 중도절단 우도: 완판 기록은 log P(Y ≥ y) 로 재구성 항을 계산 (완판 = 수요의 하한)
  2. 가격 단조성 벌점: 같은 (x, z)에서 가격을 올렸을 때 μ 가 커지면 벌점
     → 과거 가격이 수요에 맞춰 정해진 데이터(내생성)에서 '비쌀수록 잘 팔린다'를 배우는 것을 줄인다
"""
from __future__ import annotations

import math

import numpy as np

from salesopt.models.base import DemandModel


def _torch():
    try:
        import torch
    except ImportError as e:                       # pragma: no cover
        raise ImportError("CVAEDemand 는 PyTorch 가 필요합니다: pip install 'salesopt[cvae]'") from e
    return torch


class CVAEDemand(DemandModel):
    def __init__(self, latent_dim: int = 2, hidden: int = 64, epochs: int = 400, batch_size: int = 256,
                 lr: float = 2e-3, weight_decay: float = 1e-4, beta_kl: float = 1.0,
                 monotone_weight: float = 10.0, seed: int = 0, device: str = "cpu"):
        self.latent_dim, self.hidden, self.epochs = latent_dim, hidden, epochs
        self.batch_size, self.lr, self.wd = batch_size, lr, weight_decay
        self.beta_kl, self.mono_w, self.seed, self.device = beta_kl, monotone_weight, seed, device

    def _build(self, d_in):
        torch = _torch()
        nn = torch.nn
        h, k = self.hidden, self.latent_dim

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.enc = nn.Sequential(nn.Linear(d_in + 3, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                      nn.Linear(h, 2 * k))
                s.dec = nn.Sequential(nn.Linear(d_in + 1 + k, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                      nn.Linear(h, 2))

            def decode(s, x, lp, z):
                out = s.dec(torch.cat([x, lp[:, None], z], 1))
                return out[:, 0], out[:, 1].clamp(-5, 2)

            def encode(s, x, lp, y, c):
                out = s.enc(torch.cat([x, lp[:, None], y[:, None], c[:, None]], 1))
                return out[:, :k], out[:, k:].clamp(-8, 4)

        return Net()

    def fit(self, X, log_price, sold, censored, groups=None, n_groups=None):
        torch = _torch()
        torch.manual_seed(self.seed)
        X = np.asarray(X, np.float32)
        lp = np.asarray(log_price, np.float32)
        y = np.log1p(np.asarray(sold, np.float32))
        c = np.asarray(censored, np.float32)
        self.lp_m, self.lp_s = float(lp.mean()), float(lp.std() + 1e-6)
        self.y_m, self.y_s = float(y.mean()), float(y.std() + 1e-6)
        T = lambda a: torch.tensor(a, device=self.device)                      # noqa: E731
        Xt, lpt = T(X), T((lp - self.lp_m) / self.lp_s)
        yt, ct = T((y - self.y_m) / self.y_s), T(c)
        self.net = self._build(X.shape[1]).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        n = len(X)
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        delta = 0.1                                       # 표준화된 log 가격에서 단조성 확인 간격
        half_log2pi = 0.5 * math.log(2 * math.pi)
        self.net.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                xb, lb, yb, cb = Xt[idx], lpt[idx], yt[idx], ct[idx]
                mz, lv = self.net.encode(xb, lb, yb, cb)
                z = mz + torch.exp(0.5 * lv) * torch.randn_like(mz)
                mu, ls = self.net.decode(xb, lb, z)
                zz = (yb - mu) / torch.exp(ls)
                ll = torch.where(cb > 0.5, torch.special.log_ndtr(-zz), -0.5 * zz ** 2 - ls - half_log2pi)
                kl = 0.5 * (torch.exp(lv) + mz ** 2 - 1 - lv).sum(1)
                z0 = torch.randn_like(mz)
                mu1, _ = self.net.decode(xb, lb, z0)
                mu2, _ = self.net.decode(xb, lb + delta, z0)
                mono = torch.relu(mu2 - mu1).mean()
                loss = (-ll + self.beta_kl * kl).mean() + self.mono_w * mono
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.net.eval()
        return self

    def sample(self, X, log_price, groups, m, rng):
        torch = _torch()
        X = np.asarray(X, np.float32)
        n = len(X)
        lp = ((np.asarray(log_price, np.float32) - self.lp_m) / self.lp_s)
        z = rng.standard_normal((n * m, self.latent_dim)).astype(np.float32)
        eps = rng.standard_normal(n * m).astype(np.float32)
        with torch.no_grad():
            xr = torch.tensor(np.repeat(X, m, axis=0), device=self.device)
            lr_ = torch.tensor(np.repeat(lp, m), device=self.device)
            mu, ls = self.net.decode(xr, lr_, torch.tensor(z, device=self.device))
            y = (mu + torch.exp(ls) * torch.tensor(eps, device=self.device)).cpu().numpy()
        y = y * self.y_s + self.y_m
        return np.floor(np.expm1(y)).clip(min=0).reshape(n, m)
