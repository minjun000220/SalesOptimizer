"""중도절단 로그정규 수요 모델 (데이터가 적을 때 기본 모델).

    y = log(1 + D) ~ N(μ, σ²)
    μ      = b₀ + x·b − e_g · (log p − c_g)      e_g = softplus(θ_g) ≥ 0  (굿즈 종류별 가격 탄력성)
    log σ  = a₀ + x·a                            (상황별로 불확실성이 다름)

완판 기록(판매량 = 생산량)은 D ≥ 판매량 만 알기 때문에 로그 생존확률 log P(Y ≥ y) 로 우도를 계산한다.
(Tobit 모형). 릿지 벌점으로 적은 데이터에서도 과적합을 줄인다.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

from salesopt.models.base import DemandModel


def _softplus(x):
    return np.logaddexp(0.0, x)


RIDGE_GRID = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


class LogNormalDemand(DemandModel):
    """ridge="auto" (기본): 릿지 강도를 5-fold 교차검증 (중도절단 로그우도)으로 고른다.
    데이터가 적을수록 강한 릿지가 골라져 과적합 → σ 과소추정 → 낙관적 수량 문제를 줄인다."""

    def __init__(self, ridge: float | str = "auto", ridge_sigma: float | None = None,
                 ridge_elasticity: float = 0.5, elasticity_prior: float = 1.0, cv_folds: int = 5,
                 min_elasticity: float = 0.0, elasticity_sd: float | None = None):
        """min_elasticity: 가격 탄력성 하한 (도메인 지식). 과거 가격이 '잘 팔릴 걸 알고' 매겨진 데이터(내생성)는
        탄력성을 과소추정해 비싼 가격을 낙관하게 된다 → 하한을 두면 그 위험이 줄어든다."""
        self.ridge, self.ridge_sigma, self.ridge_el = ridge, ridge_sigma, ridge_elasticity
        self.min_el = float(min_elasticity)
        # elasticity_sd 가 있으면: 가격 탄력성 e ~ N(elasticity_prior, elasticity_sd²) 사전분포 (공개 판매 데이터로 추정,
        # pretrain.py). 기록이 적으면 이 값 근처, 많아지면 데이터 쪽으로 옮겨 간다. 없으면 예전 방식 (θ 공간 릿지).
        self.el_prior, self.el_sd = float(elasticity_prior), elasticity_sd
        self.th_prior = np.log(np.expm1(max(elasticity_prior - self.min_el, 0.05)))     # softplus⁻¹
        self.cv_folds = cv_folds

    def fit(self, X, log_price, sold, censored, groups, n_groups: int | None = None):
        G = n_groups or int(np.max(groups)) + 1
        if self.ridge != "auto":
            return self._fit(X, log_price, sold, censored, groups, float(self.ridge), G=G)
        X, lp, sold = np.asarray(X, float), np.asarray(log_price, float), np.asarray(sold, float)
        cens, groups = np.asarray(censored, bool), np.asarray(groups, int)
        n = len(X)
        folds = np.random.default_rng(0).permutation(n) % self.cv_folds
        best, best_ll = RIDGE_GRID[-1], -np.inf
        for lam in RIDGE_GRID:
            ll = 0.0
            for f in range(self.cv_folds):
                tr, te = folds != f, folds == f
                if te.sum() == 0 or tr.sum() < 5:
                    continue
                self._fit(X[tr], lp[tr], sold[tr], cens[tr], groups[tr], lam, G=G)
                ll += self.loglik(X[te], lp[te], sold[te], cens[te], groups[te])
            if ll > best_ll:
                best, best_ll = lam, ll
        self.ridge_selected_ = best
        return self._fit(X, lp, sold, cens, groups, best, G=G)

    def loglik(self, X, log_price, sold, censored, groups) -> float:
        mu, sigma = self.params(np.asarray(X, float), log_price, groups)
        z = (np.log1p(np.asarray(sold, float)) - mu) / sigma
        return float(np.where(censored, norm.logsf(z), norm.logpdf(z) - np.log(sigma)).sum())

    def _unpack(self, w):
        d, G = self.d_, self.G_
        b0, b = w[0], w[1:1 + d]
        th = w[1 + d:1 + d + G]
        a0, a = w[1 + d + G], w[2 + d + G:]
        return b0, b, th, a0, a

    def _fit(self, X, log_price, sold, censored, groups, ridge: float, G: int | None = None):
        X = np.asarray(X, float)
        n, self.d_ = X.shape
        groups = np.asarray(groups, int)
        self.G_ = G or int(groups.max()) + 1
        r_b = ridge
        r_s = self.ridge_sigma if self.ridge_sigma is not None else 4.0 * ridge
        y = np.log1p(np.asarray(sold, float))
        cens = np.asarray(censored, bool)
        lp = np.asarray(log_price, float)
        self.lp_center_ = np.array([lp[groups == g].mean() if (groups == g).any() else lp.mean()
                                    for g in range(self.G_)])
        dlp = lp - self.lp_center_[groups]
        onehot_g = np.eye(self.G_)[groups]

        def nll(w):
            b0, b, th, a0, a = self._unpack(w)
            e = self.min_el + _softplus(th)
            mu = b0 + X @ b - e[groups] * dlp
            ls = np.clip(a0 + X @ a, -4, 3)
            s = np.exp(ls)
            z = (y - mu) / s
            # 값
            lik = np.where(cens, norm.logsf(z), norm.logpdf(z) - ls)
            # 기울기: dnll/dμ, dnll/dlogσ
            haz = np.exp(norm.logpdf(z) - norm.logsf(z))
            g_mu = np.where(cens, -haz / s, -z / s)
            g_ls = np.where(cens, -haz * z, 1.0 - z * z)
            g_ls = np.where((a0 + X @ a > 3) | (a0 + X @ a < -4), 0.0, g_ls)
            grad = np.concatenate([
                [g_mu.sum()],
                X.T @ g_mu,
                onehot_g.T @ (g_mu * (-dlp)) * expit(th),
                [g_ls.sum()],
                X.T @ g_ls,
            ])
            if self.el_sd:
                de = e - self.el_prior
                pen_e, gpen_e = (de @ de) / (2 * self.el_sd ** 2), de / self.el_sd ** 2 * expit(th)
            else:
                dt = th - self.th_prior
                pen_e, gpen_e = self.ridge_el * (dt @ dt), 2 * self.ridge_el * dt
            pen = r_b * (b @ b) + r_s * (a @ a) + pen_e
            gpen = np.concatenate([[0.0], 2 * r_b * b, gpen_e, [0.0], 2 * r_s * a])
            return -lik.sum() + pen, grad + gpen          # grad 는 이미 음의 로그우도 기준

        # 초기값: 완판 아닌 기록으로 최소제곱
        unc = ~cens if (~cens).sum() > self.d_ + 2 else np.ones(n, bool)
        A = np.hstack([np.ones((unc.sum(), 1)), X[unc]])
        coef = np.linalg.lstsq(A.T @ A + np.eye(A.shape[1]), A.T @ y[unc], rcond=None)[0]
        resid = y[unc] - A @ coef
        w0 = np.concatenate([coef, np.full(self.G_, self.th_prior), [np.log(resid.std() + 1e-3)],
                             np.zeros(self.d_)])
        res = minimize(nll, w0, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
        self.w_ = res.x
        self.converged_ = bool(res.success)
        return self

    def params(self, X, log_price, groups):
        b0, b, th, a0, a = self._unpack(self.w_)
        groups = np.asarray(groups, int)
        e = self.min_el + _softplus(th)
        mu = b0 + X @ b - e[groups] * (np.asarray(log_price) - self.lp_center_[groups])
        sigma = np.exp(np.clip(a0 + X @ a, -4, 3))
        return mu, sigma

    def elasticity(self) -> np.ndarray:
        return self.min_el + _softplus(self._unpack(self.w_)[2])

    def sample(self, X, log_price, groups, m, rng):
        mu, sigma = self.params(np.asarray(X, float), log_price, groups)
        y = mu[:, None] + sigma[:, None] * rng.standard_normal((len(mu), m))
        return np.floor(np.expm1(y)).clip(min=0)
