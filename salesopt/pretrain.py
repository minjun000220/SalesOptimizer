"""공개 판매 데이터로 '가격 반응'(가격 탄력성)의 출발값(사전분포)을 추정한다.

데이터 (아이돌 굿즈가 아니므로 '가격을 올리면 수요가 얼마나 줄어드는가' 만 빌려 온다):
  food   : Kaggle Food Demand Forecasting — 배달 메뉴 × 지점, 주별 주문 수·판매가·홍보 여부
  retail : UCI Online Retail II — 영국 선물·소품 온라인몰, 상품별 거래 (수량·개당 가격)
  m5     : Kaggle M5 (월마트) — 상품 × 매장, 일별 판매량·주별 판매가 (HOBBIES 분류만 사용)

추정 (양방향 고정효과):
  log(1 + 수량) = 상품·지점 효과 + 주 효과 − e · log(가격) (+ 홍보 여부) + 오차
  - 상품 효과: '원래 잘 팔리는 상품이 비싸다' 같은 상품 간 차이를 지운다
  - 주 효과  : 계절·전체 수요 변화를 지운다
  → 같은 상품의 가격이 바뀌었을 때 수량이 얼마나 바뀌었나만 남는다.
  상품마다 탄력성이 다르므로 상품별 기울기의 분산(측정오차를 뺀 값)으로 '탄력성의 퍼짐'도 구한다.

결과(config/prior.json)는 LogNormalDemand 의 가격 탄력성 사전분포로 쓰인다:
  실제 판매 기록이 적을 때는 이 값 근처로, 기록이 쌓이면 실제 데이터 쪽으로 옮겨 간다.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PRIOR_PATH = ROOT / "config" / "prior.json"


# --------------------------------------------------------------------------- 추정
def _demean_2way(df: pd.DataFrame, cols: list[str], unit: str, time: str, tol=1e-9, max_iter=200) -> pd.DataFrame:
    """상품·주 고정효과를 번갈아 빼서 제거 (불균형 패널용 교대 투영)."""
    out = df[cols].astype(float).copy()
    for _ in range(max_iter):
        before = out.to_numpy().copy()
        out -= out.groupby(df[unit]).transform("mean")
        out -= out.groupby(df[time]).transform("mean")
        if np.abs(out.to_numpy() - before).max() < tol:
            break
    return out


@dataclass
class Estimate:
    source: str
    elasticity: float          # 합동 추정 (양수 = 비싸지면 덜 팔림)
    se: float                  # 상품 단위 군집 표준오차
    spread: float              # 상품 간 탄력성의 표준편차 (측정오차 제외)
    resid_sigma: float         # 고정효과·가격을 뺀 뒤 log(1+수량) 의 흔들림
    n_obs: int
    n_units: int
    by_group: dict = field(default_factory=dict)
    note: str = ""


def estimate(df: pd.DataFrame, unit: str, time: str, source: str, controls: tuple[str, ...] = (),
             group: str | None = None, min_obs: int = 8, note: str = "") -> Estimate:
    """df: [unit, time, q(수량), p(가격), controls…]"""
    d = df[(df.q >= 0) & (df.p > 0)].copy()
    d["y"] = np.log1p(d.q)
    d["x"] = np.log(d.p)
    # 가격이 한 번도 안 바뀐 상품·관측이 적은 상품은 기울기에 정보가 없다
    g = d.groupby(unit)
    keep = (g.x.transform("std") > 1e-6) & (g.x.transform("size") >= min_obs)
    d = d[keep].reset_index(drop=True)
    cols = ["y", "x", *controls]
    w = _demean_2way(d, cols, unit, time)
    controls = tuple(c for c in controls if w[c].std() > 1e-9)        # 변화 없는 통제변수는 뺀다
    X = w[["x", *controls]].to_numpy()
    y = w.y.to_numpy()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    # 상품 단위 군집 표준오차 (가격 계수)
    XtX_inv = np.linalg.inv(X.T @ X)
    scores = pd.DataFrame(X * resid[:, None]).groupby(d[unit].to_numpy()).sum().to_numpy()
    V = XtX_inv @ (scores.T @ scores) @ XtX_inv
    e, se = -float(beta[0]), float(math.sqrt(V[0, 0]))
    dof = len(d) - d[unit].nunique() - d[time].nunique() - X.shape[1]
    sigma = float(math.sqrt((resid ** 2).sum() / max(dof, 1)))

    # 상품별 기울기 → 퍼짐 (경험적 베이즈: 관측된 분산 − 평균 측정오차 분산)
    yr = w.y.to_numpy() - (w[list(controls)].to_numpy() @ beta[1:] if controls else 0)
    tmp = pd.DataFrame({"u": d[unit].to_numpy(), "x": w.x.to_numpy(), "y": yr})
    slopes, ses = [], []
    for _, gg in tmp.groupby("u"):
        xc = gg.x - gg.x.mean()
        sxx = float((xc ** 2).sum())
        if len(gg) < min_obs or sxx < 1e-4:
            continue
        b = float((xc * (gg.y - gg.y.mean())).sum() / sxx)
        r = gg.y - gg.y.mean() - b * xc
        s2 = float((r ** 2).sum() / max(len(gg) - 2, 1))
        slopes.append(-b)
        ses.append(s2 / sxx)
    slopes, ses = np.array(slopes), np.array(ses)
    if len(slopes) > 5:
        # 가격 변화가 아주 작아 기울기가 부정확한 상품은 '정밀도' 기준으로 뺀다 (기울기 값으로 자르면 퍼짐이 줄어 편향)
        m = ses <= np.quantile(ses, 0.9)
        spread = float(math.sqrt(max(slopes[m].var() - ses[m].mean(), 0.0)))   # 관측 분산 − 평균 측정오차 분산
    else:
        spread = float("nan")

    by_group = {}
    if group:
        for gname, gd in df.groupby(group):
            if gd[unit].nunique() < 20:
                continue
            sub = estimate(gd, unit, time, f"{source}:{gname}", controls, None, min_obs)
            by_group[str(gname)] = {"elasticity": round(sub.elasticity, 3), "se": round(sub.se, 3),
                                    "n_units": sub.n_units}
    return Estimate(source, e, se, spread, sigma, len(d), int(d[unit].nunique()), by_group, note)


# --------------------------------------------------------------------------- 데이터별 준비
def prepare_food(folder: Path) -> pd.DataFrame:
    tr = pd.read_csv(folder / "train.csv")
    meal = pd.read_csv(folder / "meal_info.csv")
    tr = tr.merge(meal[["meal_id", "category"]], on="meal_id", how="left")
    return pd.DataFrame({
        "unit": tr.center_id.astype(str) + ":" + tr.meal_id.astype(str), "time": tr.week,
        "q": tr.num_orders, "p": tr.checkout_price,
        "emailer": tr.emailer_for_promotion, "homepage": tr.homepage_featured, "category": tr.category,
    })


NON_PRODUCT = {"POST", "DOT", "M", "C2", "D", "S", "B", "CRUK", "PADS", "BANK CHARGES", "AMAZONFEE", "ADJUST",
               "ADJUST2", "TEST001", "TEST002", "m"}


def prepare_retail(csv: Path, retail_max_qty: int = 12) -> pd.DataFrame:
    """상품 × 주. 수량 = 그 주 전체 판매 수량, 가격 = 소매 단위(한 번에 12개 이하) 거래의 최빈 가격.
    대량 구매 할인가(도매)를 가격으로 쓰면 '많이 살수록 싸다' 가 탄력성처럼 보이므로 소매 단위 가격만 쓴다."""
    df = pd.read_csv(csv, dtype={"Invoice": str, "StockCode": str})
    df = df[~df.Invoice.str.startswith(("C", "A"))]
    df = df[(df.Quantity > 0) & (df.Price > 0)]
    code = df.StockCode.str.upper()
    df = df[~code.isin({c.upper() for c in NON_PRODUCT}) & ~code.str.startswith(("GIFT", "TEST", "DCGS"))]
    df["week"] = pd.to_datetime(df.InvoiceDate).dt.to_period("W").astype(str)
    qty = df.groupby(["StockCode", "week"]).Quantity.sum()
    small = df[df.Quantity <= retail_max_qty]
    price = small.groupby(["StockCode", "week"]).Price.agg(lambda s: s.round(2).mode().iloc[0])
    out = pd.concat([qty.rename("q"), price.rename("p")], axis=1).dropna().reset_index()
    return out.rename(columns={"StockCode": "unit", "week": "time"})


def prepare_m5(folder: Path, category: str = "HOBBIES") -> pd.DataFrame:
    """M5 에서 HOBBIES(취미용품) 분류만: 상품 × 매장 × 주 판매량, 그 주 판매가."""
    sales = pd.read_csv(folder / "sales_train_evaluation.csv") if (folder / "sales_train_evaluation.csv").exists() \
        else pd.read_csv(folder / "sales_train_validation.csv")
    sales = sales[sales.cat_id == category]
    cal = pd.read_csv(folder / "calendar.csv")[["d", "wm_yr_wk"]]
    prices = pd.read_csv(folder / "sell_prices.csv")
    long = sales.melt(id_vars=["item_id", "store_id", "dept_id"], value_vars=[c for c in sales if c.startswith("d_")],
                      var_name="d", value_name="q").merge(cal, on="d")
    wk = long.groupby(["item_id", "store_id", "dept_id", "wm_yr_wk"]).q.sum().reset_index()
    wk = wk.merge(prices, on=["item_id", "store_id", "wm_yr_wk"])           # 판매가 없는 주(미판매) 제외
    return pd.DataFrame({"unit": wk.item_id + ":" + wk.store_id, "time": wk.wm_yr_wk, "q": wk.q,
                         "p": wk.sell_price, "dept": wk.dept_id})


# --------------------------------------------------------------------------- 합치기
@dataclass
class DemandPrior:
    elasticity_mean: float
    elasticity_sd: float
    estimates: list
    created_at: str
    note: str

    def save(self, path: Path = PRIOR_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def load_prior(path: Path = PRIOR_PATH) -> DemandPrior | None:
    if not path.exists():
        return None
    return DemandPrior(**json.loads(path.read_text(encoding="utf-8")))


def combine(estimates: list[Estimate]) -> DemandPrior:
    """데이터별 탄력성을 동일 가중 평균. 퍼짐 = 상품 간 퍼짐과 데이터 간 차이를 합친 것
    (굿즈가 이 데이터들과 다를 수 있으므로 넓게 잡는다)."""
    es = np.array([e.elasticity for e in estimates])
    within = np.array([e.spread for e in estimates if not math.isnan(e.spread)])
    mean = float(es.mean())
    between = float(es.std(ddof=1)) if len(es) > 1 else 0.0
    sd = float(math.sqrt((within ** 2).mean() + between ** 2)) if len(within) else max(between, 0.5)
    return DemandPrior(round(mean, 4), round(max(sd, 0.3), 4), [asdict(e) for e in estimates],
                       datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                       "가격 탄력성 사전분포. 아이돌 굿즈가 아닌 데이터에서 추정했으므로 실제 판매 기록으로 보정된다.")
