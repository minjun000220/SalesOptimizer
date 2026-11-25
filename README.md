# SalesOptimizer

특정 아이돌 그룹의 굿즈를 **몇 개, 어떤 가격에 만들면 안전하게 완판할 수 있는지** 추천한다.

- 입력: [IdolSentiments](https://github.com/minjun000220/IdolSentiments)의 그룹 점수(호감도·관심도·논란도…) + [Goods](https://github.com/minjun000220/Goods)의 굿즈 종류 점수(핫지수·언급량·품절 비율…) + 실제 판매 기록
- 방법: Gong, Liu & Zhang (2024) *Deep Generative Demand Learning for Newsvendor and Pricing* (arXiv:2411.08631)의 조건부 생성모델 newsvendor를 **완판 확률 제약**으로 바꾼 것

```
IdolSentiments ──(매일 저장: so_idol_snapshot)──┐
Goods (hot_score, mention, KREAM) ──────────────┼─▶ 특징 x ─┐
실제 판매 기록 CSV (나중에 추가) ─────────────────┘           ▼
                                   수요분포 생성기 G(x, p, η)  (앙상블)
                                                             ▼
                     가격 격자마다: 안전 수량 = 수요의 a 분위수 (a ≤ α, conformal 보정)
                                                             ▼
              추천: 가격 · 수량 · 완판 확률 · 기대이익 (+ 논문 원래 기준과 비교)
```

## 논문에서 바꾼 점

| | 논문 (Algorithm 1·2) | SalesOptimizer |
|---|---|---|
| 목표 | 기대이익 최대 | 기대이익 최대 **s.t. 완판 확률 ≥ 1 − α** |
| 수량 | 생성 샘플의 ρ(p) = (p−c)/(p−s) 분위수 | **min(a, ρ(p))** 분위수, a ≤ α |
| 모델 불확실성 | 생성기 1개 | 생성기 B개(bootstrap)의 분위수 중 **하위 10%** |
| 보장 | n, M → ∞ 일관성 | 판매 기록 n건으로 **conformal 보정** (유한 표본, 모델이 틀려도) |
| 완판 기록 | 다루지 않음 | **중도절단 우도** (완판 = 수요의 하한만 앎) |
| 가격 효과 | 학습 그대로 | **단조성**(비싸면 덜 팔림) + 선택적 **탄력성 하한** |
| 생성모델 | cGAN/cVAE | 데이터 적을 때 **중도절단 로그정규**, 많을 때 **cVAE** (같은 인터페이스) |

추가 요소:
- 최소 생산 수량(MOQ)과 고정비 반영. 안전 수량이 MOQ보다 작거나 이익이 안 나면 "만들지 않음"
- 학습 데이터 가격 범위 밖은 외삽이라고 경고. 추천 가격이 검토 범위 끝이면 알림

수식:

$$\max_{p,q}\ \mathbb E[\Pi(D,p,q)\mid x,p]\quad \text{s.t.}\quad P(D\ge q\mid x,p)\ge 1-\alpha \;\Rightarrow\; q^*(p)=F^{-1}_{D|x,p}\big(\min(\alpha,\rho(p))\big)$$

## 설치

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # cVAE 까지: pip install -e ".[dev,cvae]" / Supabase: ".[postgres]"
python -m pytest -q
```

## 실행

```bash
python -m salesopt demo --n 300                    # 가짜 데이터로 전체 흐름 (DB·네트워크 불필요)
python -m salesopt backtest --n-train 50,300,2000  # 정말 안전한지 검증 (정답을 아는 시뮬레이션)
python -m salesopt template --out sales.csv        # 실제 판매 기록 양식
python -m salesopt ingest-idol                     # IdolSentiments 결과를 DB 에 저장 (매일)
python -m salesopt import-sales --sales sales.csv   # 실제 판매 기록을 DB 에 저장
python -m salesopt publish                         # 웹이 쓸 모델·특징 올리기 (매일)
python -m salesopt recommend --group 아일릿 --sales sales.csv --types 키링,인형 \
    --cost 키링=4000,인형=11000 --fixed-cost 300000 --moq 50 --price 키링=9000~16000
python -m salesopt recommend --group 아일릿 --sales sales.csv           # 종류를 비우면 Goods 점수로 후보 선정
```

### 비용·가격 입력
한 값은 모든 굿즈 종류에 적용되고, `종류=값` 은 그 종류에만 적용됩니다. 쉼표로 여러 개를 섞을 수 있습니다 (`--cost 5000,인형=11000`).

| 옵션 | 뜻 | 기본값 |
|---|---|---|
| `--cost` | 개당 제작 원가 | 없음. 빠진 종류는 터미널에서 물어봄 |
| `--salvage` | 못 판 재고 개당 처분가 | 0 |
| `--fixed-cost` | 1회 생산 고정비 (디자인·금형 등) | 0 |
| `--moq` | 최소 생산 수량. 안전 수량이 이보다 적으면 "만들지 않음" | 1 |
| `--max-qty` | 생산 가능 최대 수량 | 제한 없음 |
| `--price` | 검토할 판매가 `하한~상한`. 값을 하나만 주면 그 가격으로 고정 | 판매 기록의 가격 범위 |
| `--price-step` | 가격 격자 간격 | 500원 |

천 단위 쉼표 대신 `4000`, `4_000`, `4000원` 처럼 씁니다. 쉼표는 종류 구분에 쓰기 때문입니다.

그 밖의 옵션: `--alpha 0.1` (완판 확률 90%), `--model lognormal|cvae`, `--min-elasticity 0.8`, `--censored strict|impute`, `--calib-frac 0.3`.
그룹과 Goods 갤러리의 연결은 `config/groups.yaml` 에 있습니다.

## 웹 화면 (굿즈 완판 계산기)

아이돌 그룹, 굿즈 종류, 개당 원가를 입력하면 다음을 보여 줍니다.
- 적정 판매가와 추천 생산 수량
- 그 가격에서의 예상 판매량(수요의 중앙값과 10~90% 범위)
- 완판 확률과 기대 이익
- 가격별 그래프와 표, 계산에 쓴 그룹·굿즈 점수

세부 설정에서 목표 완판 확률, 판매가 범위, 고정비, 최소·최대 생산 수량, 재고 처분가를 바꿀 수 있습니다.

```
매일 GitHub Actions: ingest-idol → publish  ─▶  Supabase (so_model: 학습된 모델 파라미터, so_context: 그룹×굿즈 특징)
웹 (Vercel, web/):   입력 → so_model + so_context 읽기 → 추천을 TypeScript 로 즉시 계산 (web/lib/optimize.ts)
```

- 웹은 Python 을 돌리지 않습니다. 매일 학습한 모델의 파라미터만 읽어서 바로 계산합니다(대기 없음).
- 계산 결과는 Python 과 같습니다. `web/lib/optimize.test.ts` 가 `tests/test_publish.py` 에서 만든 값과 비교합니다.
- 실제 판매 기록이 20건 미만이면 **가짜 데이터로 학습한 예시 모델**로 계산하고, 화면에 경고를 크게 표시합니다.
- 실제 판매 기록은 웹의 **판매 기록** 페이지(`/sales`)에서 넣습니다.
  - 한 줄씩 입력하거나 CSV 를 올립니다. 엑셀 기본 CSV(CP949)도 되고, 열 이름은 한글·영문 모두 됩니다.
  - 저장된 기록은 표로 보고 고치거나 지울 수 있습니다.
  - `ACCESS_PASSWORD` 가 설정돼 있어야 쓸 수 있고, 그 비밀번호를 아는 사람만 씁니다.
  - 명령줄로 넣으려면 `python -m salesopt import-sales --sales 판매기록.csv` 를 씁니다.
- 판매 기록은 다음 날 publish 때 모델에 반영됩니다. 바로 반영하려면 Actions 를 직접 실행합니다.

### Vercel 배포
1. vercel.com → **Add New → Project** → `SalesOptimizer` 저장소 Import
2. **Root Directory** 를 `web` 으로 지정합니다 (Framework 는 Next.js 로 자동 인식)
3. **Environment Variables**:
   - `DATABASE_URL`: Goods 와 같은 Supabase Session pooler 주소 (GitHub Secrets 에 넣은 것과 같은 값)
   - `ACCESS_PASSWORD`: 판매 기록 페이지를 쓰려면 꼭 넣어야 합니다. 넣으면 계산기도 이 비밀번호를 아는 사람만 쓸 수 있습니다
4. **Deploy** → 나온 주소를 폰·컴퓨터에서 엽니다

배포 전에 GitHub Actions 의 "아이돌 점수 저장 + 웹 데이터 갱신" 을 한 번 실행해야 화면에 그룹 목록이 나옵니다.

로컬 실행: `cd web && npm install && DATABASE_URL=... npm run dev` → http://localhost:3000 · 계산 테스트: `npm test`

## 매일 자동 저장 (GitHub Actions)

IdolSentiments는 최근 90일 추이만 남긴다. 그래서 나중에 들어올 판매 기록에 **그때의 그룹 점수**를 붙이려면 지금부터 날짜별로 쌓아야 한다.
1. Settings → Secrets and variables → Actions → `DATABASE_URL` 에 Goods 와 같은 Supabase Session pooler 주소를 넣는다
2. `.github/workflows/ingest.yml` 이 매일 06:43(KST)에 `so_idol_snapshot` 에 저장한다

## 시뮬레이션 백테스트 결과 (α = 0.1, 목표 완판 확률 90%)

lognormal 모델, 시드 3개 평균, 상황 80개. `profit` = 참 분포를 아는 "안전 기준 최선" 대비 기대이익 비율.

| 학습 기록 | 보정 기록 | 평균 완판 확률 | 목표 미달 상황 비율 | 이익 비율 | 비고 |
|---|---|---|---|---|---|
| 50 | – | 0.889 | 29% | 0.30 | 적은 데이터: 평균은 지키지만 보수적 |
| 50 | 50 | 0.921 | 24% | 0.21 | 보정하면 더 안전하고 더 보수적 |
| 100 | – | 0.891 | 28% | 0.43 | |
| 300 | – | 0.922 | 20% | 0.55 | |
| 300 | 100 | 0.929 | 15% | 0.64 | strict 보정 (impute: 0.927 / 0.65) |
| 2000 | – | 0.906 | 23% | 0.86 | |
| (논문 원래 기준) | – | 0.17~0.21 | – | 1.3~2.1 | 이익은 크지만 완판 확률이 20% 안팎 |

**숨은 내생성** (과거 가격이 판매자만 아는 수요 신호에 맞춰 정해진 경우, `--confounding`), 학습 2000건, 시드 2개, 상황 60개:

| 대응 | 평균 완판 확률 | 목표 미달 상황 비율 | 이익 비율 |
|---|---|---|---|
| 없음 | 0.832 | 67% | 1.04 |
| 보정 150건 | 0.860 | 58% | 0.97 |
| 탄력성 하한 0.8 | 0.880 | 36% | 0.84 |
| 탄력성 하한 0.8 + 보정 150건 | **0.916** | 18% | 0.74 |

→ 내생성은 가장 큰 위험이다. 가격을 무작위로 바꿔 보는 실험이 가장 확실한 해결책이고, 그게 어려우면 탄력성 하한과 보정을 함께 쓴다.

참고: 시뮬레이션 n=2000에서 cVAE를 보정 없이 쓰면 하위 꼬리를 낙관해 평균 완판 확률이 0.82로 떨어졌다(시드 1개). 그래서 **cVAE는 conformal 보정과 함께** 쓴다(보정 150건 → 0.94).

## 사전학습: 공개 판매 데이터로 가격 반응 배우기

아이돌 굿즈 판매 기록이 적을 때 가장 불확실한 것은 **가격을 올리면 수요가 얼마나 줄어드는지**(가격 탄력성)입니다.
그래서 가격과 판매량이 함께 있는 공개 데이터로 그 값의 출발점(사전분포)을 먼저 정하고,
실제 판매 기록이 쌓이면 그쪽으로 옮겨 가게 했습니다.

```bash
python -m salesopt pretrain --food <Food Demand 폴더> --retail <online_retail_ii.csv> [--m5 <M5 폴더>]
# → config/prior.json (학습·publish·recommend 가 자동으로 사용)
```

| 데이터 | 내용 | 가격 탄력성 | 상품 간 퍼짐 |
|---|---|---|---|
| [Kaggle Food Demand Forecasting](https://www.kaggle.com/datasets/kannanaikkal/food-demand-forecasting) | 배달 메뉴 × 지점, 145주, 45.6만 관측 (논문이 쓴 데이터) | 1.58 ± 0.02 | 0.88 |
| [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) | 영국 선물·소품 온라인몰, 상품 × 주, 17.3만 관측 | 0.98 ± 0.02 | 0.75 |
| [Kaggle M5](https://www.kaggle.com/competitions/m5-forecasting-accuracy) (HOBBIES) | 월마트 취미용품 | (대회 규칙 동의 후 추가 예정) | |
| **사전분포** | | **1.28 ± 0.92** | |

- 탄력성 1.28은 가격을 10% 올리면 판매량이 약 12% 줄어든다는 뜻입니다.
- 추정은 상품·주 양방향 고정효과로 했습니다. 같은 상품의 가격이 바뀌었을 때의 판매량 변화만 쓰므로, "원래 잘 팔리는 상품이 비싸다"나 계절 효과가 섞이지 않습니다.
- Online Retail 은 대량 구매 할인가가 섞이지 않도록 소매 단위(12개 이하) 거래의 가격만 씁니다.
- 퍼짐(±0.92)을 넓게 잡은 이유는 두 데이터의 추정치가 서로 다르고, 굿즈는 또 다를 수 있기 때문입니다. 그래서 실제 기록이 조금만 쌓여도 그쪽으로 움직입니다.
- 이 데이터들에서 빌려 오는 것은 **가격 반응뿐**입니다. 아이돌 점수·굿즈 화제성의 효과와 절대 수량은 실제 판매 기록에서 배웁니다.

## 판매 데이터 (나중에 추가)

`python -m salesopt template` 양식. 필수 열: group, goods_type, price, quantity_made, quantity_sold, sale_start

- 대략 20~50건이면 절대 수량 추천을 시작할 수 있고, 100건 이상이면 안전성을 숫자로 검증할 수 있다
- conformal 보정 최소 건수: α=0.1 → 9건, α=0.05 → 19건
- **완판되지 않은 기록**(남은 재고)이 꼭 필요하다. 가격도 여러 수준이어야 한다

판매 기록이 없을 때는 `salesopt.sources.goods.kream_proxy_records` 로 KREAM 거래 수 증가량을 대리 수요로 만들 수 있다. 다만 리셀 시장 수치라 1차 판매의 절대 수량으로는 믿을 수 없다(상대 비교용).

## 구조

```
salesopt/
  optimize.py       확률 제약 newsvendor (논문 Algorithm 1·2 변형), 비용·MOQ
  inputs.py         명령줄 비용·가격 입력 (전체 / 종류별)
  publish.py        웹용 모델 파라미터·특징 올리기 + 웹과 같은 결정적 추천 계산
  pretrain.py       공개 판매 데이터로 가격 탄력성 사전분포 추정 (양방향 고정효과)
  conformal.py      분위수 수준 conformal 보정 (중도절단 strict / impute)
  models/           lognormal(중도절단 Tobit, 교차검증 릿지, 탄력성 ≥ 하한), cvae(PyTorch), 앙상블
  simulate.py       정답을 아는 가짜 판매 데이터 (숨은 수요 충격·내생성 옵션)
  backtest.py       안전성·이익 검증
  features.py       (그룹, 종류, 날짜) → 특징. 판매 시작일 이전 정보만 사용 (누수 방지)
  sources/idol.py   IdolSentiments JSON → so_idol_snapshot
  sources/goods.py  Goods 의 hot_score, mention, KREAM 스냅샷 읽기
  tagger.py         상품 제목 → 그룹 (IdolSentiments 별칭)
  schema.py         판매 기록 CSV
  cli.py            명령줄
web/                Next.js 계산기 화면 (Vercel)
```

## 한계

- 보장은 보정 기록과 비슷한 상황 **전체의 평균**에 대한 것이다. 개별 상황마다 보장되지는 않는다(목표 미달 상황 비율 참고).
- 추천 가격을 고르는 과정 자체가 모델이 낙관하는 가격을 고르는 경향이 있다(optimizer's curse). 데이터가 적을수록 크다.
- IdolSentiments의 그룹 관심도는 수집량(유튜브 할당량)에 묶여 있어 그룹 간 비교가 어렵고, 감성 모델(앵커 zero-shot)은 검증되지 않았다. 특징으로 쓰기 전에 함께 개선하는 것이 좋다.
- Goods의 아이돌 갤러리 목록에 없는 그룹(IVE, 소녀시대, 리센느)은 그룹 × 굿즈 언급 특징이 비어 있다.
