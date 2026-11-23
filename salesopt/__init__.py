"""SalesOptimizer: 아이돌 그룹 굿즈를 몇 개, 어떤 가격에 만들면 안전하게 완판할 수 있는지 추천한다.

Gong, Liu & Zhang (2024) "Deep Generative Demand Learning for Newsvendor and Pricing" 의 조건부 생성모델
newsvendor 를 '완판 확률 ≥ 1 − α' 확률 제약으로 바꾸고, IdolSentiments(그룹 점수)와 Goods(굿즈 점수)를 입력으로 쓴다.
"""
__version__ = "0.1.0"
