import numpy as np
import pandas as pd


def simulate_stepdown(prices, assets, strikes=(0.90, 0.90, 0.85, 0.85, 0.80, 0.75),
                      ki=0.50, obs_months=6, start=None, end=None):
    """매 거래일 ELS가 하나씩 발행됐다고 가정하고 각각의 결과를 계산"""
    px = prices[assets].dropna()
    dates = px.index
    vals = px.values
    n_obs = len(strikes)

    issue_dates = dates
    if start is not None:
        issue_dates = issue_dates[issue_dates >= pd.Timestamp(start)]
    if end is not None:
        issue_dates = issue_dates[issue_dates <= pd.Timestamp(end)]

    rows = []
    for t0 in issue_dates:
        # ① 평가일 6개 만들기
        obs_dates = [t0 + pd.DateOffset(months=obs_months * (k + 1)) for k in range(n_obs)]
        if obs_dates[-1] > dates[-1]:
            break   # 만기가 아직 안 온 ELS는 제외

        i0 = dates.get_loc(t0)
        obs_idx = dates.searchsorted(obs_dates)
        base = vals[i0]

        # ② 6개월마다 조기상환 평가
        result = None
        end_idx = obs_idx[-1]                      # 기본값: 만기일
        for k, (oi, s) in enumerate(zip(obs_idx, strikes)):
            worst = (vals[oi] / base).min()
            if worst >= s:
                kind = '조기상환' if k < n_obs - 1 else '만기상환(행사가)'
                result = (kind, k + 1, 1.0)
                end_idx = oi                       # 상환된 날까지만 보유
                break

        # ③ 실제 보유 기간 동안의 최악 지수 경로 (수정된 부분)
        worst_path = (vals[i0:end_idx + 1] / base).min(axis=1)

        # ④ 끝까지 상환 못 했으면 만기 판정
        if result is None:
            ki_hit = True if ki is None else worst_path.min() < ki
            worst_final = (vals[obs_idx[-1]] / base).min()
            if not ki_hit:
                result = ('만기상환(원금)', n_obs, 1.0)
            else:
                result = ('손실', n_obs, worst_final)

        kind, k, principal = result
        rows.append({
            'issue_date': t0,
            'outcome': kind,
            'redeem_step': k,
            'years': k * obs_months / 12,
            'principal': principal,
            'coupon_years': k * obs_months / 12 if kind in ('조기상환', '만기상환(행사가)') else 0.0,
            'ki_touch': ki is not None and worst_path.min() < ki,
            'worst_min': worst_path.min(),
        })
    return pd.DataFrame(rows)


def apply_coupon(res, coupon):
    """연 쿠폰을 대입해 상환금액과 연환산 수익률 계산"""
    payoff = res['principal'] + coupon * res['coupon_years']   # 돌려받은 돈 (원금=1)
    ann = payoff ** (1 / res['years']) - 1                      # 연환산 수익률
    return payoff, ann


def required_coupon(res, benchmark=0.03, lo=0.0, hi=0.50, tol=1e-5):
    """평균 연환산 수익률이 benchmark와 같아지는 최소 쿠폰 = 필요 쿠폰"""
    f = lambda c: apply_coupon(res, c)[1].mean() - benchmark
    if f(hi) < 0:
        return np.nan        # 쿠폰 50%를 줘도 안 되는 구조
    if f(lo) >= 0:
        return 0.0
    while hi - lo > tol:     # 이분법: 범위를 반씩 좁혀가며 정답 찾기
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return hi


def summarize(res, coupon=None):
    """결과 표를 핵심 지표 한 줄로 요약"""
    loss = res['outcome'] == '손실'
    out = {
        '발행건수': len(res),
        '조기상환율': (res['outcome'] == '조기상환').mean(),
        '1차상환율': (res['redeem_step'] == 1).mean(),
        '평균보유(년)': res['years'].mean(),
        '녹인터치율': res['ki_touch'].mean(),
        '손실확률': loss.mean(),
        '손실시평균손실률': 1 - res.loc[loss, 'principal'].mean() if loss.any() else 0.0,
    }
    if coupon is not None:
        out['연환산수익률(평균)'] = apply_coupon(res, coupon)[1].mean()
    return pd.Series(out)

