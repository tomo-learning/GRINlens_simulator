import numpy as np

def d2_model(d1, a, n1, L):
    c = np.cos(a*L)
    s = np.sin(a*L)
    num = c*d1 + s/(n1*a)
    den = c - n1*a*d1*s
    return -num/den

def fit_alpha_n1(d1, d2, L,
                 a_bounds=(1e-6, 3.0),   # 例: [1e-6, 1/mm] 等に調整
                 n1_bounds=(1.3, 2.0),   # 例: ガラスなら ~1.4–1.9
                 a0=0.05, n10=1.5):      # 初期値は実系に合わせて
    d1 = np.asarray(d1, float)
    d2 = np.asarray(d2, float)

    def sse(theta):
        a, n1 = theta
        # 非物理/特異点の回避
        if not (a_bounds[0] <= a <= a_bounds[1] and n1_bounds[0] <= n1 <= n1_bounds[1]):
            return np.inf
        pred = d2_model(d1, a, n1, L)
        # 極近で分母が0に近づくと暴れるので安全策
        if not np.all(np.isfinite(pred)):
            return np.inf
        return np.sum((pred - d2)**2)

    # 1) SciPy があれば使う
    try:
        from scipy.optimize import least_squares
        def resid(theta):  # 残差ベクトル
            a, n1 = theta
            pred = d2_model(d1, a, n1, L)
            return pred - d2
        res = least_squares(
            resid, x0=np.array([a0, n10]),
            bounds=([a_bounds[0], n1_bounds[0]],
                    [a_bounds[1], n1_bounds[1]]),
            method='trf', loss='linear'  # 必要に応じて 'soft_l1' など
        )
        a_hat, n1_hat = res.x
        return a_hat, n1_hat, res.cost*2/len(d1)  # 平均二乗誤差的な値
    except Exception:
        pass

    # 2) SciPy が無ければ：粗いグリッド→局所微調整（簡易）
    rng_a = np.linspace(a_bounds[0], a_bounds[1], 80)
    rng_n = np.linspace(n1_bounds[0], n1_bounds[1], 80)
    best = (np.inf, a0, n10)
    for a in rng_a:
        for n in rng_n:
            val = sse((a, n))
            if val < best[0]:
                best = (val, a, n)
    a_hat, n1_hat = best[1], best[2]

    # 近傍の微調整（小さなステップで数回）
    for _ in range(20):
        step_a = 0.2*(a_bounds[1]-a_bounds[0])/80
        step_n = 0.2*(n1_bounds[1]-n1_bounds[0])/80
        cand = [
            (a_hat, n1_hat),
            (a_hat+step_a, n1_hat),
            (a_hat-step_a, n1_hat),
            (a_hat, n1_hat+step_n),
            (a_hat, n1_hat-step_n),
        ]
        vals = [sse(x) for x in cand]
        idx = int(np.argmin(vals))
        if vals[idx] < sse((a_hat, n1_hat)):
            a_hat, n1_hat = cand[idx]
        else:
            break

    mse = sse((a_hat, n1_hat))/len(d1)
    return a_hat, n1_hat, mse

# 使い方例:
d1=[26.5,30.5,32,35,]

d2=[37,32,30,28] #実測配列（同じ長さ）、L: 既知の厚み（[mm] など単位一貫）
L=35.13
a_hat, n1_hat, mse = fit_alpha_n1(d1, d2, L, a_bounds=(1e-3, 2), n1_bounds=(0,10.0), a0=0.05, n10=1.6)
print(a_hat, n1_hat, mse)


