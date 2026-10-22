"""b0 prediction-first test: two new profile families, not used in the curvature study.
Predictions (b0, b1, delta, G0) are computed from the derived local formulas only and written to
blind_predictions.json, which is committed to git BEFORE any zero-mode or scattering run on these profiles.
kappa = v'(0) = 1, horizon v(0) = -1, monotone v, supersonic (v < -1) for x < 0, subsonic for x > 0."""
import os, sys
import numpy as np
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))


class ErfSum:
    """v = -1 + sum_i A_i [erf((x - s_i)/w_i) + erf(s_i/w_i)],  Gaussian bumps in v'; A_i scaled so v'(0) = 1."""

    def __init__(self, comps, name):
        # comps: (weight_i, s_i, w_i); weight_i = share of v'(0)
        self.comps = []
        for wt, s, w in comps:
            dv0 = 2 / np.sqrt(np.pi) / w * np.exp(-(s / w) ** 2)
            self.comps.append((wt / dv0, s, w))
        self.kap = 1.0
        self.breaks = [0.0]
        self.name = name
        self.cauchy_r = 0.5 * min(w for _, _, w in comps)

    def v(self, x):
        return -1 + sum(A * (erf((x - s) / w) + erf(s / w)) for A, s, w in self.comps)

    def dv(self, x):
        return sum(A * 2 / np.sqrt(np.pi) / w * np.exp(-((x - s) / w) ** 2) for A, s, w in self.comps)

    def default_L(self):
        return max(abs(s) + 6.5 * w for _, s, w in self.comps)  # exp(-42)


class TanhCubic:
    """v = -1 + D tanh(p(x)),  p = x/D + beta x^2 + gamma x^3  (monotone if beta^2 < 3 gamma / D)."""

    def __init__(self, D, beta, gamma, name):
        assert beta ** 2 < 3 * gamma / D
        self.D, self.b, self.g = D, beta, gamma
        self.kap = 1.0
        self.breaks = [0.0]
        self.name = name
        self.cauchy_r = 0.12

    def p(self, x):
        return x / self.D + self.b * x ** 2 + self.g * x ** 3

    def v(self, x):
        return -1 + self.D * np.tanh(self.p(x))

    def dv(self, x):
        pp = 1 / self.D + 2 * self.b * x + 3 * self.g * x ** 2
        return self.D * pp / np.cosh(self.p(x)) ** 2

    def default_L(self):
        # |p(x)| >= 19 on both sides (sech^2 ~ 4e-17)
        from scipy.optimize import brentq
        xp = brentq(lambda x: self.p(x) - 19, 0, 100)
        xm = brentq(lambda x: self.p(x) + 19, -100, 0)
        return max(xp, -xm)


BLIND = {
    'blind_erf': lambda: ErfSum([(0.55, 0.35, 0.6), (0.45, -0.5, 1.0)], 'erfsum(0.55,0.35,0.6;0.45,-0.5,1.0)'),
    'blind_tcub': lambda: TanhCubic(0.6, -0.7, 0.5, 'tanhcubic(D=0.6,beta=-0.7,gamma=0.5)'),
}
