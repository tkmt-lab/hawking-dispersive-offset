"""b0, step check: the zero-frequency quantities at finite Lambda, computed directly.

psi'' = Lam^2 (1 - v^2) psi, psi decaying for x -> +inf (integrated from the exterior inwards),
  N = int psi dx, M = int (-v) psi dx   (tails beyond x = -L added in closed form:
      int_{-inf}^{-L} psi = -psi'(-L)/k^2 for constant v, Abel-regularised),
  psi = b e^{ikx} + c.c. at x = -L,  k = Lam sqrt(v_-^2 - 1),  |W_c| = 2 k |b|^2,
  F = |W_c| / (2 Lam^2 N^2)                 (hard-mode flux of f0 = int psi / N),
  delta = M/N - 1 = -int (1+v) psi / N,
  T(0) = F / (1 + delta) = |W_c| / (2 Lam^2 N M),
  predicted S-matrix limits:  G0 = |A|^2 -> (delta/(2+delta))^2,  w|beta|^2 -> 4F/(2+delta)^2.
kappa = v'(0) = 1 for all profiles used here."""
import os, sys
import numpy as np
from scipy.integrate import solve_ivp, quad

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))


def zero_mode(prof, Lam, L=None, rtol=1e-12, atol=1e-14, xr_exp=48.0):
    if L is None:
        L = prof.default_L()
    # exterior start: where Lam * int_0^x sqrt(1-v^2) reaches xr_exp (psi there ~ e^-xr_exp of its horizon size)
    g = lambda x: np.sqrt(max(1 - float(prof.v(x)) ** 2, 0.0))
    xs = np.linspace(0, L, 4001)
    cum = np.concatenate([[0], np.cumsum(0.5 * (np.array([g(a) for a in xs[1:]]) + np.array([g(a) for a in xs[:-1]])) * np.diff(xs))])
    i = np.searchsorted(Lam * cum, xr_exp)
    xR = xs[min(i, len(xs) - 1)]
    vR = float(prof.v(xR))
    qR = Lam * np.sqrt(1 - vR * vR)

    def rhs(x, y):
        v = float(prof.v(x))
        return [y[1], Lam ** 2 * (1 - v * v) * y[0], y[0], -v * y[0]]

    sol = solve_ivp(rhs, (xR, -L), [1.0, -qR, 0.0, 0.0], method='DOP853', rtol=rtol, atol=atol)
    psi, dpsi, In, Im = sol.y[:, -1]
    # the ODE integrated from xR down to -L gives -(int_{-L}^{xR})
    In, Im = -In, -Im
    vL = float(prof.v(-L))
    k = Lam * np.sqrt(vL * vL - 1)
    tail = -dpsi / k ** 2
    N = In + tail
    M = Im + (-vL) * tail
    b = 0.5 * (psi - 1j * dpsi / k)
    Wc = 2 * k * abs(b) ** 2
    F = Wc / (2 * Lam ** 2 * N ** 2)
    delta = M / N - 1
    T0 = Wc / (2 * Lam ** 2 * N * M)
    TH = 1 / (2 * np.pi)
    return dict(Lam=Lam, L=L, xR=float(xR), N=N, M=M, Wc=Wc, F_over_TH=F / TH, delta=delta,
                T0_over_TH=T0 / TH, G0_pred=(delta / (2 + delta)) ** 2, Cbeta_pred_over_TH=4 * F / (2 + delta) ** 2 / TH,
                nfev=int(sol.nfev))
