"""Boundary-value implementation of the dispersive Hawking scattering problem.

Method: global boundary-value problem on [-L, L] solved by multi-domain Chebyshev
collocation (all elements and boundary conditions in one sparse linear system).
This is NOT a marching/shooting integration: the four boundary conditions
(two at each end) are imposed simultaneously, so exponentially growing and
decaying evanescent modes are controlled by the conditions at the end where
each would blow up.

Wave equation (CPF 2012 eq. (9), DPLS 2026 eq. (5); c_s = 1):
    (d_t + d_x v)(d_t + v d_x) phi = d_x^2 phi - d_x^4 phi / Lam^2
Stationary mode phi = exp(-i w t) f(x):
    f'''' = Lam^2 [ (w^2 + i w v') f + (2 i w v - 2 v v') f' + (1 - v^2) f'' ]
Plane wave exp(i k x) at constant v:  (w - v k)^2 = k^2 + k^4 / Lam^2.

Conserved current (derived by hand, see report; J' = 0 for any two solutions):
    J(f1,f2) = v [ f1* Df2 - f2 (Df1)* ] - (f1* f2' - f2 f1*')
               + (f1* f2''' - f1*' f2'' + f1*'' f2' - f1*''' f2) / Lam^2,
    D f = -i w f + v f'.
For a plane wave a exp(ikx):  (i/2) J = Omega_c * v_g * |a|^2,
    Omega_c = w - v k (comoving frequency; its sign = sign of the norm),
    v_g = dw/dk = (v Omega_c + k + 2 k^3/Lam^2) / Omega_c (lab group velocity).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import erf

# ---------------------------------------------------------------- profiles


class Tanh:
    """v = -1 + D tanh(kap x / D)  (v''(0) = 0, v'''(0) = -2 kap^3 / D^2)."""

    def __init__(self, D, kap=1.0):
        self.D, self.kap = D, kap
        self.breaks = [0.0]
        self.name = f"tanh(D={D})"

    def v(self, x):
        return -1 + self.D * np.tanh(self.kap * x / self.D)

    def dv(self, x):
        return self.kap / np.cosh(self.kap * x / self.D) ** 2

    def default_L(self):
        return 16.0 * self.D / self.kap

    def vpp0(self):
        return 0.0


class LinCore:
    """Exactly linear core: v = -1 + kap x for |x| <= a; outside, v' = kap exp(-((|x|-a)/w)^2).
    v, v', v'' continuous; v''' jumps at |x| = a (an element boundary is put there).
    Asymptotes: v(+-inf) = -1 +- kap (a + w sqrt(pi)/2)."""

    def __init__(self, a, w, kap=1.0):
        self.a, self.w, self.kap = a, w, kap
        self.breaks = [-a, 0.0, a]
        self.name = f"lincore(a={a},w={w})"

    def v(self, x):
        x = np.asarray(x, float)
        ax = np.abs(x)
        inner = -1 + self.kap * x
        outer = -1 + np.sign(x) * self.kap * (self.a + self.w * np.sqrt(np.pi) / 2 * erf((ax - self.a) / self.w))
        return np.where(ax <= self.a, inner, outer)

    def dv(self, x):
        x = np.asarray(x, float)
        ax = np.abs(x)
        return np.where(ax <= self.a, self.kap, self.kap * np.exp(-((ax - self.a) / self.w) ** 2))

    def default_L(self):
        return self.a + 7.0 * self.w  # exp(-49) ~ 5e-22

    def vpp0(self):
        return 0.0


class FlatCore:
    """Analytic profile with an increasingly linear core:
        v'(x) = kap exp(-(x/sig)^(2m)),  v(x) = -1 + kap * int_0^x v'/kap.
    v''(0) = 0 exactly; v(x) - (-1 + kap x) = O(x^(2m+1)).  m=1 is a Gaussian
    (like tanh, v''' != 0); m=2,3 flatten the core.  Entire function: no kinks.
    Asymptotes: -1 +- kap sig Gamma(1 + 1/(2m))."""

    def __init__(self, m, sig, kap=1.0):
        from scipy.special import gamma as G
        self.m, self.sig, self.kap = m, sig, kap
        self.breaks = [0.0]
        self.name = f"flatcore(m={m},sig={sig})"
        self._G = G(1.0 / (2 * m))

    def v(self, x):
        from scipy.special import gammainc
        x = np.asarray(x, float)
        p = 1.0 / (2 * self.m)
        integ = self.sig * p * self._G * gammainc(p, (np.abs(x) / self.sig) ** (2 * self.m))
        return -1 + self.kap * np.sign(x) * integ

    def dv(self, x):
        x = np.asarray(x, float)
        return self.kap * np.exp(-(np.abs(x) / self.sig) ** (2 * self.m))

    def default_L(self):
        return self.sig * 45.0 ** (1.0 / (2 * self.m))

    def vpp0(self):
        return 0.0


class SumTanh:
    """Sum of shifted tanh steps, each with partial slope kap_i at x=0:
        v = -1 + sum_i A_i [tanh(q_i (x - s_i)) + t_i],  t_i = tanh(q_i s_i),
        A_i = kap_i / (q_i (1 - t_i^2)),  v'(0) = sum kap_i,
        v''(0) = sum 2 kap_i q_i t_i.
    Used for asymmetric profiles with v''(0) = 0."""

    def __init__(self, comps):
        self.comps = [(k, q, t, np.arctanh(t) / q, k / (q * (1 - t * t))) for (k, q, t) in comps]
        self.kap = sum(c[0] for c in self.comps)
        self.breaks = [0.0]
        self.name = "sumtanh(" + ";".join(f"k={k},q={q},t={t}" for (k, q, t) in comps) + ")"

    def v(self, x):
        return -1 + sum(A * (np.tanh(q * (x - s)) + t) for (k, q, t, s, A) in self.comps)

    def dv(self, x):
        return sum(A * q / np.cosh(q * (x - s)) ** 2 for (k, q, t, s, A) in self.comps)

    def default_L(self):
        return max(abs(s) + 17.0 / q for (k, q, t, s, A) in self.comps)

    def vpp0(self):
        return sum(2 * k * q * t for (k, q, t, s, A) in self.comps)


class ShiftTanh:
    """v = -1 + A [tanh(q (x - s)) + tanh(q s)], scaled so v'(0) = kap.
    Parametrised by q and t = tanh(q s):  A = kap / (q (1 - t^2)),
    v''(0) = 2 kap q t, v(+inf) = -1 + kap/(q(1-t)), v(-inf) = -1 - kap/(q(1+t))."""

    def __init__(self, q, t, kap=1.0):
        self.q, self.t, self.kap = q, t, kap
        self.s = np.arctanh(t) / q
        self.A = kap / (q * (1 - t * t))
        self.breaks = [0.0]
        self.name = f"shifttanh(q={q},t={t})"

    def v(self, x):
        return -1 + self.A * (np.tanh(self.q * (x - self.s)) + self.t)

    def dv(self, x):
        return self.A * self.q / np.cosh(self.q * (x - self.s)) ** 2

    def default_L(self):
        return abs(self.s) + 17.0 / self.q  # sech^2 ~ 4 exp(-34)

    def vpp0(self):
        return 2 * self.kap * self.q * self.t


# ---------------------------------------------------------------- local modes


def roots(w, v, Lam):
    """Roots k of (w - v k)^2 = k^2 + k^4/Lam^2, polished by Newton."""
    c = [1 / Lam**2, 0.0, 1 - v * v, 2 * w * v, -w * w]
    ks = np.roots(c).astype(complex)
    for _ in range(4):
        P = ks**4 / Lam**2 + (1 - v * v) * ks**2 + 2 * w * v * ks - w * w
        dP = 4 * ks**3 / Lam**2 + 2 * (1 - v * v) * ks + 2 * w * v
        ks = ks - P / dP
    return ks


def mode_props(k, w, v, Lam):
    Oc = w - v * k
    vg = (v * Oc + k + 2 * k**3 / Lam**2) / Oc
    return Oc, vg


def classify(w, v, Lam, side, tol=1e-9):
    """Return dict label -> k for the four local solutions at an asymptotic end.
    side='R' (x=+L, subsonic): 'u' out (vg>0), 'vin' in (vg<0), 'grow' (Im k<0), 'dec' (Im k>0).
    side='L' (x=-L, supersonic): 'red' in (vg>0, norm+), 'or' in (vg>0, norm-),
                                 'vout' out (vg<0, norm+), 'P' out (vg<0, norm-)."""
    ks = roots(w, v, Lam)
    scale = max(1.0, np.max(np.abs(ks)))
    real = [k.real for k in ks if abs(k.imag) < tol * scale]
    cplx = [k for k in ks if abs(k.imag) >= tol * scale]
    out = {}
    if side == 'R':
        if len(real) != 2:
            raise ValueError(f"expected 2 real roots at +L, got {ks}")
        for k in real:
            Oc, vg = mode_props(k, w, v, Lam)
            if Oc <= 0:
                raise ValueError("negative norm real mode on subsonic side?")
            out['u' if vg > 0 else 'vin'] = k
        for k in cplx:
            out['grow' if k.imag < 0 else 'dec'] = k
        if set(out) != {'u', 'vin', 'grow', 'dec'}:
            raise ValueError(f"bad classification R: {ks}")
    else:
        if len(real) != 4:
            return None  # w > w_max
        for k in real:
            Oc, vg = mode_props(k, w, v, Lam)
            if vg > 0:
                out['red' if Oc > 0 else 'or'] = k
            else:
                out['vout' if Oc > 0 else 'P'] = k
        if set(out) != {'red', 'or', 'vout', 'P'}:
            raise ValueError(f"bad classification L: {ks}")
    return out


# ---------------------------------------------------------------- Chebyshev


def cheb(N):
    """Chebyshev-Gauss-Lobatto nodes x_j = cos(pi j/N) and differentiation matrix (Trefethen)."""
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1)
    c[0] = c[-1] = 2
    c *= (-1) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    Dm = np.outer(c, 1 / c) / (dX + np.eye(N + 1))
    Dm -= np.diag(Dm.sum(axis=1))
    return x, Dm


def build_mesh(prof, w, Lam, L, N, gamma, hmax):
    """Element boundaries from -L to L; element length ~ gamma*N/kmax(x), capped by hmax,
    with mandatory breakpoints of the profile."""
    brk = sorted(set([b for b in prof.breaks if -L < b < L]))
    pts = [-L]
    targets = brk + [L]
    x = -L
    for tgt in targets:
        while x < tgt - 1e-12:
            # local resolution need: max |k| over a few points in the next candidate element
            h = hmax
            for _ in range(3):
                xs = np.linspace(x, min(x + h, tgt), 5)
                km = max(np.max(np.abs(roots(w, float(prof.v(xx)), Lam))) for xx in xs)
                h = min(hmax, gamma * N / max(km, 1e-3))
            nxt = min(x + h, tgt)
            if tgt - nxt < 0.3 * h:  # avoid a sliver element
                nxt = tgt if tgt - x < 1.3 * h else 0.5 * (x + tgt)
            pts.append(nxt)
            x = nxt
    return np.array(pts)


def solve(prof, w, Lam, L=None, N=28, gamma=0.55, hmax=0.25, check_flux=True):
    """Solve the 3 scattering problems; return dict with S (3x3, rows out=(u,vout,P),
    cols in=(vin,red,or)), eta, residuals, and the mode wavenumbers."""
    if L is None:
        L = prof.default_L()
    vL, vR = float(prof.v(-L)), float(prof.v(L))
    mL = classify(w, vL, Lam, 'L')
    if mL is None:
        return None
    mR = classify(w, vR, Lam, 'R')
    s = Lam  # derivative scaling: Y = (f, f'/s, f''/s^2, f'''/s^3)
    edges = build_mesh(prof, w, Lam, L, N, gamma, hmax)
    M = len(edges) - 1
    xr, Dm = cheb(N)
    n_el = 4 * (N + 1)
    nunk = M * n_el
    rows, cols, vals = [], [], []
    req = 0

    def idx(e, c, j):  # element e, component c, node j
        return e * n_el + c * (N + 1) + j

    xs_all = []
    for e in range(M):
        xa, xb = edges[e], edges[e + 1]
        xe = 0.5 * (xa + xb) + 0.5 * (xb - xa) * xr  # node 0 = xb, node N = xa
        xs_all.append(xe)
        De = Dm * (2.0 / (xb - xa))
        v = prof.v(xe)
        dv = prof.dv(xe)
        # coefficients of Y3' = sum_c a_c Y_c
        a0 = (Lam**2 / s**3) * (w * w + 1j * w * dv)
        a1 = (Lam**2 / s**2) * (2j * w * v - 2 * v * dv)
        a2 = (Lam**2 / s) * (1 - v * v)
        for j in range(1, N + 1):  # drop node 0 (= xb)
            for c in range(4):
                r = req
                req += 1
                # D Y_c at node j
                for jj in range(N + 1):
                    rows.append(r); cols.append(idx(e, c, jj)); vals.append(De[j, jj])
                if c < 3:
                    rows.append(r); cols.append(idx(e, c + 1, j)); vals.append(-s)
                else:
                    for cc, aa in ((0, a0[j]), (1, a1[j]), (2, a2[j])):
                        rows.append(r); cols.append(idx(e, cc, j)); vals.append(-aa)
        if e < M - 1:  # continuity: Y_e(xb) = Y_{e+1}(xa)
            for c in range(4):
                r = req
                req += 1
                rows += [r, r]; cols += [idx(e, c, 0), idx(e + 1, c, N)]; vals += [1.0, -1.0]

    def basis(kdict, labels):
        E = np.array([[(1j * kdict[l] / s) ** p for l in labels] for p in range(4)], complex)
        return E

    labR = ['u', 'vin', 'grow', 'dec']
    labL = ['red', 'or', 'vout', 'P']
    ER, EL = basis(mR, labR), basis(mL, labL)
    ERi, ELi = np.linalg.inv(ER), np.linalg.inv(EL)
    # BC rows: at +L (element M-1, node 0): coefficients of 'vin' and 'grow';
    #          at -L (element 0, node N): coefficients of 'red' and 'or'.
    bc_rows = {}
    for lab, Ei, e, j, li in (('vin', ERi, M - 1, 0, 1), ('grow', ERi, M - 1, 0, 2),
                              ('red', ELi, 0, N, 0), ('or', ELi, 0, N, 1)):
        r = req
        req += 1
        bc_rows[lab] = r
        for c in range(4):
            rows.append(r); cols.append(idx(e, c, j)); vals.append(Ei[li, c])
    assert req == nunk, (req, nunk)
    A = sp.csc_matrix((np.array(vals, complex), (rows, cols)), shape=(nunk, nunk))
    lu = spla.splu(A)

    def props(kd, l, v):
        Oc, vg = mode_props(kd[l].real if l not in ('grow', 'dec') else kd[l], w, v, Lam)
        return Oc, vg

    norm = {}
    for l in ('u', 'vin'):
        Oc, vg = props(mR, l, vR); norm[l] = (np.sqrt(abs(Oc * vg)), np.sign(Oc), np.sign(vg))
    for l in labL:
        Oc, vg = props(mL, l, vL); norm[l] = (np.sqrt(abs(Oc * vg)), np.sign(Oc), np.sign(vg))

    ins = ['vin', 'red', 'or']
    outs = ['u', 'vout', 'P']
    S = np.zeros((3, 3), complex)
    flux_err = 0.0
    sols = []
    for ci, lin in enumerate(ins):
        b = np.zeros(nunk, complex)
        b[bc_rows[lin]] = 1.0 / norm[lin][0]  # unit flux-normalised amplitude
        y = lu.solve(b)
        sols.append(y)
        YR = np.array([y[idx(M - 1, c, 0)] for c in range(4)])
        YL = np.array([y[idx(0, c, N)] for c in range(4)])
        cR = ERi @ YR
        cL = ELi @ YL
        amp = {'u': cR[0], 'vin': cR[1], 'grow': cR[2], 'dec': cR[3],
               'red': cL[0], 'or': cL[1], 'vout': cL[2], 'P': cL[3]}
        for ri, lo in enumerate(outs):
            S[ri, ci] = amp[lo] * norm[lo][0]
        if check_flux:
            # (i/2) J evaluated at every node of every element (uses Y and its derivatives)
            F = []
            for e in range(M):
                Ye = np.array([[y[idx(e, c, j)] for j in range(N + 1)] for c in range(4)])
                f, f1, f2, f3 = Ye[0], Ye[1] * s, Ye[2] * s**2, Ye[3] * s**3
                xe = xs_all[e]
                v = prof.v(xe)
                Df = -1j * w * f + v * f1
                J = (v * (np.conj(f) * Df - f * np.conj(Df)) - (np.conj(f) * f1 - f * np.conj(f1))
                     + (np.conj(f) * f3 - np.conj(f1) * f2 + np.conj(f2) * f1 - np.conj(f3) * f) / Lam**2)
                F.append((0.5j * J).real)
            F = np.concatenate(F)
            flux_err = max(flux_err, (F.max() - F.min()) / max(1.0, np.abs(F).max()))
    eta = np.diag([1.0, 1.0, -1.0])
    unit_err = np.abs(S.conj().T @ eta @ S - eta).max()
    return dict(S=S, eta=eta, unit_err=unit_err, flux_err=flux_err, M=M, N=N, L=L,
                nunk=nunk, kL=mL, kR=mR, vL=vL, vR=vR)


def temps(res, w, kap=1.0):
    """Temperatures from the S-matrix.  Rows out=(u, vout, P); cols in=(vin, red, or).
    T_col : column of in-mode 'red' (DPLS eq. (125)-(127) picture): |S_u,red|^2 / |S_P,red|^2
    T_row : row of out-mode 'u':   |S_u,red|^2 / |S_u,or|^2  (n_u = |S_u,or|^2)
    T_n   : from n_u = 1/(exp(w/T)-1)."""
    S = res['S']
    a2_col, b2_col, A2_col = abs(S[0, 1])**2, abs(S[2, 1])**2, abs(S[1, 1])**2
    a2_row, b2_row, A2_row = abs(S[0, 1])**2, abs(S[0, 2])**2, abs(S[0, 0])**2
    TH = kap / (2 * np.pi)
    return dict(
        alpha2=a2_row, beta2_row=b2_row, A2_row=A2_row, beta2_col=b2_col, A2_col=A2_col,
        T_row=w / np.log(a2_row / b2_row) / TH,
        T_col=w / np.log(a2_col / b2_col) / TH,
        T_n=w / np.log1p(1 / b2_row) / TH,
    )
