"""b0: exact-rational local expansion of the zero-frequency temperature (no sympy needed).

Derivation (see the article, Section 7):
  At w -> 0 the outgoing-u mode gives the exact (all-orders-in-Lambda) formula
      T(0) = |W_c| / (2 Lam^2 * N * M),
      N = int psi dx,  M = int (-v) psi dx  (oscillatory tails Abel-regularised),
      psi'' = Lam^2 (1 - v^2) psi,  psi decaying for x -> +inf,
      W_c = Wronskian of the positive-frequency (e^{+ikx}) half of psi with its conjugate.
  Scaling x = eps*y with 2 Lam^2 eps^3 = 1 (kappa = 1), u = 1 + v = x + a2 x^2 + a3 x^3 + a4 x^4 + ...,
      psi_yy = V(y) psi,  V = (2u - u^2)/(2 eps) = y + eps c1 y^2 + eps^2 c2 y^3 + eps^3 c3 y^4 + ...
  Perturbation:  psi = P(y) Ai(y) + Q(y) Ai'(y), P, Q polynomials (series in eps).
      (d^2 - y)(P Ai + Q Ai') = (P'' + 2y Q' + Q) Ai + (2P' + Q'') Ai'.
  The same P, Q with Bi give the second solution, so
      pi * W(psi_Ai, psi_Bi) = P^2 + P Q' - Q P' - y Q^2   (must be constant),
      |W_c| = W(psi_Ai, psi_Bi)/2,  T/T_H = (P^2+PQ'-QP'-yQ^2) / (I_N I_M),
      I_N = int psi dy,  I_M = int (1-u) psi dy,  with Airy moments m_n = int y^n Ai = (3j)!/(3^j j!) if n=3j else 0.
Everything is exact rational arithmetic in (a2, a3, a4)."""
from fractions import Fraction as Fr
from math import factorial
import json, os, sys

ORDER = int(os.environ.get('B0_ORDER', '3'))  # eps order kept (Lam^-2 = 2 eps^3)
NA = ORDER  # parameters a2..a_{ORDER+1}  (a_n = v^(n)(0)/n!, kappa = v'(0) = 1)


# ---- multivariate polynomials in (a2,a3,a4): dict exponent-tuple -> Fraction
def cadd(p, q, s=1):
    r = dict(p)
    for k, c in q.items():
        r[k] = r.get(k, 0) + s * c
        if r[k] == 0:
            del r[k]
    return r


def cmul(p, q):
    r = {}
    for k1, c1 in p.items():
        for k2, c2 in q.items():
            k = tuple(a + b for a, b in zip(k1, k2))
            r[k] = r.get(k, 0) + c1 * c2
    return {k: c for k, c in r.items() if c != 0}


def cscale(p, s):
    return {k: c * s for k, c in p.items() if c * s != 0}


ONE = {(0,) * NA: Fr(1)}
A = [None, None] + [{tuple(1 if i == j else 0 for i in range(NA)): Fr(1)} for j in range(NA)]  # A[n] = a_n


# ---- polynomials in y with coefficients in Q[a]: dict power -> coeff-poly
def yadd(p, q, s=1):
    r = dict(p)
    for n, c in q.items():
        r[n] = cadd(r.get(n, {}), c, s)
        if not r[n]:
            del r[n]
    return r


def ymul(p, q):
    r = {}
    for n1, c1 in p.items():
        for n2, c2 in q.items():
            r[n1 + n2] = cadd(r.get(n1 + n2, {}), cmul(c1, c2))
    return {n: c for n, c in r.items() if c}


def yder(p):
    return {n - 1: cscale(c, n) for n, c in p.items() if n > 0}


def deg(p):
    return max(p) if p else -1


# ---- series in eps of y-polynomials: list index = eps power
def solve_PQ(RA, RB):
    """Polynomial P, Q with P'' + 2yQ' + Q = RA, 2P' + Q'' = RB, p0 = 0."""
    D = max(deg(RA), deg(RB)) + 4
    p = {n: {} for n in range(D + 3)}
    q = {n: {} for n in range(D + 3)}
    for n in range(D, -1, -1):
        # B at y^n: 2(n+1) p_{n+1} + (n+2)(n+1) q_{n+2} = RB_n
        p[n + 1] = cscale(cadd(RB.get(n, {}), cscale(q[n + 2], (n + 2) * (n + 1)), -1), Fr(1, 2 * (n + 1)))
        # A at y^n: (n+2)(n+1) p_{n+2} + (2n+1) q_n = RA_n
        q[n] = cscale(cadd(RA.get(n, {}), cscale(p[n + 2], (n + 2) * (n + 1)), -1), Fr(1, 2 * n + 1))
    P = {n: c for n, c in p.items() if c}
    Q = {n: c for n, c in q.items() if c}
    # verify
    lhsA = yadd(yadd(yder(yder(P)), ymul({1: cscale(ONE, 2)}, yder(Q))), Q)
    lhsB = yadd(cscalep(yder(P), 2), yder(yder(Q)))
    assert not yadd(lhsA, RA, -1) and not yadd(lhsB, RB, -1)
    return P, Q


def cscalep(p, s):
    return {n: cscale(c, s) for n, c in p.items() if cscale(c, s)}


def ser_mul(s1, s2, order=ORDER, mul=ymul, add=yadd):
    r = [dict() for _ in range(order + 1)]
    for i, a in enumerate(s1):
        for j, b in enumerate(s2):
            if i + j <= order:
                r[i + j] = add(r[i + j], mul(a, b))
    return r


def moment(n):
    if n < 0 or n % 3:
        return Fr(0)
    j = n // 3
    return Fr(factorial(3 * j), 3 ** j * factorial(j))


def int_PQ(P, Q):
    """int (P Ai + Q Ai') dy = sum p_n m_n - sum n q_n m_{n-1}."""
    r = {}
    for n, c in P.items():
        r = cadd(r, cscale(c, moment(n)))
    for n, c in Q.items():
        r = cadd(r, cscale(c, -n * moment(n - 1)))
    return r


def cser_mul(s1, s2, order=ORDER):
    r = [dict() for _ in range(order + 1)]
    for i, a in enumerate(s1):
        for j, b in enumerate(s2):
            if i + j <= order:
                r[i + j] = cadd(r[i + j], cmul(a, b))
    return r


def cser_inv(s, order=ORDER):
    assert s[0] == ONE
    r = [ONE] + [dict() for _ in range(order)]
    for n in range(1, order + 1):
        acc = {}
        for k in range(1, n + 1):
            acc = cadd(acc, cmul(s[k], r[n - k]))
        r[n] = cscale(acc, -1)
    return r


def main():
    # u(x) series: u = x + a2 x^2 + a3 x^3 + a4 x^4 ; in y with x = eps y: u_n (eps power n) = a_n y^n
    useries = [dict(), {1: ONE}] + [{n: A[n]} for n in range(2, ORDER + 2)]  # eps^n coefficient (y-poly)
    # V = (2u - u^2)/(2 eps): eps^m coefficient = [2 u_{m+1} - (u^2)_{m+1}]/2
    u2 = ser_mul(useries, useries, order=ORDER + 1)
    V = []
    for m in range(ORDER + 1):
        term = yadd(cscalep(useries[m + 1], 2), u2[m + 1], -1)
        V.append(cscalep(term, Fr(1, 2)))
    assert V[0] == {1: ONE}
    Ps, Qs = [{0: ONE}], [{}]
    for n in range(1, ORDER + 1):
        RA, RB = {}, {}
        for m in range(1, n + 1):
            RA = yadd(RA, ymul(V[m], Ps[n - m]))
            RB = yadd(RB, ymul(V[m], Qs[n - m]))
        P, Q = solve_PQ(RA, RB)
        Ps.append(P); Qs.append(Q)
    # Wronskian combination D = P^2 + P Q' - Q P' - y Q^2
    PP = ser_mul(Ps, Ps)
    PQd = ser_mul(Ps, [yder(q) for q in Qs])
    QPd = ser_mul(Qs, [yder(p) for p in Ps])
    QQ = ser_mul(Qs, Qs)
    Dser = []
    for n in range(ORDER + 1):
        d = yadd(yadd(PP[n], PQd[n]), QPd[n], -1)
        d = yadd(d, ymul({1: ONE}, QQ[n]), -1)
        Dser.append(d)
    for n, d in enumerate(Dser):
        assert set(d) <= {0}, f"Wronskian not constant at order {n}: {d}"
    Dc = [d.get(0, {}) for d in Dser]
    # integrals
    IN = [int_PQ(Ps[n], Qs[n]) for n in range(ORDER + 1)]
    upsi_P = ser_mul(useries[:ORDER + 1], Ps)
    upsi_Q = ser_mul(useries[:ORDER + 1], Qs)
    IU = [int_PQ(upsi_P[n], upsi_Q[n]) for n in range(ORDER + 1)]  # int u psi dy
    IM = [cadd(IN[n], IU[n], -1) for n in range(ORDER + 1)]
    # T/T_H = D / (IN * IM) ; F/T_H = D/IN^2 ; delta = -IU/IN
    Tser = cser_mul(Dc, cser_inv(cser_mul(IN, IM)))
    Fser = cser_mul(Dc, cser_inv(cser_mul(IN, IN)))
    dser = cser_mul([cscale(c, -1) for c in IU], cser_inv(IN))

    def fmt(c):
        if not c:
            return '0'
        names = [f'a{n}' for n in range(2, NA + 2)]
        out = []
        for k, v in sorted(c.items()):
            mon = '*'.join(f"{nm}^{e}" if e > 1 else nm for nm, e in zip(names, k) if e) or '1'
            out.append(f"({v})*{mon}")
        return ' + '.join(out)

    res = {}
    for lab, s in (('T/T_H', Tser), ('F/T_H', Fser), ('delta', dser), ('pi*W', Dc), ('I_N', IN), ('I_M', IM)):
        print(lab)
        for n, c in enumerate(s):
            print(f"   eps^{n}: {fmt(c)}")
        res[lab] = [fmt(c) for c in s]
    for n in range(ORDER + 1):
        print(f"P{n} = {{{', '.join(f'y^{k}: {fmt(c)}' for k, c in sorted(Ps[n].items()))}}}")
        print(f"Q{n} = {{{', '.join(f'y^{k}: {fmt(c)}' for k, c in sorted(Qs[n].items()))}}}")
    # b0: T/T_H - 1 = beta3 eps^3 = beta3 Lam^-2 / 2  => b0 = beta3/2 ; in v-derivatives a2 = v''/2, a3 = v'''/6, a4 = v''''/24
    def tov(c):  # a_n = v^(n)/n!  ->  monomials in v^(n)
        out = {}
        for k, cc in c.items():
            f = Fr(1)
            for i, e in enumerate(k):
                f /= factorial(i + 2) ** e
            out[k] = cc * f
        return out
    b0 = cscale(Tser[3], Fr(1, 2))
    conv = tov(b0)
    print('b0 in v-derivative monomials (exponents of v2, v3, v4, ...):', {str(k): str(v) for k, v in conv.items()})
    z = (0,) * NA
    def e(*ix):
        return tuple(ix) + (0,) * (NA - len(ix))
    target = {e(0, 0, 1): Fr(1, 12), e(1, 1, 0): Fr(-4, 12), e(3, 0, 0): Fr(3, 12)}
    if ORDER >= 6:
        b1 = cscale(Tser[6], Fr(1, 4))  # Lam^-4 = 4 eps^6
        print('b1 (coefficient of Lam^-4 in T(0)/T_H - 1) in v-derivative monomials:', {str(k): str(v) for k, v in tov(b1).items()})
        print('nonzero orders of T/T_H - 1:', [n for n in range(1, ORDER + 1) if Tser[n]])
        res['b1_vmonomials'] = {str(k): str(v) for k, v in tov(b1).items()}
        res['delta_Lam^-4_vmonomials'] = {str(k): str(v) for k, v in tov(cscale(dser[6], Fr(1, 4))).items()}
        res['F_Lam^-4_vmonomials'] = {str(k): str(v) for k, v in tov(cscale(Fser[6], Fr(1, 4))).items()}
        res['nonzero_orders_T'] = [n for n in range(1, ORDER + 1) if Tser[n]]
    print('matches (v4 - 4 v2 v3 + 3 v2^3)/12 :', conv == target)
    dlt = tov(cscale(dser[3], Fr(1, 2)))
    Fb = tov(cscale(Fser[3], Fr(1, 2)))
    res['b0_vmonomials'] = {str(k): str(v) for k, v in conv.items()}
    res['delta_Lam^-2_vmonomials'] = {str(k): str(v) for k, v in dlt.items()}
    res['F_Lam^-2_vmonomials'] = {str(k): str(v) for k, v in Fb.items()}
    res['matches_conjecture'] = conv == target
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'local_expansion_out_order{ORDER}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
