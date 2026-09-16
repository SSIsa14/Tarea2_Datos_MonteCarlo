"""Pruebas de uniformidad e independencia, sin usar scipy.

Cada prueba devuelve un diccionario con el estadistico y el p-valor.
Las funciones de distribucion (gamma incompleta y Kolmogorov) estan
escritas a mano siguiendo Numerical Recipes.
"""

import math

_ITER = 300
_EPS = 3.0e-14


# --- Funciones de distribucion, a mano porque no hay scipy ---


def _serie_gamma(a, x):
    """Serie para la gamma incompleta inferior regularizada."""
    if x <= 0:
        return 0.0
    suma = termino = 1.0 / a
    ap = a
    for _ in range(_ITER):
        ap += 1
        termino *= x / ap
        suma += termino
        if abs(termino) < abs(suma) * _EPS:
            break
    return suma * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _fraccion_gamma(a, x):
    """Fraccion continua para la gamma incompleta superior regularizada."""
    minimo = 1e-300
    b = x + 1.0 - a
    c = 1.0 / minimo
    d = 1.0 / b
    h = d
    for i in range(1, _ITER):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < minimo:
            d = minimo
        c = b + an / c
        if abs(c) < minimo:
            c = minimo
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_cola(x, gl):
    """P(X > x) para una chi-cuadrado con gl grados de libertad."""
    a, y = gl / 2.0, x / 2.0
    return 1.0 - _serie_gamma(a, y) if y < a + 1 else _fraccion_gamma(a, y)


def normal_cola_doble(z):
    """p-valor a dos colas de la normal estandar."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def kolmogorov_cola(lam):
    """Q(lam) = 2 * sum (-1)^{k-1} exp(-2 k^2 lam^2)."""
    if lam < 0.2:
        return 1.0
    total, signo = 0.0, 2.0
    for k in range(1, 100):
        termino = signo * math.exp(-2.0 * k * k * lam * lam)
        total += termino
        if abs(termino) < 1e-12:
            break
        signo = -signo
    return min(1.0, max(0.0, total))


# --- Las cuatro pruebas: dos de uniformidad y dos de independencia ---


def chi_cuadrado(u, k=10):
    """Bondad de ajuste a Unif(0,1) con k celdas de igual ancho."""
    n = len(u)
    conteos = [0] * k
    for x in u:
        conteos[min(int(x * k), k - 1)] += 1
    esperado = n / k
    stat = sum((o - esperado) ** 2 / esperado for o in conteos)
    return {"nombre": "chi2", "stat": stat, "gl": k - 1,
            "p": chi2_cola(stat, k - 1), "conteos": conteos, "esperado": esperado}


def kolmogorov_smirnov(u):
    """Maxima distancia entre la acumulada empirica y la de Unif(0,1)."""
    xs = sorted(u)
    n = len(xs)
    d_mas = max((i + 1) / n - x for i, x in enumerate(xs))
    d_menos = max(x - i / n for i, x in enumerate(xs))
    d = max(d_mas, d_menos)
    raiz = math.sqrt(n)
    return {"nombre": "KS", "stat": d,
            "p": kolmogorov_cola((raiz + 0.12 + 0.11 / raiz) * d)}


def autocorrelacion_lag1(u):
    """Correlacion entre u_i y u_{i+1}. Bajo H0, sqrt(n)*r1 ~ N(0,1)."""
    n = len(u)
    media = sum(u) / n
    num = sum((u[i] - media) * (u[i + 1] - media) for i in range(n - 1))
    den = sum((x - media) ** 2 for x in u)
    r1 = num / den if den > 0 else 0.0
    z = math.sqrt(n) * r1
    return {"nombre": "lag1", "stat": r1, "z": z, "p": normal_cola_doble(z)}


def rachas(u):
    """Numero de rachas arriba/abajo de la mediana."""
    xs = sorted(u)
    n = len(u)
    mediana = xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
    signos = [1 if x > mediana else 0 for x in u if x != mediana]
    n1 = sum(signos)
    n2 = len(signos) - n1
    if n1 == 0 or n2 == 0:
        return {"nombre": "rachas", "stat": 0, "esperado": 0.0,
                "z": float("nan"), "p": 0.0}
    r = 1 + sum(1 for i in range(len(signos) - 1) if signos[i] != signos[i + 1])
    total = n1 + n2
    esperado = 2.0 * n1 * n2 / total + 1.0
    var = (2.0 * n1 * n2 * (2.0 * n1 * n2 - total)) / (total ** 2 * (total - 1.0))
    z = (r - esperado) / math.sqrt(var)
    return {"nombre": "rachas", "stat": r, "esperado": esperado, "z": z,
            "p": normal_cola_doble(z)}


def todas(u, k=10):
    return {"chi2": chi_cuadrado(u, k), "ks": kolmogorov_smirnov(u),
            "lag1": autocorrelacion_lag1(u), "rachas": rachas(u)}


if __name__ == "__main__":
    # Valores de tabla: los tres primeros deberian dar ~0.05
    print("chi2_cola(16.919, 9) =", round(chi2_cola(16.919, 9), 4))
    print("normal_cola_doble(1.96) =", round(normal_cola_doble(1.96), 4))
    print("kolmogorov_cola(1.358) =", round(kolmogorov_cola(1.358), 4))
