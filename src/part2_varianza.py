"""Parte 2.4: reduccion de varianza con variables antiteticas y de control.

Aplica las dos tecnicas a las integrales de la seccion 2.2 y mide cuanto
baja la varianza con el mismo numero de evaluaciones.

Uso:  python3 src/part2_varianza.py
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prngs import MersenneTwister
from montecarlo import media_ic, escribir_tex, macro, miles, cientifica, FIGS

SEMILLA = 77140
N = 200_000        # evaluaciones de f por estimador
N_PILOTO = 20_000  # muestras para estimar el coeficiente de control
K = 200            # repeticiones para el histograma
N_REP = 10_000     # evaluaciones por repeticion

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25})


def seno(x):
    return math.sin(math.pi * x)


def normal(x):
    return math.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)


# (clave, f, a, b, exacto, control h(x), E[h(U)], etiqueta)
CASOS = [
    ("seno", seno, 0.0, 1.0, 2.0 / math.pi,
     lambda x: x * (1 - x), 1.0 / 6.0, "(a) $\\sin(\\pi x)$ en $[0,1]$"),
    ("normal", normal, 0.0, 2.0, 0.5 * math.erf(2.0 / math.sqrt(2.0)),
     lambda x: x * x, 4.0 / 3.0, "(b) densidad normal en $[0,2]$"),
]


# --- Las tres maneras de estimar la misma integral ---


def simple(f, a, b, n, gen):
    return [(b - a) * f(a + (b - a) * gen.random()) for _ in range(n)]


def antiteticas(f, a, b, n, gen):
    """Promedia f(U) con f(1-U). Usa n evaluaciones en n/2 pares."""
    valores = []
    for _ in range(n // 2):
        u = gen.random()
        x1 = a + (b - a) * u
        x2 = a + (b - a) * (1 - u)
        valores.append((b - a) * 0.5 * (f(x1) + f(x2)))
    return valores


def coef_control(f, a, b, h, media_h, n, gen):
    """c* = Cov(f, h) / Var(h), estimado con una muestra piloto."""
    fs, hs = [], []
    for _ in range(n):
        x = a + (b - a) * gen.random()
        fs.append((b - a) * f(x))
        hs.append(h(x))
    mf = sum(fs) / n
    mh = sum(hs) / n
    cov = sum((fs[i] - mf) * (hs[i] - mh) for i in range(n)) / (n - 1)
    var = sum((v - mh) ** 2 for v in hs) / (n - 1)
    corr = cov / math.sqrt(var * (sum((v - mf) ** 2 for v in fs) / (n - 1)))
    return cov / var, corr


def control(f, a, b, n, gen, h, media_h, c):
    return [(b - a) * f(a + (b - a) * u) - c * (h(a + (b - a) * u) - media_h)
            for u in (gen.random() for _ in range(n))]


def figura(muestras):
    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.6))
    colores = {"simple": "#4C72B0", "antiteticas": "#C44E52",
               "control": "#55A868"}
    for ax, (clave, _, _, _, exacto, _, _, etiqueta) in zip(ejes, CASOS):
        for metodo, color in colores.items():
            ax.hist(muestras[(clave, metodo)], bins=30, alpha=0.6, color=color,
                    label=metodo)
        ax.axvline(exacto, color="black", ls="--", lw=1.2, label="valor exacto")
        ax.set_title(etiqueta, fontsize=10)
        ax.set_xlabel("estimacion")
        ax.set_ylabel("frecuencia")
        ax.legend(fontsize=7)
    fig.suptitle(f"Dispersion de {K} estimaciones con {miles(N_REP).replace('{,}', ',')} "
                 f"evaluaciones cada una", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "reduccion_varianza.png"))
    plt.close(fig)


def muestras_equivalentes(factor):
    """Muestras simples que harian falta para igualar esta precision."""
    return cientifica(factor * N, 1)


def escribir(filas, correlaciones):
    cortas = {"seno": "(a)", "normal": "(b)"}
    L = [r"\begin{table}[htbp]\centering",
         r"\caption{Reduccion de varianza con $N=\NVar$ evaluaciones de $f$ "
         r"por estimador, sobre la integral (a) del seno y la (b) de la "
         r"densidad normal. El factor compara la varianza del estimador "
         r"contra Monte Carlo simple, y el speedup efectivo es el numero de "
         r"muestras simples que harian falta para igualar esa precision.}",
         r"\label{tab:varianza}", r"\footnotesize",
         r"\begin{tabular}{l|l|r|l|r|r|r}", r"\toprule",
         r"Integral & Tecnica & Estimacion & IC $95\%$ & Varianza & Factor & "
         r"Speedup \\", r"\midrule"]
    for f in filas:
        etiqueta = cortas[f["clave"]] if f["metodo"] == "simple" else ""
        L.append(f"{etiqueta} & {f['metodo']} & ${f['media']:.6f}$ & "
                 f"$[{f['lo']:.6f},\\ {f['hi']:.6f}]$ & "
                 f"{cientifica(f['var_est'])} & {f['factor']:.2f}x & "
                 f"{muestras_equivalentes(f['factor'])} \\\\")
        if f["metodo"] == "control":
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    escribir_tex("tabla_varianza.tex", L)

    def buscar(clave, metodo, campo):
        return next(f[campo] for f in filas
                    if f["clave"] == clave and f["metodo"] == metodo)

    L = [macro("NVar", miles(N)),
         macro("KRepeticiones", K),
         macro("NRep", miles(N_REP)),
         macro("SemillaVar", SEMILLA),
         macro("FactorAntiSeno", f"{buscar('seno', 'antiteticas', 'factor'):.2f}"),
         macro("FactorAntiNormal",
               f"{buscar('normal', 'antiteticas', 'factor'):.2f}"),
         macro("FactorControlSeno", f"{buscar('seno', 'control', 'factor'):.2f}"),
         macro("FactorControlNormal",
               f"{buscar('normal', 'control', 'factor'):.2f}"),
         macro("CorrSeno", f"{correlaciones['seno']:.4f}"),
         macro("CorrNormal", f"{correlaciones['normal']:.4f}"),
         macro("CoefSeno", f"{correlaciones['c_seno']:.4f}"),
         macro("CoefNormal", f"{correlaciones['c_normal']:.4f}")]
    escribir_tex("datos_varianza.tex", L)


def main():
    print("Parte 2.4: reduccion de varianza")
    filas, muestras, correlaciones = [], {}, {}

    for clave, f, a, b, exacto, h, media_h, etiqueta in CASOS:
        print(f"\n  {clave} (exacto = {exacto:.6f})")
        c, corr = coef_control(f, a, b, h, media_h, N_PILOTO,
                               MersenneTwister(SEMILLA))
        correlaciones[clave] = corr
        correlaciones[f"c_{clave}"] = c
        print(f"    control: c* = {c:.4f}, correlacion = {corr:.4f}")

        estimadores = {
            "simple": lambda g: simple(f, a, b, N, g),
            "antiteticas": lambda g: antiteticas(f, a, b, N, g),
            "control": lambda g: control(f, a, b, N, g, h, media_h, c),
        }
        base = None
        for metodo, hacer in estimadores.items():
            valores = hacer(MersenneTwister(SEMILLA + 7))
            r = media_ic(valores)
            var_est = r["var"] / len(valores)
            if base is None:
                base = var_est
            filas.append({"clave": clave, "etiqueta": etiqueta if metodo == "simple" else "",
                          "metodo": metodo, "media": r["media"], "lo": r["lo"],
                          "hi": r["hi"], "var_est": var_est,
                          "factor": base / var_est})
            print(f"    {metodo:12s} {r['media']:.6f} "
                  f"[{r['lo']:.6f}, {r['hi']:.6f}]  var={var_est:.2e}  "
                  f"factor={base/var_est:.2f}x")

        # repeticiones para ver la dispersion
        for metodo, hacer in estimadores.items():
            valores_rep = []
            for k in range(K):
                g = MersenneTwister(SEMILLA + 1000 * (k + 1))
                if metodo == "simple":
                    v = simple(f, a, b, N_REP, g)
                elif metodo == "antiteticas":
                    v = antiteticas(f, a, b, N_REP, g)
                else:
                    v = control(f, a, b, N_REP, g, h, media_h, c)
                valores_rep.append(sum(v) / len(v))
            muestras[(clave, metodo)] = valores_rep

    figura(muestras)
    escribir(filas, correlaciones)
    print("\nListo.")


if __name__ == "__main__":
    main()
