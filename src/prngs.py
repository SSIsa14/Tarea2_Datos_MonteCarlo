"""Cinco generadores de numeros pseudoaleatorios implementados desde cero.

Todos comparten la misma interfaz:
    g = LCG(141903)
    g.next_int()   -> entero crudo
    g.random()     -> flotante en [0,1)
    g.sample(n)    -> lista de n flotantes
"""


class PRNG:
    """Base comun: las subclases definen el modulo m y next_int()."""

    name = "PRNG"
    m = 1

    def next_int(self):
        raise NotImplementedError

    def random(self):
        return self.next_int() / float(self.m)

    def sample(self, n):
        return [self.random() for _ in range(n)]

    def __repr__(self):
        return f"<{self.name} semilla={getattr(self, 'seed', None)}>"


# --- Los cinco generadores ---


class LCG(PRNG):
    """X_{n+1} = (a*X_n + c) mod m. Parametros de Numerical Recipes."""

    name = "LCG"

    def __init__(self, seed, a=1664525, c=1013904223, m=2 ** 32):
        self.seed = seed
        self.a, self.c, self.m = a, c, m
        self.state = seed % m

    def next_int(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state


class MiddleSquare(PRNG):
    """Cuadrados medios de von Neumann: los k digitos centrales de X^2."""

    name = "MiddleSquare"

    def __init__(self, seed, n_digits=8):
        self.seed = seed
        self.k = n_digits
        self.m = 10 ** n_digits
        self.state = seed % self.m

    def next_int(self):
        s = str(self.state * self.state).zfill(2 * self.k)
        inicio = (len(s) - self.k) // 2
        self.state = int(s[inicio:inicio + self.k])
        return self.state


class MersenneTwister(PRNG):
    """MT19937. Periodo 2^19937-1, estado de 624 palabras de 32 bits."""

    name = "MersenneTwister"

    N = 624
    M = 397
    MATRIZ_A = 0x9908B0DF
    MASCARA_ALTA = 0x80000000
    MASCARA_BAJA = 0x7FFFFFFF

    def __init__(self, seed):
        self.seed = seed
        self.m = 2 ** 32
        self.mt = [0] * self.N
        self.idx = self.N
        self._sembrar(seed & 0xFFFFFFFF)

    def _sembrar(self, s):
        self.mt[0] = s
        for i in range(1, self.N):
            previo = self.mt[i - 1]
            self.mt[i] = (1812433253 * (previo ^ (previo >> 30)) + i) & 0xFFFFFFFF
        self.idx = self.N

    def _torcer(self):
        for i in range(self.N):
            y = ((self.mt[i] & self.MASCARA_ALTA) |
                 (self.mt[(i + 1) % self.N] & self.MASCARA_BAJA))
            nuevo = self.mt[(i + self.M) % self.N] ^ (y >> 1)
            if y & 1:
                nuevo ^= self.MATRIZ_A
            self.mt[i] = nuevo
        self.idx = 0

    def next_int(self):
        if self.idx >= self.N:
            self._torcer()
        y = self.mt[self.idx]
        self.idx += 1
        # templado: mejora la equidistribucion de los bits
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & 0xFFFFFFFF

    # --- sembrado y salida al estilo de CPython, para poder comparar ---

    def _sembrar_por_arreglo(self, clave):
        self._sembrar(19650218)
        i, j = 1, 0
        for _ in range(max(self.N, len(clave))):
            previo = self.mt[i - 1]
            self.mt[i] = (((self.mt[i] ^ ((previo ^ (previo >> 30)) * 1664525))
                           + clave[j] + j) & 0xFFFFFFFF)
            i, j = i + 1, j + 1
            if i >= self.N:
                self.mt[0] = self.mt[self.N - 1]
                i = 1
            if j >= len(clave):
                j = 0
        for _ in range(self.N - 1):
            previo = self.mt[i - 1]
            self.mt[i] = (((self.mt[i] ^ ((previo ^ (previo >> 30)) * 1566083941))
                           - i) & 0xFFFFFFFF)
            i += 1
            if i >= self.N:
                self.mt[0] = self.mt[self.N - 1]
                i = 1
        self.mt[0] = 0x80000000
        self.idx = self.N

    @classmethod
    def como_cpython(cls, seed):
        """Instancia sembrada igual que random.seed(seed)."""
        g = cls(0)
        n, clave = abs(seed), []
        while n:
            clave.append(n & 0xFFFFFFFF)
            n >>= 32
        g.seed = seed
        g._sembrar_por_arreglo(clave or [0])
        return g

    def random53(self):
        """Flotante de 53 bits, igual que random.random()."""
        a = self.next_int() >> 5
        b = self.next_int() >> 6
        return (a * 67108864.0 + b) / 9007199254740992.0


class BlumBlumShub(PRNG):
    """x_{n+1} = x_n^2 mod N, con N = p*q y p, q primos seguros = 3 mod 4."""

    name = "BlumBlumShub"

    # p = 2*2147483693 + 1 y q = 2*4294967681 + 1: ambos primos seguros.
    P = 4294967387
    Q = 8589935363

    def __init__(self, seed, p=None, q=None, bits_salida=32):
        self.seed = seed
        self.p = p or self.P
        self.q = q or self.Q
        self.N = self.p * self.q
        self.bits_salida = bits_salida
        self.m = 2 ** bits_salida
        # bits utiles por paso: j = floor(log2(log2 N))
        self.j = self.N.bit_length().bit_length() - 1
        x = seed % self.N
        if x in (0, 1) or mcd(x, self.N) != 1:
            raise ValueError("la semilla debe ser coprima con N y distinta de 0 y 1")
        self.state = (x * x) % self.N

    def next_int(self):
        valor, bits = 0, 0
        while bits < self.bits_salida:
            self.state = (self.state * self.state) % self.N
            toma = min(self.j, self.bits_salida - bits)
            valor = (valor << toma) | (self.state & ((1 << toma) - 1))
            bits += toma
        return valor


class RANDU(PRNG):
    """X_{n+1} = 65539*X_n mod 2^31. El LCG multiplicativo de IBM."""

    name = "RANDU"

    def __init__(self, seed):
        if seed % 2 == 0:
            raise ValueError("RANDU necesita semilla impar")
        self.seed = seed
        self.a = 65539
        self.m = 2 ** 31
        self.state = seed % self.m

    def next_int(self):
        self.state = (self.a * self.state) % self.m
        return self.state


# --- Utilidades ---


def mcd(a, b):
    while b:
        a, b = b, a % b
    return a


def detectar_periodo(gen, max_iter=1_000_000):
    """Devuelve (indice de la primera repeticion, largo del ciclo)."""
    vistos = {}
    for i in range(max_iter):
        if gen.state in vistos:
            return vistos[gen.state], i - vistos[gen.state]
        vistos[gen.state] = i
        gen.next_int()
    return None, None


if __name__ == "__main__":
    for g in [LCG(141903), MiddleSquare(141903), MersenneTwister(141903),
              BlumBlumShub(14773), RANDU(141903)]:
        print(f"{g.name:16s} {[round(u, 5) for u in g.sample(5)]}")
