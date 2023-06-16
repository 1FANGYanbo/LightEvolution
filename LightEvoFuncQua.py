import numpy as np
from scipy import linalg
from scipy.fft import fft, ifft

Alpha = 5
U = 2
Gamma = 5
zTot = 100
Q = 10000
dz = zTot / Q
nStart = -500
L = 2 * 500  # lattice length
nRange = range(nStart, nStart + L)
N = int(L / 2)
C0 = 1
kc = 1 / 100000
d = 0.01


def C1(x):
    return C0 - kc * x ** 2


def C2(x):
    return C0 + kc * x ** 2


def beta0(x):
    return -2 * C0


def beta1(x):
    return -2 * C0


def sech(x):
    return 2 / (np.exp(x) + np.exp(-x))


def EIni(n):
    return d * np.sqrt(2 * Alpha * C0 / Gamma) * np.exp(1j * U / 2 * n * d) * sech(np.sqrt(Alpha) * n * d)


Ei = []
for i in nRange:
    Ei.append(EIni(i))


def avgPos(EVec):
    """

    :param EVec:
    :return: <x>
    """
    rst = 0
    for j in range(0, len(nRange)):
        rst += nRange[j] * np.abs(EVec[j]) ** 2

    rst /= (np.linalg.norm(EVec, 2)) ** 2
    return rst


def avgPos2(EVec):
    """

    :param EVec:
    :return: <x^{2}>
    """
    rst = 0
    for j in range(0, len(nRange)):
        rst += (nRange[j]) ** 2 * np.abs(EVec[j]) ** 2
    rst /= (np.linalg.norm(EVec, 2)) ** 2
    return rst


def sd(psiVec):
    x = avgPos(psiVec)
    x2 = avgPos2(psiVec)
    s = np.sqrt(np.abs(x2 - x ** 2))
    return s


def norm2(EVec):
    """

    :param EVec:
    :return: 2 norm pf psi
    """
    return np.linalg.norm(EVec, 2)


def height(EVec):
    """

    :param EVec:
    :return: maximum height of psi
    """
    # absTmp=[np.abs(elem) for elem in psiVec]
    return np.max(np.abs(EVec))


def FNLStepA(Eq, dx):
    A0 = []
    for j in range(N):
        A0.append(np.exp(1j * Gamma * abs(Eq[2 * j]) ** 2 * dx) * Eq[2 * j])

    return A0


def FNLStepB(Eq, dx):
    B0 = []
    for j in range(N):
        B0.append(np.exp(1j * Gamma * abs(Eq[2 * j + 1]) ** 2 * dx) * Eq[2 * j + 1])

    return B0


def LNLStep(Eq1, Eq2, dx):
    A1 = []
    B1 = []
    for j in range(N):
        A1.append(np.exp(1j * Gamma * abs(Eq1[j]) ** 2 * dx) * Eq1[j])
        B1.append(np.exp(1j * Gamma * abs(Eq2[j]) ** 2 * dx) * Eq2[j])

    C = []

    for j in range(N):
        C.append(A1[j])
        C.append(B1[j])

    return C


def h(k, x):
    h0 = np.array([[-beta0(x), -C1(x) * np.exp(1j * k) - C2(x)],
                   [-C1(x) * np.exp(-1j * k) - C2(x), -beta1(x)]])

    return h0


K = np.array(range(N)) * 2 * np.pi / N


def LIStepw(A, B, t, deltat):
    x = fft(A)
    y = fft(B)
    u = []
    v = []
    for j in range(N):
        a = np.array([x[j], y[j]])
        b = np.matmul(linalg.expm(-1j * h(K[j], t) * deltat), a)
        u.append(b[0])
        v.append(b[1])

    w = ifft(u)
    return w


def LIStepz(A, B, t, deltat):
    x = fft(A)
    y = fft(B)
    u = []
    v = []
    for j in range(N):
        a = np.array([x[j], y[j]])
        b = np.matmul(linalg.expm(-1j * h(K[j], t) * deltat), a)
        u.append(b[0])
        v.append(b[1])

    z = ifft(v)
    return z


def OneStep(Eq, tCurr, dt):
    EA = FNLStepA(Eq, dt / 2)
    EB = FNLStepB(Eq, dt / 2)
    EA1 = LIStepw(EA, EB, tCurr, dt)
    EB1 = LIStepz(EA, EB, tCurr, dt)
    EqPlus = LNLStep(EA1, EB1, dt / 2)
    return EqPlus
