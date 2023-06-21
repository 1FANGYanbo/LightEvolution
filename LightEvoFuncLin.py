import numpy as np
from scipy import linalg
from scipy.fft import fft, ifft

Alpha = 5
U = 2
Gamma = 5
zTot = 100
Q = 100*100
dz = zTot / Q
nStart = -500
L = 2 * 500  # lattice length
nRange = range(nStart, nStart + L)
N = int(L / 2)
C0 = 1
kc = 0.005
d = 0.01


def C1(x):
    return C0 - kc * x


def C2(x):
    return C0 + kc * x


def beta0(x):
    return -2 * C0


def beta1(x):
    return -2 * C0


def sech(x):
    return 2 / (np.exp(x) + np.exp(-x))


def EIni(n):
    return d * np.sqrt(2 * Alpha * C0 / Gamma) * np.exp(1j * U / 2 * n * d) * sech(np.sqrt(Alpha) * n * d)


Ei = np.array([EIni(i) for i in nRange])
# for i in nRange:
#     Ei.append(EIni(i))


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


def FNLStepA(Eq, dz):
    #dx changed to dz for clarity of meaning
    A0 = []
    for j in range(N):
        A0.append(np.exp(1j * Gamma * np.abs(Eq[2 * j]) ** 2 * dz) * Eq[2 * j])
    #abs() changed to np.abs() for consistency

    return A0


def FNLStepB(Eq, dz):
    # dx changed to dz for clarity of meaning
    B0 = []
    for j in range(N):
        B0.append(np.exp(1j * Gamma * np.abs(Eq[2 * j + 1]) ** 2 * dz) * Eq[2 * j + 1])
    # abs() changed to np.abs() for consistency
    return B0


def LNLStep(Eq1, Eq2, dz):
    # dx changed to dz for clarity of meaning
    A1 = []
    B1 = []
    for j in range(N):
        A1.append(np.exp(1j * Gamma * np.abs(Eq1[j]) ** 2 * dz) * Eq1[j])
        B1.append(np.exp(1j * Gamma * np.abs(Eq2[j]) ** 2 * dz) * Eq2[j])
    # abs() changed to np.abs() for consistency
    C = []

    for j in range(N):
        C.append(A1[j])
        C.append(B1[j])

    return np.array(C)


def h(k, z):
    h0 = np.array([[-beta0(z), -C1(z) * np.exp(1j * k) - C2(z)],
                   [-C1(z) * np.exp(-1j * k) - C2(z), -beta1(z)]])

    return h0


K = np.array(range(N)) * 2 * np.pi / N


def LIStepw(A, B, z, deltaz):
    #t changed to z
    #deltat changed to deltaz
    # x = fft(A)
    x=ifft(A,norm="ortho")
    #x should be ifft of A by definition in notes and documentation of scipy.fft
    #norm added for consistency with definition in notes
    # y = fft(B)
    y=ifft(B,norm="ortho")
    #y should be ifft of B by definition
    # norm added for consistency with definition in notes
    u = []
    v = []
    for j in range(N):
        a = np.array([x[j], y[j]])
        b = np.matmul(linalg.expm(-1j * h(K[j], z) * deltaz), a)
        u.append(b[0])
        v.append(b[1])

    # w = ifft(u)
    w=fft(u,norm="ortho")
    # norm added for consistency with definition in notes
    sigma=fft(v,norm="ortho")
    return w,sigma


# def LIStepz(A, B, t, deltat):
#     x = fft(A)
#     y = fft(B)
#     u = []
#     v = []
#     for j in range(N):
#         a = np.array([x[j], y[j]])
#         b = np.matmul(linalg.expm(-1j * h(K[j], t) * deltat), a)
#         u.append(b[0])
#         v.append(b[1])
#
#     z = ifft(v)
#     return z


def OneStep(Eq, zCurr, dz):
    EA = FNLStepA(Eq, dz / 2)
    EB = FNLStepB(Eq, dz / 2)
    # EA = LIStepw(EA, EB, tCurr, dz)
    # EB1 = LIStepz(EA, EB, tCurr, dz)
    w,sigma=LIStepw(EA,EB,zCurr,dz)
    EqPlus = LNLStep(w, sigma, dz / 2)
    return EqPlus
