import numpy as np
from ODEmethods.coefficients import *

# RKMethod -------------------------------------------------------------------------------------------------
rk_methods = {
        # eksplicit RK methods
        "euler" : np.array([
                [0, 0], 
                [None, a5]]),
        "midpoint" : np.array([
                [0, 0, 0], 
                [a6, a6, 0], 
                [None, 0, a5]]),
        "heun" : np.array([
                [0, 0, 0], 
                [a5, a5, 0], 
                [None, a6, a6]]),
        "ralston" : np.array([
                [0, 0, 0], 
                [a10, a10, 0], 
                [None, a2, a7]]),
        "kutta_second" : np.array([
                [0, 0, 0], 
                [a6, a6, 0], 
                [None, 0, a5]]),
        "kutta_third" : np.array([
                [0, 0, 0, 0], 
                [a6, a6, 0, 0], 
                [a5, -a5, a11, 0], 
                [None, a12, a10, a12]]),
        "heun_third" : np.array([
                [0, 0, 0, 0], 
                [a9, a9, 0, 0], 
                [a10, 0, a10, 0], 
                [None, a2, 0, a7]]),
        "ralston_third" : np.array([
                [0, 0, 0, 0], 
                [a6, a6, 0, 0], 
                [a7, 0, a7, 0], 
                [None, a13, a9, a14]]),
        "ssprk3" : np.array([
                [0, 0, 0, 0], 
                [a5, a5, 0, 0], 
                [a6, a2, a2, 0], 
                [None, a12, a12, a10]]),
        "original_rk" : np.array([
                [0, 0, 0, 0, 0], 
                [a6, a6, 0, 0, 0], 
                [a6, 0, a6, 0, 0], 
                [a5, 0, 0, a5, 0], 
                [None, a12, a9, a9, a12]]),
        "third_eight" : np.array([
                [0, 0, 0, 0, 0], 
                [a9, a9, 0, 0, 0], 
                [a10, -a9, a5, 0, 0], 
                [a5, a5, -a5, a5, 0], 
                [None, a8, a3, a3, a8]]),
        "gill" : np.array([
                [0, 0, 0, 0, 0], 
                [a6, a6, 0, 0, 0], 
                [a6, a6 * (-a5 + a15), a5 - a6 * a15, 0, 0],
                [a5, 0, -a6 * a15, a5 + a6 * a15, 0], 
                [None, a12, (a11 - a15) * a12, (a11 + a15) * a12, a12]]),
        # Adaptive step RK methods
        "heun_euler" : np.array([
                [0, 0, 0], 
                [a5, a5, 0], 
                [None, a6, a6], 
                [None, a5, 0]]),
        "fehlberg_rk12" : np.array([
                [0, 0, 0, 0], 
                [a6, a6, 0, 0], 
                [a5, a16, a17, 0], 
                [None, a16, a17, 0], 
                [None, a18, a17, a18]]),
        "bogacki_shampine" : np.array([
                [0, 0, 0, 0, 0], 
                [a6, a6, 0, 0, 0], 
                [a7, 0, a7, 0, 0], 
                [a5, a13, a9, a14, 0],
                [None, a13, a9, a14, 0], 
                [None, a19, a2, a9, a8]]),
        "fehlberg" : np.array([
                [0, 0, 0, 0, 0, 0, 0], 
                [a2, a2, 0, 0, 0, 0, 0], 
                [a3, b31, b32, 0, 0, 0, 0], 
                [a4, b41, b42, b43, 0, 0, 0],
                [a5, b51, b52, b53, b54, 0, 0], 
                [a6, b61, b62, b63, b64, b65, 0], 
                [None, p1, 0, p3, p4, p5, r6],
                [None, c1, 0, c3, c4, c5, 0]]),
        "cash_karp" : np.array([
                [0, 0, 0, 0, 0, 0, 0], 
                [a1, a1, 0, 0, 0, 0, 0], 
                [ck1, ck2, ck3, 0, 0, 0, 0], 
                [ck4, ck1, ck5, ck6, 0, 0, 0],
                [a5, ck7, ck8, ck9, ck10, 0, 0], 
                [ck11, ck12, ck13, ck14, ck15, ck16, 0],
                [None, ck17, 0, ck18, ck19, 0, ck20], 
                [None, ck21, 0, ck22, ck23, ck24, a2]]),
        "dormand_prince" : np.array([
                [0, 0, 0, 0, 0, 0, 0, 0], 
                [a1, a1, 0, 0, 0, 0, 0, 0], 
                [ck1, ck2, ck3, 0, 0, 0, 0, 0],
                [dp1, dp2, dp3, dp4, 0, 0, 0, 0], 
                [dp5, dp6, dp7, dp8, dp9, 0, 0, 0],
                [a5, dp10, dp11, dp12, dp13, dp14, 0, 0], 
                [a5, dp15, 0, dp16, dp17, dp18, dp19, 0],
                [None, dp15, 0, dp16, dp17, dp18, dp19, 0], 
                [None, dp20, 0, dp21, dp22, dp23, dp24, a20]]),
}

# PECE ------------------------------------------------------------------------------------
# Predictors: (Adams-Bashforth)
predictor = [
        np.array([1]),
        np.array([3. / 2., -1. / 2.]),
        np.array([23. / 12., -16. / 12., 5. / 12.]),
        np.array([55. / 24., -59. / 24., 37. / 24., -9. / 24.]),
        np.array([1901. / 720., -2774. / 720., 2616. / 720., -1274. / 720., 251. / 720.]),
]

#Correctors: (Adams-Moulton)
corrector = [
        np.array([0.5, 0.5]),
        np.array([5. / 12., 2. / 3., -1. / 12.]),
        np.array([9. / 24., 19. / 24., -5. / 24., 1. / 24.]),
        np.array([251. / 720., 646. / 720., -264. / 720., 106. / 720., -19. / 720.]),
]

# SymIntegrator ---------------------------------------------------------------------------
# Simplectic integrators
sym_methods = {
        "euler" : np.array([
                [1],
                [1]]),
        "verlet" : np.array([
                [0, 1],
                [0.5, 0.5]]),
        "ruth" : np.array([
                [1, -2./3., 2./3.],
                [-1./24., 3./4., 7./24.]]),
        #leapfrog_yoshida
        "forest_ruth" : np.array([
                [(1./(2.*(2. - sym_crt))), ((1. - sym_crt)/(2.*(2. - sym_crt))), ((1. - sym_crt)/(2.*(2. - sym_crt))), (1./(2.*(2. - sym_crt))) ],
                [(1./(2. - sym_crt)), (-sym_crt/(2. - sym_crt)), (1./(2. - sym_crt)), 0.]]), 
        "VEFRL" : np.array([
                [0., (1. - 2.*sym_lam1)/2., sym_lam1, sym_lam1, (1. - 2.*sym_lam1)/2.],
                [sym_eps1, sym_ksi1,(1. - 2.*(sym_ksi1 + sym_eps1)), sym_ksi1, sym_eps1]]),
        "PEFRL" : np.array([
                [sym_eps2, sym_ksi2, (1. - 2.*(sym_ksi2 + sym_eps2)), sym_ksi2, sym_eps2],
                [(1. -2.*sym_lam2)/2., sym_lam2, sym_lam2, (1. -2.*sym_lam2)/2., 0.]]),
}