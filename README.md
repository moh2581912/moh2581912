import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Paramètres du circuit
R = 1  # Résistance (Ohms)
L = 0.1  # Inductance (H)
C = 0.01  # Capacité (F)
E = 220  # Amplitude de la source (V)
f = 50  # Fréquence de la source (Hz)

# Durée d'étude : 5 périodes
T = 1 / f
t_max = 5 * T

# Fonction source v(t)
def v_t(t, regime="impulsion"):
    if regime == "impulsion":
        return E  # Signal constant
    elif regime == "harmonique":
        return E * np.sqrt(2) * np.sin(2 * np.pi * f * t + np.pi / 4)  # Signal sinusoïdal

# Équations différentielles
def rlc_system(t, y, regime):
    i_L, v_C = y
    dv_C = (v_t(t, regime) - v_C - R * i_L) / L
    di_L = i_L / C
    return [dv_C, di_L]

# Résolution avec solve_ivp
def solve_rlc(regime):
    t_eval = np.linspace(0, t_max, 1000)
    sol = solve_ivp(
        rlc_system, [0, t_max], [0, 0], args=(regime,), t_eval=t_eval, method="RK45"
    )
    return sol.t, sol.y

# Étude pour les deux régimes
for regime in ["impulsion", "harmonique"]:
    t, y = solve_rlc(regime)
    i_L, v_C = y

    # Visualisation
    plt.figure()
    plt.plot(t, v_t(t, regime), label="v(t) (source)")
    plt.plot(t, v_C, label="v_C(t) (tension aux bornes de C)")
    plt.plot(t, i_L, label="i_L(t) (courant dans L)")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Circuit RLC - Régime {regime}")
    plt.legend()
    plt.grid()

plt.show()
