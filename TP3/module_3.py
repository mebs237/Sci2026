from typing import Tuple,List,NamedTuple,Any
from collections import namedtuple
import logging
import numpy as np
from math import hypot
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from os import getenv

level = getenv("LOG_LEVEL" , "INFO")
logging.basicConfig(
    level= level , 
    format="[%(levelname)s][%(name)s] [%(funcName)s] %(message)s"
)
logger = logging.getLogger(__name__)


#=============================
#       Dynamiques
#===============================

def f_in_A(t: float, u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Dynamique dans A = {(x,y) : x+y < 1}.

    Équation : ∂²(x,y) = (-y, x-1)
    État étendu u = (x, y, ẋ, ẏ)
    """
    x, y, dx, dy = u
    return np.array([dx, dy, -y, -1 + x])


def f_in_B(t: float, u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Dynamique dans B = {(x,y) : x+y > 1}.

    Équation : ∂²(x,y) = (-‖v‖ẋ, -1 - ‖v‖ẏ)  avec v = (ẋ,ẏ)
    État étendu u = (x, y, ẋ, ẏ)
    """
    x, y, dx, dy = u
    v_norm = hypot(dx,dy)
    return np.array([dx, dy, -v_norm * dx, -1 - v_norm * dy])


#======================================================================
# Événements de traversée (zone tampon ±eps autour de la frontière Γ )
#======================================================================

def event_A_to_B(eps: float = 1e-10):
    """Détecte quand la trajectoire a pénétré dans B d'un montant eps.

    φ(t) = x+y-1. On veut φ = +eps  →  f_event = φ - eps.
    Se déclenche quand f_event passe de négatif à positif (direction=+1).
    """
    def event(t: float, u: NDArray[np.float64]) -> float:
        return (u[0] + u[1] - 1) - eps

    event.terminal  = True
    event.direction = +1
    return event


def event_B_to_A(eps: float = 1e-10):
    """Détecte quand la trajectoire a pénétré dans A d'un montant eps.

    φ(t) = x+y-1. On veut φ = -eps  →  f_event = φ + eps.
    Se déclenche quand f_event passe de positif à négatif (direction=-1).
    """
    def event(t: float, u: NDArray[np.float64]) -> float:
        return (u[0] + u[1] - 1) + eps

    event.terminal  = True
    event.direction = -1
    return event


def hit_target(t: float, u: NDArray[np.float64]) -> float:
    """Détecte quand x(t) = 2 (ligne d'arrivée)."""
    return u[0] - 2.0

hit_target.terminal = True



#      fonction d'intégration

Solution = NamedTuple('Solution',[ ('t', NDArray[np.float64]) , 
                                  ('u' , NDArray[np.float64] ) 
                                 ]
                     )

def solve_problem(s: float,
                  t_span: Tuple[float, float] = (0, 30),
                  max_cross: int = 100,
                  max_tf_extension: int = 50,
                  eps: float = 1e-10,
                  **kwargs:Any) -> Solution:
    """Résout l'EDO par morceaux jusqu'à ce que x(t) = 2.

    Args:
        s               : paramètre de tir — condition initiale u0 = (-2, -2, s, 2s)
        t_span          : intervalle d'intégration initial (t0, tf)
        max_cross       : nombre maximal de traversées de la frontière
        max_tf_extension: nombre maximal de doublements de tf si x=2 non atteint
        eps             : demi-largeur de la zone tampon autour de la frontière
        **kwargs        : options passées à solve_ivp (rtol, atol, method, ...)

    Returns:
        Solution qui contient :
        - t_all (NDArray[np.float64]) : tableau des temps (N,)
        - u_all (NDArray[np.float64]) : tableau des états (4, N) — lignes : x, y, ẋ, ẏ

    Notes:
        • u0 = (-2, -2, s, 2s) est toujours dans A (φ = -5 ≪ 0) → in_A = True.
        • end_idx = -1 quand un événement a tiré : on exclut sol.t[-1]
          (dernier pas numérique ≈ temps de l'événement) pour n'insérer que
          le temps exact fourni par l'interpolant dense (sol.t_events), évitant
          ainsi les quasi-doublons sources de points anguleux dans le tracé.
    """
    u0 = np.array([-2.0, -2.0, s, 2.0 * s])
    t0 = t_span[0]
    tf = t_span[1]

    current_t = t0
    current_u = u0.copy()

    t_all : List[float] = [current_t]
    u_all : List[NDArray]= [current_u]

    # u0 = (-2,-2,...) → φ(-2,-2) = -5 < 0 → toujours dans A
    in_A = True

    cross_A_to_B = event_A_to_B(eps)
    cross_B_to_A = event_B_to_A(eps)

    cross    = 0
    n_tf_ext = 0

    while current_t < tf:

        f  , bnd_event  = f_in_A , cross_A_to_B  if in_A else f_in_B ,cross_B_to_A 

        sol = solve_ivp(
            f,
            (current_t, tf),
            current_u,
            events=[bnd_event, hit_target],
            **kwargs
        )

        cross_bnd   = sol.t_events[0].size > 0
        hit_x2      = sol.t_events[1].size > 0
        event_fired = cross_bnd or hit_x2

        # end_idx = -1 : on exclut le dernier pas numérique quand un événement
        # a tiré, pour éviter le quasi-doublon avec le temps exact ci-dessous.
        end_idx = -1 if event_fired else None
        if len(sol.t) > 1:
            t_all.extend(sol.t[1:end_idx])
            u_all.extend(sol.y[:, 1:end_idx].T)

        if hit_x2:
            current_t = sol.t_events[1][0]
            current_u = sol.y_events[1][0]
            t_all.append(current_t)
            u_all.append(current_u)
            logger.info(
                    f"x=2 atteint à t={current_t:.4f}, y={current_u[1]:.6f} "
                    f"après {cross} traversées"
            )
            break

        elif cross_bnd:
            current_t = sol.t_events[0][0]
            current_u = sol.y_events[0][0]
            t_all.append(current_t)
            u_all.append(current_u)
            in_A   = not in_A
            cross += 1
            if cross % 10 == 0:
                logger.debug(f"traversée n°{cross} à t={current_t:.4f}")
            if cross >= max_cross:
                logger.warning(f"max_cross={max_cross} atteint, arrêt")
                break

        else:
            current_t = sol.t[-1]
            current_u = sol.y[:, -1]
            if current_t >= tf - 1e-10:
                n_tf_ext += 1
                if n_tf_ext > max_tf_extension:
                    logger.warning("max d'extensions de tf atteint, arrêt")
                    break
                tf += tf - t0
                logger.info(f"tf étendu à {tf:.1f}")
            else:
                logger.error(f"solve_ivp a échoué : status={sol.status}")
                break

    return Solution(np.array(t_all), np.array(u_all).T)


# ==================================================
#           Recherche de s*
# ==================================================

def g(s: float, **kwargs:Any) -> float:
    """Ecart : y(t*(s)) - 2, à l'instant t* où x(t) = 2."""
    _, u = solve_problem(s, **kwargs)
    return u[1, -1] - 2.0


def find_s_value(s_bracket: Tuple[float, float] = (0.0, 100.0), **kwargs:Any) -> float:
    """Trouve s* tel que la trajectoire atteigne (2, 2).

    Résout l'équation g(s) = y(t*(s)) - 2 = 0
    
    Args:
        s_bracket : intervalle [a, b] contenant s* (g doit changer de signe)
        **kwargs  : options transmises à solve_problem

    Returns:
        s* (float) : valeur du paramètre de tir
    """
    def _g(s: float) -> float:
        return g(s, **kwargs)

    s_star, r = brentq(_g, *s_bracket, rtol=np.float64(1e-8), xtol=1e-12, full_output=True)
    if not r.converged:
        raise RuntimeError(f"brentq n'a pas convergé : {r}")
    return float(s_star)



if __name__ == "__main__":
    s_optimal = find_s_value((10,15),rtol=1e-11,atol=1e-18)

    logger.info(f"la valeur optimale de s est : {s_optimal:.8f}")
    
