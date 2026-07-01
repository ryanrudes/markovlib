"""Exact-inference support for a FACTORED (product) state space.

A product state space is the Cartesian product of ``E`` independent factors, each over ``K`` values —
so the joint alphabet has ``K**E`` states. Exact inference over it is a plain
:class:`~markovlib.model.DiscreteChain` on that ``K**E`` alphabet; what a *factored* problem needs
*beyond* the shared engines are the two constructions this module provides:

* :func:`product_log_emission` — the joint log-emission built from per-factor evidence under a
  **conditional-independence** assumption (the emission FACTORIZES: a sum over factors), and
* :func:`product_marginals` — the marginalization of a joint posterior back to per-factor marginals.

Between them sits an ordinary ``DiscreteChain`` over the ``K**E`` alphabet: the caller builds the joint
transition (transition *construction* is the engine seam — the caller's job), calls
:func:`product_log_emission`, runs :func:`~markovlib.query.smooth` / ``decode``, then
:func:`product_marginals`. This is the "factored & small ⇒ exact product chain" path of the design.

The factor-value of joint state ``k`` at factor ``e`` is ``(k // K**e) % K`` (factor 0 is the least
significant digit); :func:`product_membership` materializes that ``(K**E, E)`` table once. For ``K == 2``
this is the subset/active-set membership ``(k >> e) & 1``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Float = npt.NDArray[np.float64]
Int = npt.NDArray[np.intp]


def product_membership(n_factors: int, n_states: int = 2) -> Int:
    """``(n_states**n_factors, n_factors)`` table: ``[k, e]`` is factor ``e``'s value in joint state ``k``.

    Joint state ``k`` enumerates the product alphabet with factor 0 as the least significant digit:
    ``membership[k, e] = (k // n_states**e) % n_states``. For ``n_states == 2`` this is the bit table
    ``(k >> e) & 1`` — the standard subset/active-set membership.
    """
    if n_factors < 0:
        raise ValueError(f"n_factors must be >= 0; got {n_factors}")
    if n_states < 1:
        raise ValueError(f"n_states must be >= 1; got {n_states}")
    n = n_states**n_factors
    k = np.arange(n)[:, None]
    powers = n_states ** np.arange(n_factors)
    membership: Int = ((k // powers[None, :]) % n_states).astype(np.intp)
    return membership


def product_log_emission(log_evidence: Float, membership: Int) -> Float:
    """``(T, K**E)`` joint log-emission from per-factor evidence, under conditional independence.

    ``log_evidence`` is ``(T, E, K)`` — ``log_evidence[t, e, v]`` is the log-evidence that factor ``e``
    is in value ``v`` at frame ``t``. The factors being conditionally independent given the joint state,
    the joint emission is the per-factor sum::

        log_emission[t, k] = sum_e log_evidence[t, e, membership[k, e]].

    ``membership`` is the ``(K**E, E)`` table from :func:`product_membership`.
    """
    log_evidence = np.asarray(log_evidence, dtype=float)
    if log_evidence.ndim != 3:
        raise ValueError(f"log_evidence must be 3-D (T, E, K); got shape {log_evidence.shape}")
    n_edges = log_evidence.shape[1]
    if membership.ndim != 2 or membership.shape[1] != n_edges:
        raise ValueError(f"membership must be (n_states**E, E) with E={n_edges}; got shape {membership.shape}")
    e_idx = np.arange(n_edges)
    selected = log_evidence[:, e_idx[None, :], membership]  # (T, K**E, E)
    emission: Float = selected.sum(axis=2).astype(np.float64)
    return emission


def product_marginals(joint: Float, membership: Int, n_states: int = 2) -> Float:
    """``(T, E, n_states)`` per-factor marginals from a joint ``(T, K**E)`` distribution.

    ``marginals[t, e, v] = sum_{k : membership[k, e] == v} joint[t, k]`` — the total mass of joint states
    in which factor ``e`` takes value ``v``. Evaluated as a matrix product against the one-hot
    membership, so it is exact and vectorized. For binary factors, ``marginals[..., 1]`` is
    "P(factor active)".
    """
    joint = np.asarray(joint, dtype=float)
    if joint.ndim != 2:
        raise ValueError(f"joint must be 2-D (T, K**E); got shape {joint.shape}")
    if membership.ndim != 2:
        raise ValueError(f"membership must be 2-D (K**E, E); got shape {membership.shape}")
    n_edges = membership.shape[1]
    out = np.empty((joint.shape[0], n_edges, n_states), dtype=np.float64)
    for v in range(n_states):
        out[:, :, v] = joint @ (membership == v).astype(np.float64)
    return out
