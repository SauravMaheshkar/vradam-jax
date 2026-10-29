"""VRAdam: Velocity-Regularized Adam, as an optax gradient transformation.

Port of the authors' PyTorch reference (https://github.com/pranavjv/vradam) for
"A Physics-Inspired Optimizer: Velocity Regularized Adam" (arXiv:2505.13196).

VRAdam is AdamW whose step size shrinks when the optimizer is moving fast:

    m_t = b1 * m_{t-1} + (1 - b1) * g_t
    v_t = b2 * v_{t-1} + (1 - b2) * g_t^2
    S_t = sum_i |m_t,i|^power            (over *every* leaf; |g_t,i|^power if normgrad)
    lr_t = eta / (1 + min(b3 * S_t, lr_cutoff))
    p_t = p_{t-1} - lr_t * (m_hat_t / (sqrt(v_hat_t) + eps) + weight_decay * p_{t-1})

S_t is global over the whole pytree, which matches a single torch param group.
Note that S_t uses the raw momentum, not the bias-corrected one, and is not
rooted, so power=2 gives the *squared* L2 norm.
"""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, ArrayLike, Float, Int, PyTree

Tree = PyTree[Float[Array, "..."]]
Mask = PyTree[bool] | Callable[[Tree], PyTree[bool]] | None


class ScaleByVRAdamState(NamedTuple):
    count: Int[ArrayLike, ""]
    mu: Tree
    nu: Tree


def velocity_penalty(
    tree: Tree,
    b3: ArrayLike,
    power: ArrayLike,
    lr_cutoff: ArrayLike,
) -> Float[Array, ""]:
    """The `1 / (1 + min(b3 * sum|x|^power, lr_cutoff))` factor applied to the step.

    Bounded in `[1 / (1 + lr_cutoff), 1]`, so the effective learning rate never
    drops below `eta / (1 + lr_cutoff)`.

    Accumulates in at least float32: summing squared momenta in bf16
    generates rounding errors.
    """

    def leaf_sum(x: Float[Array, "..."]) -> Float[Array, ""]:
        acc = jnp.promote_types(x.dtype, jnp.float32)
        return jnp.sum(jnp.abs(x.astype(acc)) ** power)

    total = sum(jax.tree.leaves(jax.tree.map(leaf_sum, tree)), start=jnp.zeros([]))
    return 1.0 / (1.0 + jnp.minimum(b3 * total, lr_cutoff))


def scale_by_vradam(
    b1: ArrayLike = 0.9,
    b2: ArrayLike = 0.999,
    b3: ArrayLike = 1.0,
    eps: ArrayLike = 1e-8,
    weight_decay: float | Array = 0.0,
    mask: Mask = None,
    power: ArrayLike = 2.0,
    normgrad: bool = False,
    lr_cutoff: ArrayLike = 10.0,
) -> optax.GradientTransformation:
    """Rescale updates by VRAdam, everything except the final `-eta` scaling.

    Weight decay lives here rather than in a chained `optax.add_decayed_weights`
    because the reference damps the decay by the same velocity penalty as the
    Adam direction, and that penalty depends on this transform's momentum.

    Args:
      b1: Decay rate for the first moment (the "velocity").
      b2: Decay rate for the second moment.
      b3: Velocity regularization strength. 0 recovers AdamW.
      eps: Added to `sqrt(nu_hat)` in the denominator.
      weight_decay: Decoupled (AdamW-style) weight decay coefficient.
      mask: Which params get weight decay, as in `optax.adamw`. A bool pytree
        (or prefix of the params), or a callable returning one. `None` decays
        everything.
      power: Exponent in the velocity penalty `sum |m|^power`.
      normgrad: Penalize the raw gradient instead of the momentum.
      lr_cutoff: Caps the penalty, so the step never shrinks below
        `1 / (1 + lr_cutoff)` of the maximum.
    """
    decay = optax.add_decayed_weights(weight_decay, mask)
    skip_decay = isinstance(weight_decay, (int, float)) and weight_decay == 0.0

    def init_fn(params: Tree) -> ScaleByVRAdamState:
        return ScaleByVRAdamState(
            count=jnp.zeros([], jnp.int32),
            mu=optax.tree.zeros_like(params),
            nu=optax.tree.zeros_like(params),
        )

    def update_fn(
        updates: optax.Updates,
        state: optax.OptState,
        params: optax.Params | None = None,
    ) -> tuple[optax.Updates, ScaleByVRAdamState]:
        assert isinstance(state, ScaleByVRAdamState)
        mu = optax.tree.update_moment(updates, state.mu, b1, 1)
        nu = optax.tree.update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count = optax.safe_increment(state.count)

        penalty = velocity_penalty(updates if normgrad else mu, b3, power, lr_cutoff)

        mu_hat = optax.tree.bias_correction(mu, b1, count)
        nu_hat = optax.tree.bias_correction(nu, b2, count)
        new_updates = jax.tree.map(lambda m, v: m / (jnp.sqrt(v) + eps), mu_hat, nu_hat)
        if not skip_decay:
            if params is None:
                raise ValueError("vradam needs `params` for weight decay; pass them to update.")
            new_updates, _ = decay.update(new_updates, decay.init(params), params)
        new_updates = jax.tree.map(lambda u: u * penalty.astype(u.dtype), new_updates)

        return new_updates, ScaleByVRAdamState(
            count=count,
            mu=optax.tree.cast_like(mu, state.mu),
            nu=optax.tree.cast_like(nu, state.nu),
        )

    return optax.GradientTransformation(init_fn, update_fn)


def vradam(
    learning_rate: optax.ScalarOrSchedule = 1e-3,
    b1: ArrayLike = 0.9,
    b2: ArrayLike = 0.999,
    b3: ArrayLike = 1.0,
    eps: ArrayLike = 1e-8,
    weight_decay: float | Array = 1e-5,
    mask: Mask = None,
    power: ArrayLike = 2.0,
    normgrad: bool = False,
    lr_cutoff: ArrayLike = 10.0,
) -> optax.GradientTransformationExtraArgs:
    """VRAdam optimizer.

    `learning_rate` in this case is the *maximum* step size. It may be
    a schedule, in which case the velocity penalty scales the scheduled value.

    Example:
      >>> optimizer = vradam(learning_rate=1e-3, weight_decay=0.01)
      >>> opt_state = optimizer.init(params)
      >>> updates, opt_state = optimizer.update(grads, opt_state, params)
      >>> params = optax.apply_updates(params, updates)
    """
    return optax.chain(
        scale_by_vradam(b1, b2, b3, eps, weight_decay, mask, power, normgrad, lr_cutoff),
        optax.scale_by_learning_rate(learning_rate),
    )
