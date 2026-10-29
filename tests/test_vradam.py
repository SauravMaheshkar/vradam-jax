"""VRAdam-specific tests: parity with the authors' reference, masking, edge cases."""

from dataclasses import dataclass
from typing import Any

import chex
import jax
import jax.numpy as jnp
import optax
import pytest
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from vradam import ScaleByVRAdamState, scale_by_vradam, velocity_penalty, vradam

Tree = dict[str, Float[Array, "..."]]


@dataclass(frozen=True)
class Config:
    eta: float = 1e-3
    b1: float = 0.9
    b2: float = 0.999
    b3: float = 1.0
    eps: float = 1e-8
    weight_decay: float = 1e-5
    power: float = 2.0
    normgrad: bool = False
    lr_cutoff: float = 10.0

    def optimizer(self, **overrides: Any) -> optax.GradientTransformationExtraArgs:
        kwargs: dict[str, Any] = {
            "learning_rate": self.eta,
            "b1": self.b1,
            "b2": self.b2,
            "b3": self.b3,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "power": self.power,
            "normgrad": self.normgrad,
            "lr_cutoff": self.lr_cutoff,
        }
        return vradam(**(kwargs | overrides))


class ReferenceVRAdam:
    def __init__(self, params: Tree, cfg: Config):
        self.params = dict(params)
        self.cfg = cfg
        self.step_count = 0
        self.momentum = {k: jnp.zeros_like(v) for k, v in params.items()}
        self.sec_momentum = {k: jnp.zeros_like(v) for k, v in params.items()}

    def step(self, grads: Tree) -> None:
        c = self.cfg
        self.step_count += 1
        total = 0.0
        for k, g in grads.items():
            self.momentum[k] = c.b1 * self.momentum[k] + (1 - c.b1) * g
            src = g if c.normgrad else self.momentum[k]
            total += float(jnp.sum(jnp.abs(src) ** c.power))
            self.sec_momentum[k] = c.b2 * self.sec_momentum[k] + (1 - c.b2) * g * g
        lr = c.eta / (1 + min(c.b3 * total, c.lr_cutoff))
        t = self.step_count
        for k in grads:
            m_hat = self.momentum[k] / (1 - c.b1**t)
            v_hat = self.sec_momentum[k] / (1 - c.b2**t)
            direction = m_hat / (jnp.sqrt(v_hat) + c.eps)
            self.params[k] = self.params[k] * (1 - c.weight_decay * lr) - lr * direction


def make_tree(key: jax.Array, scale: float = 1.0) -> Tree:
    kw, kb, ks = jax.random.split(key, 3)
    return {
        "w": scale * jax.random.normal(kw, (4, 3)),
        "b": scale * jax.random.normal(kb, (3,)),
        "s": scale * jax.random.normal(ks, ()),
    }


@pytest.fixture
def params() -> Tree:
    return make_tree(jax.random.key(0))


@pytest.mark.parametrize(
    ("cfg", "grad_scale"),
    [
        pytest.param(Config(), 0.01, id="defaults-penalty-unsaturated"),
        pytest.param(Config(), 10.0, id="defaults-penalty-saturated"),
        pytest.param(Config(normgrad=True), 0.1, id="normgrad"),
        pytest.param(Config(power=1.0), 0.1, id="power-1"),
        pytest.param(Config(power=3.0, b3=5.0), 0.3, id="power-3"),
        pytest.param(Config(weight_decay=0.0), 0.1, id="no-weight-decay"),
        pytest.param(Config(weight_decay=0.1, eta=1e-2), 0.1, id="big-weight-decay"),
        pytest.param(Config(lr_cutoff=0.5, b1=0.5, b2=0.9), 1.0, id="custom-betas-cutoff"),
    ],
)
def test_matches_reference(params: Tree, cfg: Config, grad_scale: float) -> None:
    ref = ReferenceVRAdam(params, cfg)
    opt = cfg.optimizer()
    update = jax.jit(opt.update)
    state = opt.init(params)
    ours: Any = params

    for key in jax.random.split(jax.random.key(1), 20):
        grads = make_tree(key, grad_scale)
        ref.step(grads)
        updates, state = update(grads, state, ours)
        ours = optax.apply_updates(ours, updates)

    chex.assert_trees_all_close(ours, ref.params, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("normgrad", "grad_scale"),
    [
        pytest.param(False, 1e-2, id="momentum-unsaturated"),
        pytest.param(True, 1e-2, id="grad-unsaturated"),
        pytest.param(False, 1e2, id="momentum-saturated"),
        pytest.param(True, 1e2, id="grad-saturated"),
    ],
)
def test_first_step_closed_form(params: Tree, normgrad: bool, grad_scale: float) -> None:
    # Step 1: m_hat = g and v_hat = g^2, so Adam's direction is g / (|g| + eps)
    # (~sign(g)) and the update is that times -eta * penalty, with m = (1 - b1) g
    # feeding the penalty.
    eta, b1, b3, eps, cutoff = 1e-2, 0.9, 1.0, 1e-8, 10.0
    grads = make_tree(jax.random.key(2), grad_scale)
    opt = vradam(eta, b1=b1, b3=b3, eps=eps, weight_decay=0.0, normgrad=normgrad, lr_cutoff=cutoff)

    updates, _ = opt.update(grads, opt.init(params), params)

    src = grads if normgrad else jax.tree.map(lambda g: (1 - b1) * g, grads)
    total = sum(jnp.sum(x**2) for x in jax.tree.leaves(src))
    penalty = 1.0 / (1.0 + jnp.minimum(b3 * total, cutoff))
    expected = jax.tree.map(lambda g: -eta * g / (jnp.abs(g) + eps) * penalty, grads)
    chex.assert_trees_all_close(updates, expected, rtol=1e-5)
    if grad_scale > 1:
        chex.assert_trees_all_close(penalty, 1.0 / (1.0 + cutoff))


MASKS = [
    pytest.param(None, id="no-mask"),
    pytest.param({"w": True, "b": False, "s": False}, id="pytree-mask"),
    pytest.param(lambda p: jax.tree.map(lambda x: x.ndim > 1, p), id="callable-mask"),
]


@pytest.mark.parametrize("mask", MASKS)
def test_b3_zero_is_adamw(params: Tree, mask: Any) -> None:
    ours = vradam(1e-2, b3=0.0, weight_decay=0.05, mask=mask)
    adamw = optax.adamw(1e-2, weight_decay=0.05, mask=mask)
    s_ours, s_adamw = ours.init(params), adamw.init(params)

    for key in jax.random.split(jax.random.key(3), 5):
        grads = make_tree(key)
        u_ours, s_ours = ours.update(grads, s_ours, params)
        u_adamw, s_adamw = adamw.update(grads, s_adamw, params)
        chex.assert_trees_all_close(u_ours, u_adamw, rtol=1e-6, atol=1e-9)


def test_mask_only_decays_selected_leaves(params: Tree) -> None:
    grads = make_tree(jax.random.key(4))
    mask = {"w": True, "b": False, "s": False}
    masked = vradam(1e-2, weight_decay=0.5, mask=mask)
    no_decay = vradam(1e-2, weight_decay=0.0)
    full_decay = vradam(1e-2, weight_decay=0.5)

    u_masked, _ = masked.update(grads, masked.init(params), params)
    u_none, _ = no_decay.update(grads, no_decay.init(params), params)
    u_full, _ = full_decay.update(grads, full_decay.init(params), params)

    assert isinstance(u_masked, dict) and isinstance(u_none, dict) and isinstance(u_full, dict)
    chex.assert_trees_all_close(u_masked["w"], u_full["w"])
    chex.assert_trees_all_close((u_masked["b"], u_masked["s"]), (u_none["b"], u_none["s"]))


def test_all_false_mask_is_no_decay(params: Tree) -> None:
    grads = make_tree(jax.random.key(5))
    masked = vradam(1e-2, weight_decay=0.5, mask=lambda p: jax.tree.map(lambda _: False, p))
    no_decay = vradam(1e-2, weight_decay=0.0)

    u_masked, _ = masked.update(grads, masked.init(params), params)
    u_none, _ = no_decay.update(grads, no_decay.init(params), params)

    chex.assert_trees_all_close(u_masked, u_none)


@pytest.mark.parametrize("total", [0.0, 0.5, 1e3])
def test_velocity_penalty_bounds(total: float) -> None:
    penalty = velocity_penalty({"x": jnp.array([total**0.5])}, b3=2.0, power=2.0, lr_cutoff=10.0)

    chex.assert_trees_all_close(penalty, jnp.asarray(1.0 / (1.0 + min(2.0 * total, 10.0))))
    assert 1.0 / 11.0 <= float(penalty) <= 1.0


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_velocity_penalty_accumulates_in_float32(dtype: DTypeLike) -> None:
    # 2**16 * 1.0 overflows float16's max (65504)
    tree = {"x": jnp.ones((2**16,), dtype)}

    penalty = velocity_penalty(tree, b3=1e-5, power=2.0, lr_cutoff=10.0)

    chex.assert_type(penalty, jnp.float32)
    chex.assert_trees_all_close(penalty, jnp.asarray(1.0 / (1.0 + 2**16 * 1e-5)), rtol=1e-6)


def test_weight_decay_requires_params() -> None:
    tx = scale_by_vradam(weight_decay=0.1)
    params = {"x": jnp.ones(2)}

    with pytest.raises(ValueError, match="needs `params`"):
        tx.update(params, tx.init(params))


def test_zero_weight_decay_works_without_params() -> None:
    tx = scale_by_vradam()
    params = {"x": jnp.ones(2)}

    updates, state = tx.update(params, tx.init(params))

    assert isinstance(state, ScaleByVRAdamState)
    chex.assert_trees_all_equal(state.count, jnp.asarray(1, jnp.int32))
    chex.assert_trees_all_equal_shapes(updates, params)


def test_state_keeps_param_dtype() -> None:
    params = {"x": jnp.ones(3, jnp.bfloat16)}
    tx = scale_by_vradam(weight_decay=0.1)

    _, state = tx.update(params, tx.init(params), params)

    assert isinstance(state, ScaleByVRAdamState)
    chex.assert_trees_all_equal_dtypes((state.mu, state.nu), (params, params))
