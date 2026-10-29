"""Port of optax's `contrib/_common_test.py`"""

from collections.abc import Callable, Iterator
from typing import Any

import chex
import jax
import jax.numpy as jnp
import optax
import pytest
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from vradam import vradam

Params = Any
Target = tuple[Params, Params, Callable[[Params], Float[Array, ""]]]


def decay_matrices_only(params: Params) -> Params:
    return jax.tree.map(lambda p: p.ndim > 1, params)


OPT_KWARGS = [
    pytest.param({"learning_rate": 1e-2}, id="default"),
    pytest.param({"learning_rate": 1e-2, "normgrad": True}, id="normgrad"),
    pytest.param({"learning_rate": 1e-2, "b3": 100.0}, id="strong-penalty"),
    pytest.param(
        {"learning_rate": 1e-2, "weight_decay": 1e-4, "mask": decay_matrices_only},
        id="masked-decay",
    ),
]


@pytest.fixture(autouse=True)
def _reset_trace_counter() -> Iterator[None]:
    yield
    chex.clear_trace_counter()


def parabola(dtype: DTypeLike) -> Target:
    initial = jnp.array([-1.0, 10.0, 1.0], dtype=dtype)
    final = jnp.array([1.0, -1.0, 1.0], dtype=dtype)
    return initial, final, lambda p: jnp.sum((p - final) ** 2)


def rosenbrock(dtype: DTypeLike) -> Target:
    a, b = 1.0, 100.0
    initial = jnp.array([0.0, 0.0], dtype=dtype)
    final = jnp.array([a, a**2], dtype=dtype)
    return initial, final, lambda p: (a - p[0]) ** 2 + b * (p[1] - p[0] ** 2) ** 2


def matrix_parabola(dtype: DTypeLike) -> Target:
    initial = jnp.zeros((2, 2), dtype=dtype)
    final = jnp.array([[3.0, -2.0], [1.0, 4.0]], dtype=dtype)
    return initial, final, lambda p: jnp.sum((p - final) ** 2)


def mixed_tensor(dtype: DTypeLike) -> Target:
    initial = (jnp.zeros((3,), dtype=dtype), jnp.zeros((2, 2), dtype=dtype))
    final = (
        jnp.array([1.0, -1.0, 2.0], dtype=dtype),
        jnp.array([[1.0, 0.0], [-1.0, 1.0]], dtype=dtype),
    )

    def objective(p: Params) -> Float[Array, ""]:
        return jnp.sum((p[0] - final[0]) ** 2) + jnp.sum((p[1] - final[1]) ** 2)

    return initial, final, objective


@pytest.mark.parametrize("opt_kwargs", OPT_KWARGS)
def test_accepts_extra_args_and_does_not_recompile(opt_kwargs: dict[str, Any]) -> None:
    opt = vradam(**opt_kwargs)
    initial, _, objective = rosenbrock(jnp.float32)

    @jax.jit
    @chex.assert_max_traces(n=1)
    def step(params: Params, state: optax.OptState) -> tuple[Params, optax.OptState]:
        grads = jax.grad(objective)(params)
        updates, state = opt.update(
            grads, state, params, unexpected_extra_args_your_optimizer_doesnt_expect=1
        )
        return optax.apply_updates(params, updates), state

    params, state = initial, opt.init(initial)
    for _ in range(3):
        params, state = step(params, state)


@pytest.mark.parametrize("target", [parabola, rosenbrock, matrix_parabola, mixed_tensor])
@pytest.mark.parametrize("opt_kwargs", OPT_KWARGS)
def test_converges(opt_kwargs: dict[str, Any], target: Callable[[DTypeLike], Target]) -> None:
    opt = vradam(**opt_kwargs)
    initial, final, objective = target(jnp.float32)

    def step(carry: tuple[Params, optax.OptState], _: None) -> tuple[Any, None]:
        params, state = carry
        updates, state = opt.update(jax.grad(objective)(params), state, params)
        return (optax.apply_updates(params, updates), state), None

    state = opt.init(initial)
    # A no-op map, just to check the state plays nice with tree_map_params.
    state = optax.tree.map_params(opt, lambda v: v, state)
    (params, _), _ = jax.jit(lambda c: jax.lax.scan(step, c, length=30_000))((initial, state))

    chex.assert_trees_all_close(params, final, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("opt_kwargs", OPT_KWARGS)
def test_inject_hyperparams_matches_plain(opt_kwargs: dict[str, Any]) -> None:
    opt = vradam(**opt_kwargs)
    # Callables passed as hyperparams look like schedules to inject_hyperparams.
    opt_inject = optax.inject_hyperparams(vradam, static_args=("mask",))(**opt_kwargs)

    params = [-jnp.ones((2, 3)), jnp.ones((2, 5, 2))]
    grads = [jnp.ones((2, 3)), -jnp.ones((2, 5, 2))]

    updates, state = jax.jit(opt.update)(grads, jax.jit(opt.init)(params), params)
    updates_inject, state_inject = jax.jit(opt_inject.update)(
        grads, jax.jit(opt_inject.init)(params), params
    )

    chex.assert_trees_all_close(updates_inject, updates, rtol=1e-5)
    chex.assert_trees_all_close(state_inject.inner_state, state, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("opt_kwargs", OPT_KWARGS)
def test_preserves_dtype(opt_kwargs: dict[str, Any], dtype: DTypeLike) -> None:
    opt = vradam(**opt_kwargs)
    params = jnp.array([1.0, 2.0], dtype=dtype)
    grads = jax.grad(lambda x: jnp.sum(x**2))(params)

    updates, _ = jax.jit(opt.update)(grads, jax.jit(opt.init)(params), params)

    chex.assert_trees_all_equal_dtypes(updates, params)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("opt_kwargs", OPT_KWARGS)
def test_gradient_accumulation(opt_kwargs: dict[str, Any], dtype: DTypeLike) -> None:
    opt = optax.MultiSteps(vradam(**opt_kwargs), every_k_schedule=4)
    params = jnp.array([1.0, 2.0], dtype=dtype)
    grads = jax.grad(lambda x: jnp.sum(x**2))(params)

    updates, _ = jax.jit(opt.update)(grads, jax.jit(opt.init)(params), params)

    chex.assert_trees_all_equal(updates, jnp.zeros_like(grads))


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64])
@pytest.mark.parametrize("opt_kwargs", OPT_KWARGS)
def test_state_shape_dtype_stability(opt_kwargs: dict[str, Any], dtype: DTypeLike) -> None:
    with jax.enable_x64(dtype == jnp.float64):
        opt = vradam(**opt_kwargs)
        initial, _, objective = parabola(dtype)

        @jax.jit
        def step(params: Params, state: optax.OptState) -> tuple[Params, optax.OptState]:
            grads = jax.grad(objective)(params)
            # Both lax.cond branches must agree on shape and dtype, so a state
            # that drifts dtype (e.g. promoted by a float32 scalar) fails here.
            cond = jax.random.randint(jax.random.key(0), (), 0, 2) == 0
            updates, state = jax.lax.cond(
                cond, lambda: opt.update(grads, state, params), lambda: (params, state)
            )
            return optax.apply_updates(params, updates), state

        params, state = initial, opt.init(initial)
        for _ in range(2):
            params, new_state = step(params, state)
            chex.assert_trees_all_equal_shapes_and_dtypes(new_state, state)
            chex.assert_type(params, dtype)
            state = new_state


@pytest.mark.parametrize("opt_kwargs", OPT_KWARGS)
def test_learning_rate_schedule_matches_constant(opt_kwargs: dict[str, Any]) -> None:
    initial, _, objective = parabola(jnp.float32)

    def run(opt: optax.GradientTransformation) -> Params:
        params, state = initial, opt.init(initial)
        for _ in range(10):
            updates, state = opt.update(jax.grad(objective)(params), state, params)
            params = optax.apply_updates(params, updates)
        return params

    schedule_kwargs: dict[str, Any] = opt_kwargs | {"learning_rate": optax.constant_schedule(1e-3)}
    constant_kwargs: dict[str, Any] = opt_kwargs | {"learning_rate": 1e-3}
    with_schedule = run(vradam(**schedule_kwargs))
    with_constant = run(vradam(**constant_kwargs))

    chex.assert_trees_all_close(with_schedule, with_constant)
