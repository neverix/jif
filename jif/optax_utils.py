from optax._src import base
import jax
from typing import NamedTuple, Any
import equinox as eqx


class VmaptaxState(eqx.Module):
    state: Any
    treedef: jax.tree_util.PyTreeDef = eqx.field(static=True)

def vmaptax(optimizer, *, vmap_with_dims) -> base.GradientTransformationExtraArgs:
    init_vmap = lambda x: (
        jax.vmap(optimizer.init)(x)
        if x.shape[0] in vmap_with_dims
        else optimizer.init(x)
    )
    update_vmap = lambda updates, state, params=None, **kwargs: (
        jax.vmap(lambda a, b, c, d: optimizer.update(a, b, c, **d), in_axes=(0, 0, 0, None))(updates, state, params, kwargs)
        if updates.shape[0] in vmap_with_dims
        else optimizer.update(updates, state, params, **kwargs)
    )
    
    def init_fn(params):
        param_leaves, treedef = jax.tree.flatten(params)
        state = tuple(init_vmap(x) for x in param_leaves)
        return VmaptaxState(state, treedef)

    def update_fn(updates, state, params=None, **kwargs):
        update_leaves, update_treedef = jax.tree.flatten(updates)
        assert update_treedef == state.treedef
        assert len(update_leaves) == len(state.state)
        if params is not None:
            param_leaves, param_treedef = jax.tree.flatten(params)
            assert param_treedef == state.treedef
        else:
            param_leaves = (None,) * len(update_leaves)
        flat_updates, new_states = zip(*(
            update_vmap(update, state, params=param, **kwargs)
            for update, state, param in zip(update_leaves, state.state, param_leaves)
        ))
        return update_treedef.unflatten(flat_updates), VmaptaxState(new_states, state.treedef)

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


class OriginalShape(eqx.Module):
    shape: tuple[int, ...] = eqx.field(static=True)


class SqueezeflaptaxState(eqx.Module):
    state: Any
    original_shapes: OriginalShape


def squeeze_dims(params):
    return jax.tree.map(lambda x: x.reshape(*(d for d in x.shape if d != 1)), params)


def squeezeflaptax(optimizer) -> base.GradientTransformationExtraArgs:
    def init_fn(params):
        return SqueezeflaptaxState(optimizer.init(squeeze_dims(params)), jax.tree.map(lambda x: OriginalShape(x.shape), params))

    def update_fn(updates, state, params=None, **kwargs):
        updates = squeeze_dims(updates)
        if params is not None:
            params = squeeze_dims(params)
        squeezed_updates, new_state = optimizer.update(updates, state.state, params, **kwargs)
        updates = jax.tree.map(lambda x, y: x.reshape(y.shape), squeezed_updates, state.original_shapes, is_leaf=lambda x: isinstance(x, OriginalShape))
        return updates, SqueezeflaptaxState(new_state, state.original_shapes)

    return base.GradientTransformationExtraArgs(init_fn, update_fn)
