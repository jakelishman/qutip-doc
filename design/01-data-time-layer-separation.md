# Data layer separation

## Definitions

- Low-level components: parts of QuTiP which necessarily interact with the
  underlying data representations.  Typically these components will access
  `Qobj.data` and munge the `fast_csr_matrix` properties, for example state
  creation.
- High-level components: parts of QuTiP which interact only with `Qobj` as
  abstract objects, for example the `qip` package.
- accelerated method: a method which takes only one particular data-layer
  implementation, rather than acting abstractly on all data-layer objects.


## Data layer

There will be a separated data layer, which will expose operations that are
guaranteed to succeed on all data layer objects.  In order to achieve this, the
data layer will have two facets---the library-facing side expected to be used by
low-level QuTiP components, and the data-layer internal side which will
implement dispatch and implicit casting rules for all operations.

### Data layer internals

A core set of operations _must_ be provided with concrete implementations by any
new data-layer implementation.  This is likely to include

- cast to `np.ndarray`
- construct from `np.ndarray`
- copy self
- multiplication by scalar
- matrix multiplication by the same type
- addition of the same type
- equality to the same type

In order to allow these implementations to be added to a large package like
QuTiP, this minimal required set should be kept as small as is absolutely
possible.  We do not want a new implementation to have to fully implement the
entire set of linear algebra operations used in QuTiP, because this will make it
exceptionally difficult to add further capabilities.

In addition, there will be many other functions that exist on the data
layer.  Examples are

- conjugation
- transpose
- adjoint
- trace
- matrix exponential
- partial trace

and many many others.  At first, these will not _need_ to be implemented by a
concrete implementation.  Instead, the data layer will expose the methods, and
upon failing to find a suitable method in the dispatch tables, will convert to
the reference type (say, `np.ndarray`), perform the operation, and convert back.
This is not intended to be fast, it is intended to ensure that the operations
will always succeed.  We can issue a `DEBUG` logging statement or an
`EfficiencyWarning` when this happens for internal development.  This extends to
far more complex methods.  If the data layer has a method, it is guaranteed to
work on types which implement the minimal interface.

However, we also care about speed.  Instances may also implement many, or all of
the additional methods.  In this case, they will register an accelerated method
with the data-layer dispatcher, and no casting will happen.  These underlying
implementations will still be available to be called by other underlying
implementations of the same type, without having to pass-through the dispatch
layer.

I have used `np.ndarray` as the reference type here.  While we are initially
effecting the switch over, it may be simpler to use `fast_csr_matrix` as the
underlying type, since all methods are already written for it.  In the future,
we can swap to the conceptually simple `ndarray` if that is desirable.


### Low-level QuTiP components

Just because we have separated out a data layer doesn't mean that low-level
components will not want to provide accelerated methods if certain underlying
representations are in use.  If a particular method is expected to be used
throughout QuTiP, it may be prudent to register it on the data layer, but there
are plenty which will not be.

Let's take `qutip.destroy` as an example.  This is common enough that it should
have accelerated methods, but it is not itself a data-layer method.

To start, we would declare `qutip.destroy` as being an accelerated-dispatch
method, with a default `ndarray` form (NOTE: the form of the dispatch decorator
is _wildly_ subject to change right now):
```python
from . import Qobj, data

def _destroy_ndarray(size):
    return np.diag(np.sqrt(np.arange(1, size)), 1)

@data.dispatch(default=(data.ndarray, _destroy_ndarray))
def destroy(size: int) -> Qobj:
    """docstring"""
```
We may then also register further methods, say for `fast_csr_matrix`:
```python
def _destroy_fast_csr_matrix(size):
    ...
    return out

destroy.register(type=data.fast_csr_matrix, method=_destroy_fast_csr_matrix)
```
We would then call `qutip.destroy` simply as `qutip.destroy(3)` or
`qutip.destroy(3, type=data.fast_csr_matrix)`.  The default output type could
also be controlled by global QuTiP settings.

If we had a third data layer type, say `data.tensorflow` (or whatever), which
does not have the method defined, then the dispatch would do something
equivalent to
```python
def dispatch(self, datatype, args, kwargs):
    if datatype in self._dispatch:
        method = self._dispatch[datatype]
    else:
        warnings.warn("No accelerated method exists.", EfficiencyWarning)
        method = self._dispatch[self._default]
    return data.cast(method(*args, **kwargs), datatype)
```
where `data.cast` is a data-layer-internal dispatcher which will call an
accelerated method to cast a data-layer object to the correct type, or go via
`np.ndarray` if no accelerated method exists.

Important points:

1. I don't actually need to provide a body for `destroy`, because the dispatch
   table will fill it in.  It's convenient to do a proper `def` for a function
   so it's easier to declare a docstring and type hints.
2. `_destroy_ndarray` does not construct the `Qobj`, because the dispatch will
   do it for us
3. Rules on casting, how defaults are handled and everything like this may be
   the subject of global settings.


## Changes to Qobj

Currently `Qobj` is inherently tied to the `fast_csr_matrix` implementation, and
various components all over QuTiP assume that `Qobj.data` will always be an
instance of `fast_csr_matrix`.  In order for us to seamlessly use multiple
implementations of the underlying matrix structures, there must be a decoupling
between `Qobj`, which represents abstract quantum objects, and the data layer
which implements concrete operations.

In general, we expect that the majority of `Qobj` methods will become fairly
simple pass-throughs to the data layer, and the type of `Qobj.data` will become
an instance of the ABC `DataLayer`, i.e. with few implementation guarantees.

High-level QuTiP components will automatically be able to use any underlying
data representation which fulfils the data model, as they only access `Qobj`
methods.

Low-level methods will be changed to go via the data layer or use accelerated
methods, as described in the previous sections.


## Package organisation

All of the data layer and `Qobj` will exist in a new subpackage, `qutip.core`.
The first order of business of the conversion will be to separate out this
package, and update all references in QuTiP to use it.

The next most important step is to define the data layer interface, and ensure
that the dispatch works.  The dispatch in particular will allow QuTiP to
continue functioning, even while new accelerated methods are being written for
`ndarray`.
