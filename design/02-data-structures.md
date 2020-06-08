# Where and how are data structures defined?

- Should be accessible in both C and Python without causing huge overhead in C.
- It's very easy to get this working in pure Python, where we have dynamic
  lookup on all attributes and dynamic typing, but that sacrifices C speed.
- `QobjEvo`/`CQobj/Evo` _must_ be able to use them in an abstract manner (or
  what's the point?)
- For each data structure there should be associated behaviour (unary, binary,
  etc methods) which can have either concrete implementations or virtual
  implementations via a known intermediary type.
- Each data structure should have an underlying C representation to allow it to
  be passed to C functions as a single object otherwise any dispatcher will
  always have an extra step where it 'unwraps' the type and constructs a new
  argument list: we _want_ `cdef CSR *csr_add(CSR *left, CSR *right)` (with or
  without pointer indirection---I need to check Cython's passing conventions),
  but instead we have
```cython
def zcsr_add(complex[::1] dataA, int[::1] indsA, int[::1] indptrA,
             complex[::1] dataB, int[::1] indsB, int[::1] indptrB,
             int nrows, int ncols, int Annz, int Bnnz, double complex alpha=1)
```



## Heavy vs light

Each data structure type must at least contain the fields that are necessary for
it to store all of its data, e.g. `Dense` must have _at least_ a
`double complex *` pointer (or Cython `memoryview`) and information on the shape
of the matrix, and `CSR` must have _at least_ `data: double complex *`,
`indices: Py_ssize_t *` and `indptr: Py_ssize_t` pointers and its shape 2-tuple
`(Py_ssize_t, Py_ssize_t)`.  If they have only these, they are "light", i.e.
they have very few methods attached to them, and we get behaviour from them by
calling external functions.

The alternative is a "heavy" data structure, where all of its methods are
attached to it.

Points to consider:

- Heavy is "more Pythonic", with behaviour attached to a particular class.
- In the heavy model in C, how does somebody access `CSR * Dense`?  Which class
  is it attached to, and how do we know what the type of the output is?
- What is in charge of dispatch in the heavy model in C?  How do we keep that
  type-safe?
- In the light model, how do we access mathematical behaviour from Python?
  Do we have to make a Python wrapper class around every type and fill in every
  function?  How is dispatch done in this model?
- In either model, what is the C type of the output of any of these functions?
  If all types are instances of a parent class, how do we register specialised
  methods?


## Data instantiation and ownership

Last, and most low-level, how do we manage data storage and ownership?  For C
speed and simplicity reasons, it's probably preferable to store everything at C
level behind a pointer owned by the data structure.  We probably don't want to
allocate numpy arrays for every operation at C level, considering we're just
going to be iterating over contiguous regions of memory anyway and that is
unnecessary overhead.  Also, it adds further boiler-plate to every function
operating on the arrays at C level, because they always first have to unwrap the
data down to a raw pointer in order to do anything.

If we _do_ back everything with numpy arrays, then ownership is easy because
numpy and Python's GC take care of everything for us.

If not, the ownership semantics at _C_ level are simple---all data structures
own their own data and _will_ free that data once they go out of scope.  On
instantiation from C code, they take ownership of the memory they're pointed to,
and will free it when Python deallocates them using Cython `__dealloc__`.  On
instantiation from Python code, they take ownership of their data if
`copy=False` and create a copy that they own if `copy=True`.

However, in order for other parts of the library to specify specialisations over
certain data structures from _Python_ space, we should expose the underlying
data components as numpy arrays when requested.  If everything is backed by
numpy in C code, then it's already solved, but if not, then ownership now is
perhaps non-trivial: if the returned numpy array is a view onto the internal
pointer, then what if our data structure gets deallocated before the numpy array
does?  Should someone in Python be allowed to mutate our data array from under
our feet?

Two possible methods are probably both fine here:

1. exposed numpy arrays are non-writeable views onto our data.  `__dealloc__` is
   modified to transfer data ownership to the created numpy array on structure
   deallocation, as we can be certain that the numpy array will always out-live
   us as long as we maintain a reference to it after creation (i.e. we use a
   strongref cache not a weakref one), and then we gain numpy/Python GC
   semantics.  We set the non-writeable flag on the created numpy array to
   prevent someone in Python screwing with us in C code (or I suppose we _could_
   allow it so people can specialise in-place operations?).

2. exposed numpy arrays are writeable _copies_ of our data.  We copy our data
   into a new numpy array every time Python requests a C-backed data attribute.
   We don't have to worry about any ownership considerations at all, but
   in-place operations in Python become impossible and there's a copy penalty
   every time Python accesses a data attribute on us.

I would tend towards method 1, and having written this, I suspect that my
concerns about keeping the numpy arrays immutable are unfounded---we can allow
them to be modified in-place.  `CQobjEvo` and such like will by default make
copies of passed data structures, mediated by some sort of `own` initialisation
parameter.
