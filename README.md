cublas
======

This Haskell library provides FFI bindings for both the CUBLAS and
CUSPARSE CUDA C libraries. Template Haskell and language-c are used to 
automatically parse the C headers for the libraries and create the 
proper FFI declarations.

The main interfaces to use are `Foreign.CUDA.Cublas` for CUBLAS and
`Foreign.CUDA.Cusparse` for CUSPARSE. There is some primitive marhsalling
done between C types and Haskell types. For more direct FFI imports, use
the `Foreign.CUDA.Cublas.FFI` and `Foreign.CUDA.Cusparse.FFI` modules.

The `Cublas` typeclass represents elements for which CUBLAS operations can
be performed. Its instances are `CFloat`, `CDouble`, `Complex CFloat`, and
`Complex CDouble`. Similarly, there is a `Cusparse` typeclass which has
the same instances.

Use the Haddocs to see what functions are available!


Installation
------------

First, CUDA and Autoconf should be installed. This has been tested with
CUDA versions 5.5 and 6.0. Additionally, you may need
to add some CUDA directories to your `PATH` and `LD_LIBRARY_PATH`
environment variables.

Then, in the base directory, prepare a configure script by running
```shell
autoconf configure.ac > configure
```

Then (also in the base directory),
```shell
cabal configure
cabal install
```
