{ mkDerivation, base, cuda, filepath, language-c
, stdenv, storable-complex, template-haskell
, cudatoolkit
}:
mkDerivation {
  pname = "cublas";
  version = "0.3.0.0";
  src = ./.;
  libraryHaskellDepends = [
    base cuda filepath language-c storable-complex template-haskell
  ];
  librarySystemDepends = [ cudatoolkit ]; #[ cublas cusparse ];
  homepage = "https://github.com/bmsherman/cublas";
  description = "FFI bindings to the CUDA CUBLAS and CUSPARSE libraries";
  license = stdenv.lib.licenses.bsd3;
}
