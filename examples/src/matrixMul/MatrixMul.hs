{-# LANGUAGE CPP #-}
{-# LANGUAGE ScopedTypeVariables #-}

--------------------------------------------------------------------------------
--
-- Module    : MatrixMul
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Matrix multiplication using runtime interface and execution control instead
-- of calling C functions via the FFI.
--
--------------------------------------------------------------------------------

module Main where

#include "matrix_mul.h"

-- Friends
import Time
import RandomVector

-- System
import Data.Array
import Data.Array.Unsafe
import Data.Foldable
import System.IO
import Foreign
import Foreign.C.Types
import qualified Foreign.CUDA as CUDA
import qualified Foreign.CUDA.Cublas as CUBLAS
import qualified Foreign.CUDA.Cublas.FFI as CUBLAS

foreign import ccall unsafe "&matrixMul"  p_matrixMul :: FunPtr (Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> CInt -> CInt -> IO ())


--------------------------------------------------------------------------------
-- Reference
--------------------------------------------------------------------------------

matMult :: (Num e, Storable e) => Matrix e -> Matrix e -> IO (Matrix e)
matMult mx my = do
  x <- unsafeFreeze mx
  y <- unsafeFreeze my
  let ((li, lj), (ui, uj))  = bounds x
      ((li',lj'),(ui',uj')) = bounds y
      resBnds | (lj,uj) == (li',ui') = ((li,lj'),(ui,uj'))
              | otherwise            = error "matrix dimensions must agree"

  newListArray resBnds [sum [x!(i,k) * y!(k,j) | k <- range (lj,uj)]
                         | i <- range (li,ui)
                         , j <- range (lj',uj') ]


--------------------------------------------------------------------------------
-- CUDA
--------------------------------------------------------------------------------

matMultCUDA :: (Num e, Storable e) => Matrix e -> Matrix e -> IO (Matrix e)
matMultCUDA xs' ys' = doMult undefined xs' ys'
  where
    doMult :: (Num e', Storable e') => e' -> Matrix e' -> Matrix e' -> IO (Matrix e')
    doMult dummy xs ys = do

      -- Setup matrix parameters
      --
      ((li, lj), (ui, uj))  <- getBounds xs
      ((li',lj'),(ui',uj')) <- getBounds ys
      let wx = rangeSize (lj,uj)
          hx = rangeSize (li,ui)
          wy = rangeSize (lj',uj')
          hy = rangeSize (li',ui')
          resBnds | wx == hy  = ((li,lj'),(ui,uj'))
                  | otherwise = error "matrix dimensions must agree"

      -- Allocate memory and copy test data
      --
      CUDA.allocaArray (wx*hx) $ \d_xs -> do
        CUDA.allocaArray (wy*hy) $ \d_ys -> do
          CUDA.allocaArray (wy*hx) $ \d_zs -> do
            withMatrix xs $ \p -> CUDA.pokeArray (wx*hx) p d_xs
            withMatrix ys $ \p -> CUDA.pokeArray (wy*hy) p d_ys

            -- Launch the kernel
            --
            let gridDim   = (wy`div`BLOCK_SIZE, hx`div`BLOCK_SIZE)
                blockDim  = (BLOCK_SIZE,BLOCK_SIZE,1)
                sharedMem = 2 * BLOCK_SIZE * BLOCK_SIZE * fromIntegral (sizeOf dummy)

            CUDA.launchKernel (castFunPtr p_matrixMul) gridDim blockDim sharedMem Nothing [CUDA.VArg d_xs, CUDA.VArg d_ys, CUDA.VArg d_zs, CUDA.IArg wx, CUDA.IArg wy]

            -- Copy back result
            zs <- newArray_ resBnds
            withMatrix zs $ \p -> CUDA.peekArray (wy*hx) d_zs p
            return zs





test_cuda_memset :: Int -> IO (Vector CFloat)
test_cuda_memset n = do
    CUDA.allocaArray n $ \d_zs -> do
      CUDA.memset d_zs (fromIntegral (sizeOf (undefined :: CFloat) * n)) 0
      zs <- newArray_ (1,n)
      withStorableArray zs $ \p -> CUDA.peekArray n d_zs p
      return zs
      
test_cublas_scale :: Vector CFloat -> IO (Vector CFloat) -- (Matrix CFloat)
test_cublas_scale xs = do
    (li, lf)  <- getBounds xs
    let wx = rangeSize (li,lf)

    -- Allocate memory and copy test data
    --
    CUDA.allocaArray wx $ \d_xs -> do
      withVector xs $ \p -> CUDA.pokeArray wx p d_xs
      hdl <- CUBLAS.create
      print wx
      CUBLAS.scal hdl 10 10.0 d_xs 1
      zs <- newArray_ (li,lf)
      withStorableArray zs $ \p -> CUDA.peekArray wx d_xs p
      CUBLAS.destroy hdl
      return zs


test_cublas_copy :: Vector CFloat -> IO (Vector CFloat)
test_cublas_copy xs = do
    bnd@(l,u)  <- getBounds xs
    let wx = rangeSize (l,u)
    -- Allocate memory and copy test data
    --
    CUDA.allocaArray wx $ \d_xs -> do
      CUDA.allocaArray wx $ \d_zs -> do
        withVector xs $ \p -> CUDA.pokeArray wx p d_xs

        hdl <- CUBLAS.create
        CUBLAS.copy hdl wx d_xs 1 d_zs 1
        CUBLAS.destroy hdl

        zs <- newArray_ bnd
        withStorableArray zs $ \p -> CUDA.peekArray wx d_zs p
        return zs

test_cublas_setMatrix :: Matrix CFloat -> IO (Matrix CFloat)
test_cublas_setMatrix xs = do
    resBnds@((li, lj), (ui, uj))  <- getBounds xs
    let wx = rangeSize (lj,uj)
        hx = rangeSize (li,ui)
    CUDA.allocaArray (wx*hx) $ \d_zs -> do
      withMatrix xs $ \p -> do
        hdl <- CUBLAS.create
        CUBLAS.cublasSetMatrix (fromIntegral wx) (fromIntegral hx)
          (fromIntegral (sizeOf (undefined :: Float)))
          (castPtr p) (fromIntegral wx)
          (castPtr (CUDA.useDevicePtr d_zs)) (fromIntegral wx) 
        CUBLAS.destroy hdl
        zs <- newArray_ resBnds
        withStorableArray zs $ \p -> CUDA.peekArray (wx*hx) d_zs p
        return zs


test_cublas_gemm :: Matrix CFloat -> Matrix CFloat -> IO (Matrix CFloat)
test_cublas_gemm xs ys = do
    ((li, lj), (ui, uj))  <- getBounds xs
    ((li',lj'),(ui',uj')) <- getBounds ys
    let wx = rangeSize (lj,uj)
        hx = rangeSize (li,ui)
        wy = rangeSize (lj',uj')
        hy = rangeSize (li',ui')
        resBnds | wx == hy  = ((li,lj'),(ui,uj'))
                | otherwise = error "matrix dimensions must agree"

    -- Allocate memory and copy test data
    --
    CUDA.allocaArray (wx*hx) $ \d_xs -> do
      CUDA.allocaArray (wy*hy) $ \d_ys -> do
        CUDA.allocaArray (wy*hx) $ \d_zs -> do
          withMatrix xs $ \p -> CUDA.pokeArray (wx*hx) p d_xs
          withMatrix ys $ \p -> CUDA.pokeArray (wy*hy) p d_ys
          CUDA.memset d_zs (fromIntegral (sizeOf (undefined :: CFloat)*wx*hy)) 0
          hdl <- CUBLAS.create

          CUBLAS.destroy hdl

          zs <- newArray_ resBnds
          withStorableArray zs $ \p -> CUDA.peekArray (hx*wy) d_zs p
          return zs

          {- 


          withMatrix xs $ \p_xs -> 
            CUBLAS.cublasSetMatrix (fromIntegral wx) (fromIntegral hx)
              (fromIntegral (sizeOf (undefined :: Float)))
              (castPtr p_xs) (fromIntegral wx)
              (castPtr (CUDA.useDevicePtr d_xs)) (fromIntegral wx) 

          withMatrix ys $ \p_ys ->
            CUBLAS.cublasSetMatrix (fromIntegral wy) (fromIntegral hy)
              (fromIntegral (sizeOf (undefined :: Float)))
              (castPtr p_ys) (fromIntegral wy)
              (castPtr (CUDA.useDevicePtr d_ys)) (fromIntegral wy) 

          {- withMatrix zs $ \p_zs ->
            CUBLAS.cublasSetMatrix (fromIntegral wy) (fromIntegral hx)
              (fromIntegral (sizeOf (undefined :: Float)))
              (castPtr p_zs) (fromIntegral wy)
              (castPtr (CUDA.useDevicePtr d_zs)) (fromIntegral wy)  -}



          -- r <- CUBLAS.dot hdl 10 d_xs 10 d_ys 10
          -- print r
          print (hx,wx,hy,wy)
          -- (128,64,64,192)
          -- CUBLAS.gemm hdl CUBLAS.T CUBLAS.T 16 16 16 0.0 d_xs (16*16) d_ys (16*16) 0.0 d_zs (16*16)
          CUBLAS.scal hdl 100 10.0 d_xs 100

          --  hx wy wx 1.0 d_xs (hx*wx) d_ys (hy*wy) 0 (d_zs) (hx*wy)

          -- 10 10 10 1.0 d_xs 100 d_ys 100 0.0 d_zs 100
          -- wx hy hx 1.0 d_xs (wx*hx) d_ys (wy*hy) 0 (d_zs) (wx*hy)

          -}

          {- 
          -- Launch the kernel
          --
          let gridDim   = (wy`div`BLOCK_SIZE, hx`div`BLOCK_SIZE)
              blockDim  = (BLOCK_SIZE,BLOCK_SIZE,1)
              sharedMem = 2 * BLOCK_SIZE * BLOCK_SIZE * fromIntegral (sizeOf dummy)

          CUDA.launchKernel (castFunPtr p_matrixMul) gridDim blockDim sharedMem Nothing [CUDA.VArg d_xs, CUDA.VArg d_ys, CUDA.VArg d_zs, CUDA.IArg wx, CUDA.IArg wy]
          -}
          -- Copy back result 
          {- zs <- newArray_ resBnds
          withMatrix zs $ \p -> CUDA.peekArray (wy*hx) d_zs p
          return zs
          -}




main :: IO ()
main = 
    test0 >> line >> test1 >> line >> test2 >> line >> test3 >> line >> test4

line :: IO ()
line = putStrLn "==================="

test0 :: IO ()
test0 = do
    putStrLn "device ready test"
    dev   <- CUDA.get
    props <- CUDA.props dev
    putStrLn $ "Using device " ++ show dev ++ ": " ++ CUDA.deviceName props

test1 :: IO ()
test1 = do
    putStrLn "CUDA memset test"
    xs <- test_cuda_memset 32
    mapM_ print =<< getElems xs

test2 :: IO ()
test2 = do
    putStrLn "CUBLAS copy test"
    xs <- randomArr (1,16)
    zs <- test_cublas_copy xs 
    forM_ [1..16] $ \i -> do
      m <- readArray xs i
      n <- readArray zs i
      print (m,n)

test3 :: IO ()
test3 = do
    putStrLn "CUBLAS scal test"
    xs <- randomArr (1,10)
    xs' <- test_cublas_scale xs
    xlst  <- getElems xs
    x'lst <- getElems xs'
    mapM_ print $ zip xlst x'lst

printMatrix :: Matrix CFloat -> IO ()
printMatrix zs = do
    ((li,lj),(ui,uj)) <- getBounds zs
    forM_ [li..ui] $ \i -> do
      forM_ [lj..uj] $ \j -> do
        n <- readArray zs (i,j)
        putStr (show n ++ " ")
      putStrLn ""

test4 :: IO ()
test4 = do
    putStrLn "CUBLAS setMatrix test"
    xs <- randomArr ((1,1),(4,4))
    zs <- test_cublas_setMatrix xs
    printMatrix xs
    putStrLn "------"
    printMatrix zs
  


test5 :: IO ()
test5 = do
    putStrLn "CUBLAS gemm test"
    xs <- randomArr ((1,1),(16,16))
    ys <- randomArr ((1,1),(16,16))
    zs <- test_cublas_gemm xs ys
    forM_ [1..16] $ \i -> do
      forM_ [1..16] $ \j -> do
        n <- readArray zs (i,j)
        putStr (show n ++ " ")
      putStrLn ""
  
  
  {- 

  xs <- randomArr ((1,1),(16,16))-- ((1,1),(8*BLOCK_SIZE, 4*BLOCK_SIZE)) :: IO (Matrix CFloat)
  ys <- randomArr ((1,1),(16,16))  -- ((1,1),(4*BLOCK_SIZE,12*BLOCK_SIZE)) :: IO (Matrix CFloat)

  ref <- matMult xs ys
  -- mat <- matMultCUDA xs ys 
  mat' <- matMultCUBLAS xs ys

  xlst <- getElems xs
  reflst <- getElems ref
  -- matlst <- getElems mat
  mat'lst <- getElems mat'
  
  let f x y = abs ((x-y)/(x+y+epsilon))
      epsilon = 0.0005
  -- return ()
  mapM_ print $ zip xlst mat'lst
  -- mapM_ print  . filter (\(x,y,z)-> z > 0.0005) $ (zipWith (\x y -> (x,y,f x y)) reflst mat'lst)
  
  
  -- print =<< take 5 <$> getElems ref
  -- print =<< take 5 <$> getElems mat


  {- 
  putStr   "== Reference: " >> hFlush stdout
  (tr,ref) <- benchmark 100 (matMult xs ys) (return ())
  putStrLn $  shows (fromInteger (timeIn millisecond tr) / 100::Float) " ms"

  putStr   "== CUDA: " >> hFlush stdout
  (tc,mat) <- benchmark 100 (matMultCUDA xs ys) (CUDA.sync)
  putStrLn $  shows (fromInteger (timeIn millisecond tc) / 100::Float) " ms"

  putStr "== Validating: "
  verify ref mat >>= \rv -> putStrLn $ if rv then "Ok!" else "INVALID!"
  -}
  -}
