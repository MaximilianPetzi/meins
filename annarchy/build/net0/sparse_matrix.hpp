#pragma once
#ifdef __AVX__
#include <immintrin.h>  // AVX instructions
#endif
#include <iomanip>

// ANNarchy specific global definitions
#include "helper_functions.hpp"
#include "ANNarchy.h"

// Coordinate 
#include "COOMatrix.hpp"

// List of List
#include "LILMatrix.hpp"
#include "LILInvMatrix.hpp"
#ifndef SKIP_OMP_DEFS
  #include "ParallelLIL.hpp"
#endif

// compressed sparse row
#include "CSRMatrix.hpp"
#include "CSRCMatrix.hpp"
#include "CSRCMatrixT.hpp"
#ifndef SKIP_OMP_DEFS
  #include "CSRCMatrixTOMP.hpp"
#endif

// ELLPACK/ITPACK
#include "ELLMatrix.hpp"

// Hybrid (ELLPACK+Coordinate)
#include "HYBMatrix.hpp"

// allow the user defined definition aka
// "old-style" connectivity definition
#include "Specific.hpp"
