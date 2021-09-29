// RUN: mlir-opt %s -affine-loop-interchange | FileCheck %s

func @interchange_for_spatial_locality(%A: memref<2048x2048xf64>) {
  affine.for %i = 0 to 2048 {
    affine.for %j = 0 to 2048 {
      %v = affine.load %A[%j, %i] : memref<2048x2048xf64>
      affine.store %v, %A[%j, %i] : memref<2048x2048xf64>
    }
  }
  return
}

// Interchanged for spatial locality.
// CHECK:       affine.load %arg0[%arg1, %arg2] : memref<2048x2048xf64>
// CHECK-NEXT:  affine.store %{{.*}}, %arg0[%arg1, %arg2] : memref<2048x2048xf64>

// -----

func @interchange_for_spatial_temporal(%A: memref<2048xf64>) {
  affine.for %i = 0 to 2048 {
    affine.for %j = 0 to 2048 {
      %v = affine.load %A[%j] : memref<2048xf64>
      affine.load %A[%j] : memref<2048xf64>
      affine.load %A[%i] : memref<2048xf64>
    }
  }
  return
}

// More reuse with %j, %i order.
// CHECK:       affine.load %arg0[%arg1] : memref<2048xf64>
// CHECK-NEXT:  affine.load %0, %arg0[%arg1] : memref<2048xf64>
// CHECK-NEXT:  affine.load %0, %arg0[%arg2] : memref<2048xf64>

// -----

// CHECK-LABEL: func @matmul_ijk
func @matmul_ijk(%A: memref<2048x2048xf64>, %B: memref<2048x2048xf64>, %C: memref<2048x2048xf64>) {
  affine.for %i = 0 to 2048 {
    affine.for %j = 0 to 2048 {
      affine.for %k = 0 to 2048 {
        %a = affine.load %A[%i, %k] : memref<2048x2048xf64>
        %b = affine.load %B[%k, %j] : memref<2048x2048xf64>
        %ci = affine.load %C[%i, %j] : memref<2048x2048xf64>
        %p = mulf %a, %b : f64
        %co = addf %ci, %p : f64
        affine.store %co, %C[%i, %j] : memref<2048x2048xf64>
      }
    }
  }
  return
}

// Test whether the ikj permutation has been found.

// CHECK:      affine.load %arg0[%arg3, %arg4] : memref<2048x2048xf64>
// CHECK-NEXT: affine.load %arg1[%arg4, %arg5] : memref<2048x2048xf64>
// CHECK-NEXT: affine.load %arg2[%arg3, %arg5] : memref<2048x2048xf64>
// CHECK-NEXT: mulf %0, %1 : f64
// CHECK-NEXT: addf %2, %3 : f64
// CHECK-NEXT: affine.store %4, %arg2[%arg3, %arg5] : memref<2048x2048xf64>

// -----

// CHECK-LABEL: func @interchange_for_outer_parallelism
func @interchange_for_outer_parallelism(%A: memref<2048x2048x2048xf64>) {
  affine.for %i = 1 to 2048 {
    affine.for %j = 0 to 2048 {
      affine.for %k = 0 to 2048 {
        %v = affine.load %A[%i, %j, %k] : memref<2048x2048x2048xf64>
        %p = mulf %v, %v : f64
        affine.store %p, %A[%i - 1, %j, %k] : memref<2048x2048x2048xf64>
      }
    }
  }
  return
}

// %j should become outermost - provides outer parallelism and locality.
// CHECK-NEXT: affine.load %arg0[%arg2, %arg1, %arg3]
// CHECK-NEXT: mulf %0, %0 : f64
// CHECK-NEXT: affine.store %1, %arg0[%arg2 - 1, %arg1, %arg3]

// -----

func @test_group_reuse(%A: memref<2048x2048xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>) {
  affine.for %i = 0 to 2048 {
    affine.for %j = 0 to 2048 {
      %v = affine.load %A[%i, %j] : memref<2048x2048xf64>
      affine.store %v, %C[%i, %j] : memref<?x?xf64>
      %u1 = affine.load %A[%j, %i] : memref<2048x2048xf64>
      %u2 = affine.load %A[%j - 1, %i] : memref<2048x2048xf64>
      %u3 = affine.load %A[%j + 1, %i] : memref<2048x2048xf64>
      %s1 = addf %u1, %u2 : f64
      %s2 = addf %s1, %u3 : f64
      affine.store %s2, %B[%j, %i] : memref<?x?xf64>
    }
  }
  return
}

// Interchanged for better reuse.
// CHECK: affine.store %{{.*}}[%arg1, %arg2] : memref<2048x2048xf64>

// -----

func @interchange_invalid(%A: memref<2048x2048xf64>) {
  affine.for %t = 0 to 2048 {
    affine.for %i = 0 to 2048 {
      %u1 = affine.load %A[%t - 1, %i] : memref<2048x2048xf64>
      %u2 = affine.load %A[%t - 1, %i + 1] : memref<2048x2048xf64>
      %u3 = affine.load %A[%t - 1, %i - 1] : memref<2048x2048xf64>
      %s1 = addf %u1, %u2 : f64
      %s2 = addf %s1, %u3 : f64
      affine.store %s2, %A[%t, %i] : memref<2048x2048xf64>
    }
  }
  return
}

// Interchange is invalid.
// CHECK: affine.store %{{.*}}[%arg1, %arg2] : memref<2048x2048xf64>

// -----

// Test for handling other than add/mul.

func @interchange_for_spatial_locality_mod(%A: memref<2048x2048x2048xf64>) {
  affine.for %i = 0 to 2048 {
    affine.for %j = 0 to 2048 {
      affine.for %k = 0 to 2048 {
        %v = affine.load %A[%i mod 2, %k, %j] : memref<2048x2048x2048xf64>
        affine.store %v, %A[%i mod 2, %k, %j] : memref<2048x2048x2048xf64>
        // Interchanged for spatial locality.
        // CHECK:       affine.load %arg0[%{{.*}} mod 2, %arg1, %arg3] : memref<2048x2048x2048xf64>
        // CHECK-NEXT:  affine.store %0, %arg0[%{{.*}} mod 2, %arg1, %arg3] : memref<2048x2048x2048xf64>
      }
    }
  }
  return
}


// -----

// Test to make sure there are no crashes/aborts on things that aren't handled.

func @if_else(%A: memref<2048x2048xf64>) {
  %c0 = constant 0.0 : f64
  %c1 = constant 1.0 : f64
  affine.for %i = 0 to 2048 {
    affine.if affine_set<(d0) : (d0 - 1024 >= 0)> (%i) {
      affine.for %j = 0 to 2048 {
        affine.store %c0, %A[%i, %j] : memref<2048x2048xf64>
      }
    } else {
      affine.for %j = 0 to 2048 {
        affine.store %c1, %A[%i, %j] : memref<2048x2048xf64>
      }
    }
  }
  return
}

// -----

// Test for interchange on imperfect nests.

func @imperfect_nest(%A: memref<2048x2048xf64>) {
  %c0 = constant 0.0 : f64
  %c1 = constant 1.0 : f64
  affine.for %i = 0 to 2048 {
    affine.for %j = 0 to 1024 {
      affine.store %c0, %A[%j, %i] : memref<2048x2048xf64>
    }
    affine.for %j = 1024 to 2048 {
      affine.store %c1, %A[%j, %i] : memref<2048x2048xf64>
    }
  }
  return
}
// CHECK:      for %{{.*}} = 0 to 1024 {
// CHECK-NEXT:   for %{{.*}} = 0 to 2048 {
// CHECK-NEXT:     affine.store %{{.*}}, %arg0[%arg1, %arg2]
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: for %{{.*}} = 1024 to 2048 {
// CHECK-NEXT:   for %{{.*}} = 0 to 2048 {
// CHECK-NEXT:      affine.store %{{.*}}, %arg0[%arg1, %arg2]
// CHECK-NEXT:   }
// CHECK-NEXT: }


// -----

func @multi_nest_seq(%A: memref<2048x2048xf64>, %B: memref<2048x2048xf64>, %C: memref<2048x2048xf64>) {
  affine.for %i = 0 to 2048 {
    affine.for %j = 0 to 2048 {
      %v = affine.load %A[%j, %i] : memref<2048x2048xf64>
      affine.store %v, %A[%j, %i] : memref<2048x2048xf64>
      // CHECK:       affine.load %arg0[%arg3, %arg4] : memref<2048x2048xf64>
      // CHECK-NEXT:  affine.store %{{.*}}, %arg0[%arg3, %arg4] : memref<2048x2048xf64>
    }
  }

  affine.for %i = 0 to 2048 {
    affine.for %j = 0 to 2048 {
      affine.for %k = 0 to 2048 {
        %a = affine.load %A[%i, %k] : memref<2048x2048xf64>
        %b = affine.load %B[%k, %j] : memref<2048x2048xf64>
        %ci = affine.load %C[%i, %j] : memref<2048x2048xf64>
        %p = mulf %a, %b : f64
        %co = addf %ci, %p : f64
        affine.store %co, %C[%i, %j] : memref<2048x2048xf64>
      }
    }
  }
  // CHECK:      affine.load %arg0[%arg3, %arg4] : memref<2048x2048xf64>
  // CHECK-NEXT: affine.load %arg1[%arg4, %arg5] : memref<2048x2048xf64>
  // CHECK-NEXT: affine.load %arg2[%arg3, %arg5] : memref<2048x2048xf64>
  // CHECK-NEXT: mulf
  // CHECK-NEXT: addf
  // CHECK-NEXT: affine.store %{{.*}}, %arg2[%arg3, %arg5] : memref<2048x2048xf64>
  return
}

