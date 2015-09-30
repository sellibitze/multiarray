This library provides ways to create and deal with multi-dimensional arrays.
It basically tries to generalize `Box<[T]>`, `&[T]` and `&mut[T]` to multiple
dimensions with some convenient methods that allow you to slice views,
create lower-dimensional slices of it, subsampled or reversed views or
even swap dimensions (for example, to create a transposed view of a 2D
matrix).

# Example

In the following example you'll see how a 3D array and two views into it
(2D and 1D) can be created.

```rust
use multiarray::*;

let mut voxels = Array3D::new([3,4,5], 0); // 3x4x5 ints
voxels[[0,0,0]] = 1;
voxels[[1,2,3]] = 23;
voxels[[2,3,4]] = 42;
assert!(voxels[[1,2,3]] == 23);
let slice = voxels.eliminated_dim(1, 2);   // 2D slice
assert!(slice[[1,3]] == 23);
let lane = slice.eliminated_dim(1, 3);     // 1D lane
assert!(lane[1] == 23);
```

Here, `slice` is a 2D slice of the 3D volume (at y=2) and `lane` is a
one-dimensional view that also acts as an ExactSize/DoubleEnded iterator.
