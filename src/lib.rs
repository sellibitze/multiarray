//! This crate provides types to deal with multi-dimensional data.
//! It basically tries to generalize over `Box<[T]>`, `&[T]` and 
//! `&mut [T]` to multiple dimensions. As a side effect, it also
//! supports one-dimensional arrays that have a stride other than one.
//!
//! # Examples
//!
//! Here's an example of a 3D array. One 2D view and one 1D view
//! into part of the data is created.
//!
//! ```rust
//! use multiarray::*;
//!
//! let mut voxels = Array3D::new([3,4,5], 0); // 3x4x5 ints
//! voxels[[0,0,0]] = 1;
//! voxels[[1,2,3]] = 23;
//! voxels[[2,3,4]] = 42;
//! assert!(voxels[[1,2,3]] == 23);
//! let slice = voxels.eliminated_dim(1, 2);   // 2D slice
//! assert!(slice[[1,3]] == 23);
//! let lane = slice.eliminated_dim(1, 3);     // 1D lane
//! assert!(lane[1] == 23);
//! ```
//!
//! Please note that `[usize; N]` is used as index. For convenience
//! the one-dimensional case also supports `usize` as index in
//! addition to `[usize; 1]`, the one-dimensional views are convertible
//! from borrowed slices (`&[T]` and `&mut[T]`) via
//! `std::convert::{ From, Into }` and also implement the iterator traits
//! `Iterator`, `ExactSizeIterator` and `DoubleEndedIterator`.

extern crate anyrange;

use anyrange::AnyRange;

use std::convert::{ AsRef, AsMut };
use std::marker::PhantomData;
use std::ops::{ Index, IndexMut };


/// Helper type to wrap things. This helps avoiding trait coherency issues
/// w.r.t. `AsRef` and `From`.
#[derive(Copy,Clone)]
pub struct Wrapped<T>(pub T);

impl<T> From<T> for Wrapped<[T; 1]> {
    fn from(x: T) -> Self { Wrapped([x]) }
}

impl<T> From<T> for Wrapped<T> {
    fn from(x: T) -> Self { Wrapped(x) }
}

impl<B: ?Sized, O: AsRef<B>> AsRef<B> for Wrapped<O> {
    fn as_ref(&self) -> &B { self.0.as_ref() }
}

impl<B: ?Sized, O: AsMut<B>> AsMut<B> for Wrapped<O> {
    fn as_mut(&mut self) -> &mut B { self.0.as_mut() }
}


/// Helper trait for creating small `isize` and `usize` arrays
/// of a fixed size. They are used to store information about the
/// memory layout of a multi-dimensional array.
pub unsafe trait LayoutHelper {
    /// type for a small fixed-size array of isize
    type I: AsRef<[isize]> + AsMut<[isize]> + Copy + Clone;

    /// type for a small fixed-size array of usize
    type U: AsRef<[usize]> + AsMut<[usize]> + Copy + Clone;

    /// length of the fixed-size arrays this type can create
    fn dimensions() -> usize;

    /// create array of zeros
    fn zeros_i() -> Self::I;

    /// create array of zeros
    fn zeros_u() -> Self::U;
}

/// Extension trait for dimensions higher than one
pub unsafe trait LayoutHelperExt: LayoutHelper {
    /// Helper type for creating arrays of reduced size (by one).
    type OneLess: LayoutHelper;
}

macro_rules! declare_int_array_maker {
    ($name:ident, $dim:expr, $zax:expr) => {
        pub struct $name;

        unsafe impl LayoutHelper for $name {
            type I = Wrapped<[isize; $dim]>;
            type U = Wrapped<[usize; $dim]>;
            fn dimensions() -> usize { $dim }
            fn zeros_i() -> Self::I { Wrapped($zax) }
            fn zeros_u() -> Self::U { Wrapped($zax) }
        }
    };
    ($name:ident, $dim:expr, $zax:expr, $odl:ident) => {
        declare_int_array_maker!{$name, $dim, $zax}

        unsafe impl LayoutHelperExt for $name {
            type OneLess = $odl;
        }
    };
}

declare_int_array_maker! { Dim1, 1, [0] }
declare_int_array_maker! { Dim2, 2, [0,0], Dim1 }
declare_int_array_maker! { Dim3, 3, [0,0,0], Dim2 }
declare_int_array_maker! { Dim4, 4, [0,0,0,0], Dim3 }
declare_int_array_maker! { Dim5, 5, [0,0,0,0,0], Dim4 }
declare_int_array_maker! { Dim6, 6, [0,0,0,0,0,0], Dim5 }

struct MultiArrayLayout<A> where A: LayoutHelper {
    extents: A::U,
    steps: A::I,
}

impl<A> Copy for MultiArrayLayout<A> where A: LayoutHelper {}

impl<A> Clone for MultiArrayLayout<A> where A: LayoutHelper {
    fn clone(&self) -> Self {
        MultiArrayLayout { extents: self.extents, steps: self.steps }
    }
}

fn c_array_layout(extents: &[usize], steps: &mut [isize]) -> usize {
    let dim = extents.len();
    assert!(dim == steps.len());
    let mut factor = 1;
    for i in (0..dim).rev() {
        steps[i] = factor;
        factor *= extents[i] as isize;
    }
    return factor as usize;
}

impl<A> MultiArrayLayout<A> where A: LayoutHelper {
    /// create new multi array layout with a C-style memory layout
    /// for the given the extents. The second part of the pair
    /// returns the product of all extents and can be used as
    /// size for a `Vec` to create the storage for this multi array.
    fn new_c_style(extents: A::U) -> (Self, usize) {
        let dim = A::dimensions();
        let mut steps = A::zeros_i();
        let count = {
            let ex = extents.as_ref();
            assert!(dim == ex.len());
            c_array_layout(ex, steps.as_mut())
        };
        (MultiArrayLayout { extents: extents, steps: steps }, count)
    }

    /// extents for each dimension
    fn extents(&self) -> &[usize] { self.extents.as_ref() }

    /// steps for each dimension
    fn steps(&self) -> &[isize] { self.steps.as_ref() }

    /// translates a multi dimensional coordinate to an offset
    /// with which the element's memory address can be computed
    fn coord_to_offset(&self, coord: &[usize]) -> isize {
        let dims = A::dimensions();
        let ex = self.extents.as_ref();
        let st = self.steps.as_ref();
        assert!(dims == coord.len());
        assert!(dims == ex.len());
        assert!(dims == st.len());
        let mut acc = 0;
        for i in 0..dims {
            let c = coord[i];
            assert!(c < ex[i]);
            acc += (c as isize) * st[i];
        }
        acc
    }

    fn subsampled_dim(&self, d: usize, factor: usize) -> Self {
        let dims = A::dimensions();
        assert!(d < dims);
        let mut ex2 = self.extents;
        let mut st2 = self.steps;
        {
            let xref = &mut ex2.as_mut()[d];
            let full = *xref;
            *xref = full / factor;
        }
        {
            let sref = &mut st2.as_mut()[d];
            let full = *sref;
            *sref = full * (factor as isize);
        }
        MultiArrayLayout { extents: ex2, steps: st2 }
    }

    fn reversed_dim(&self, d: usize) -> (Self, isize) {
        let dims = A::dimensions();
        let mut st2 = self.steps;
        assert!(d < dims);
        let offset = {
            let s = &mut st2.as_mut()[d];
            let old_step: isize = *s;
            *s = -old_step;
            let x = self.extents()[d] as isize;
            if x == 0 { 0 }
            else { old_step * (x - 1) }
        };
        (MultiArrayLayout { extents: self.extents, steps: st2 }, offset)
    }

    fn swapped_dims(&self, d1: usize, d2: usize) -> Self {
        let dims = A::dimensions();
        let mut ex = self.extents;
        let mut st = self.steps;
        assert!(d1 < dims && d2 < dims);
        ex.as_mut().swap(d1, d2);
        st.as_mut().swap(d1, d2);
        MultiArrayLayout {
            extents: ex,
            steps: st,
        }
    }

    fn sliced_dim<R: AnyRange<usize>>(&self, dim: usize, range: R) -> (Self, isize) {
        let dims = A::dimensions();
        assert!(dim < dims);
        let ex = self.extents();
        let st = self.steps();
        debug_assert!(ex.len() == dims);
        let range = range.to_range(0, ex[dim]);
        assert!(range.start <= range.end && range.end <= ex[dim]);
        let mut l2 = *self;
        l2.extents.as_mut()[dim] = range.end - range.start;
        let offset = range.start as isize * st[dim];
        (l2, offset)
    }
}

impl<A> MultiArrayLayout<A> where A: LayoutHelperExt {
    fn eliminated_dim(&self, dim: usize, coord: usize) -> (MultiArrayLayout<A::OneLess>, isize) {
        let dims1 = A::dimensions();
        let dims2 = A::OneLess::dimensions();
        assert!(dims1 == dims2 + 1);
        assert!(dim < dims1 && coord < self.extents()[dim]);
        let mut ex2 = A::OneLess::zeros_u();
        let mut st2 = A::OneLess::zeros_i();
        let offset = {
            let ex1 = self.extents();
            let st1 = self.steps();
            let ex2 = ex2.as_mut();
            let st2 = st2.as_mut();
            assert!(ex2.len() == dims2 && st2.len() == dims2);
            for (i2, i1) in (0..dims2).zip((0..dims1).filter(|&n| n != dim)) {
                ex2[i2] = ex1[i1];
                st2[i2] = st1[i1];
            }
            coord as isize * st1[dim]
        };
        (MultiArrayLayout { extents: ex2, steps: st2 }, offset)
    }
}


/// Shared view of a multi-dimensional array
#[allow(raw_pointer_derive)]
#[derive(Copy, Clone)]
pub struct MultiArrayRef<'a, T: 'a, A> where A: LayoutHelper {
    layout: MultiArrayLayout<A>,
    data: *const T,
    _m: PhantomData<&'a [T]>,
}

/// Mutable view of a multi-dimensional array
pub struct MultiArrayRefMut<'a, T: 'a, A> where A: LayoutHelper {
    layout: MultiArrayLayout<A>,
    data: *mut T,
    _m: PhantomData<&'a mut [T]>,
}

/// Type for multi-dimensional arrays that are organized linearly in memory
/// much like a C array but with dynamically determined sizes.
///
/// # Example
///
/// ```rust
/// use multiarray::*;
///
/// let mut matrix = Array2D::new([3, 2], 0);
/// matrix[[0,0]] = 1; matrix[[0,1]] = 2;
/// matrix[[1,0]] = 3; matrix[[1,1]] = 4;
/// matrix[[2,0]] = 5; matrix[[2,1]] = 6;
/// ```
pub struct MultiArray<T, A> where A: LayoutHelper {
    layout: MultiArrayLayout<A>,
    data: Box<[T]>,
}

/// Shared view of a 1D array
pub type Array1DRef<'a, T> = MultiArrayRef<'a, T, Dim1>;
/// Shared view of a 2D array
pub type Array2DRef<'a, T> = MultiArrayRef<'a, T, Dim2>;
/// Shared view of a 3D array
pub type Array3DRef<'a, T> = MultiArrayRef<'a, T, Dim3>;
/// Shared view of a 4D array
pub type Array4DRef<'a, T> = MultiArrayRef<'a, T, Dim4>;
/// Shared view of a 5D array
pub type Array5DRef<'a, T> = MultiArrayRef<'a, T, Dim5>;
/// Shared view of a 6D array
pub type Array6DRef<'a, T> = MultiArrayRef<'a, T, Dim6>;

/// Mutable view of a 1D array
pub type Array1DRefMut<'a, T> = MultiArrayRefMut<'a, T, Dim1>;
/// Mutable view of a 2D array
pub type Array2DRefMut<'a, T> = MultiArrayRefMut<'a, T, Dim2>;
/// Mutable view of a 3D array
pub type Array3DRefMut<'a, T> = MultiArrayRefMut<'a, T, Dim3>;
/// Mutable view of a 4D array
pub type Array4DRefMut<'a, T> = MultiArrayRefMut<'a, T, Dim4>;
/// Mutable view of a 5D array
pub type Array5DRefMut<'a, T> = MultiArrayRefMut<'a, T, Dim5>;
/// Mutable view of a 6D array
pub type Array6DRefMut<'a, T> = MultiArrayRefMut<'a, T, Dim6>;

/// Type alias for a 1D array
pub type Array1D<T> = MultiArray<T, Dim1>;
/// Type alias for a 2D array
pub type Array2D<T> = MultiArray<T, Dim2>;
/// Type alias for a 3D array
pub type Array3D<T> = MultiArray<T, Dim3>;
/// Type alias for a 4D array
pub type Array4D<T> = MultiArray<T, Dim4>;
/// Type alias for a 5D array
pub type Array5D<T> = MultiArray<T, Dim5>;
/// Type alias for a 6D array
pub type Array6D<T> = MultiArray<T, Dim6>;


impl<'a, T> From<&'a [T]> for MultiArrayRef<'a, T, Dim1> {
    fn from(slice: &'a [T]) -> Self {
        MultiArrayRef {
            layout: MultiArrayLayout::new_c_style(slice.len().into()).0,
            data: slice.as_ptr(),
            _m: PhantomData,
        }
    }
}

impl<'a, T> From<&'a mut [T]> for MultiArrayRefMut<'a, T, Dim1> {
    fn from(slice: &'a mut [T]) -> Self {
        MultiArrayRefMut {
            layout: MultiArrayLayout::new_c_style(slice.len().into()).0,
            data: slice.as_mut_ptr(),
            _m: PhantomData,
        }
    }
}


impl<T, A> MultiArray<T, A> where T: Clone, A: LayoutHelper {
    /// Create new multi-dimensiopnal array with the given extents (one per dimension)
    pub fn new<X>(extents: X, fill: T) -> Self where X: Into<A::U> {
        let (l, s) = MultiArrayLayout::new_c_style(extents.into());
        MultiArray {
            layout: l,
            data: vec![fill; s].into_boxed_slice(),
        }
    }
}

impl<T, A> MultiArray<T, A> where A: LayoutHelper {
    /// get the array's extents (one item per dimension)
    pub fn extents(&self) -> &[usize] { self.layout.extents() }

    /// create a shared view that allows further manipulations of the view
    pub fn borrow(&self) -> MultiArrayRef<T, A> {
        MultiArrayRef {
            layout: self.layout,
            data: self.data.as_ptr(),
            _m: PhantomData,
        }
    }

    /// create a mutable view that allows further manipulations of the view
    pub fn borrow_mut(&mut self) -> MultiArrayRefMut<T, A> {
        MultiArrayRefMut {
            layout: self.layout,
            data: self.data.as_mut_ptr(),
            _m: PhantomData,
        }
    }

    /// Create a shared view where one given dimension is reversed
    pub fn reversed_dim(&self, dim: usize) -> MultiArrayRef<T, A> {
        self.borrow().reversed_dim(dim)
    }

    /// Create a mutable view where one given dimension is reversed
    pub fn reversed_dim_mut(&mut self, dim: usize) -> MultiArrayRefMut<T, A> {
        self.borrow_mut().reversed_dim(dim)
    }

    /// Create a shared view where one given dimension is subsampled by a given factor
    pub fn subsampled_dim(&self, dim: usize, factor: usize) -> MultiArrayRef<T, A> {
        self.borrow().subsampled_dim(dim, factor)
    }

    /// Create a shared view where one given dimension is subsampled by a given factor
    pub fn subsampled_dim_mut(&mut self, dim: usize, factor: usize) -> MultiArrayRefMut<T, A> {
        self.borrow_mut().subsampled_dim(dim, factor)
    }

    /// Create a shared view where one given dimension is sliced
    pub fn sliced_dim<R>(&self, dim: usize, range: R) -> MultiArrayRef<T, A>
    where R: AnyRange<usize> {
        self.borrow().sliced_dim(dim, range)
    }

    /// Create a mutable view where one given dimension is sliced
    pub fn sliced_dim_mut<R>(&mut self, dim: usize, range: R) -> MultiArrayRefMut<T, A>
    where R: AnyRange<usize> {
        self.borrow_mut().sliced_dim(dim, range)
    }

    /// Create a shared view where the order of two dimensions are swapped
    pub fn swapped_dims(&self, d1: usize, d2: usize) -> MultiArrayRef<T, A> {
        self.borrow().swapped_dims(d1, d2)
    }

    /// Create a mutable view where the order of two dimensions are swapped
    pub fn swapped_dims_mut(&mut self, d1: usize, d2: usize) -> MultiArrayRefMut<T, A> {
        self.borrow_mut().swapped_dims(d1, d2)
    }
}

impl<T, A> MultiArray<T, A> where A: LayoutHelperExt {
    /// Create a lower-dimensional shared view where one dimension
    /// is fixed at the given coordinate.
    pub fn eliminated_dim(&self, dim: usize, coord: usize) -> MultiArrayRef<T, A::OneLess> {
        self.borrow().eliminated_dim(dim, coord)
    }

    /// Create a lower-dimensional mutable view where one dimension
    /// is fixed at the given coordinate.
    pub fn eliminated_dim_mut(&mut self, dim: usize, coord: usize) -> MultiArrayRefMut<T, A::OneLess> {
        self.borrow_mut().eliminated_dim(dim, coord)
    }
}


impl<T, A, I> Index<I> for MultiArray<T, A> where A: LayoutHelper, I: Into<A::U> {
    type Output = T;

    fn index(&self, index: I) -> &T {
        let ofs = self.layout.coord_to_offset(index.into().as_ref());
        debug_assert!(ofs >= 0);
        &self.data[ofs as usize]
    }
}

impl<T, A, I> IndexMut<I> for MultiArray<T, A> where A: LayoutHelper, I: Into<A::U> {
    fn index_mut(&mut self, index: I) -> &mut T {
        let ofs = self.layout.coord_to_offset(index.into().as_ref());
        debug_assert!(ofs >= 0);
        &mut self.data[ofs as usize]
    }
}

impl<'a, T, A, I> Index<I> for MultiArrayRef<'a, T, A> where A: LayoutHelper, I: Into<A::U> {
    type Output = T;

    fn index(&self, index: I) -> &T {
        let ofs = self.layout.coord_to_offset(index.into().as_ref());
        unsafe {
            &*self.data.offset(ofs)
        }
    }
}

impl<'a, T, A, I> Index<I> for MultiArrayRefMut<'a, T, A> where A: LayoutHelper, I: Into<A::U> {
    type Output = T;

    fn index(&self, index: I) -> &T {
        let ofs = self.layout.coord_to_offset(index.into().as_ref());
        unsafe {
            &*(self.data as *const T).offset(ofs)
        }
    }
}

impl<'a, T, A, I> IndexMut<I> for MultiArrayRefMut<'a, T, A> where A: LayoutHelper, I: Into<A::U> {
    fn index_mut(&mut self, index: I) -> &mut T {
        let ofs = self.layout.coord_to_offset(index.into().as_ref());
        unsafe {
            &mut *self.data.offset(ofs)
        }
    }
}

impl<'a, T, A> MultiArrayRefMut<'a, T, A> where A: LayoutHelper {
    /// reborrows the content. This might be useful if you want to
    /// temporarily create another view but keep this one alive.
    pub fn reborrow(&self) -> MultiArrayRef<T, A> {
        MultiArrayRef {
            layout: self.layout.clone(),
            data: self.data as *const T,
            _m: PhantomData,
        }
    }

    /// reborrows the content. This might be useful if you want to
    /// temporarily create another view but keep this one alive.
    pub fn reborrow_mut(&mut self) -> MultiArrayRefMut<T, A> {
        MultiArrayRefMut {
            layout: self.layout.clone(),
            data: self.data,
            _m: PhantomData,
        }
    }
}

impl<'a, T, A> MultiArrayRef<'a, T, A> where A: LayoutHelper {
    /// get the array's extents (one item per dimension)
    pub fn extents(&self) -> &[usize] { &self.layout.extents() }

    /// Create a shared view where one given dimension is reversed
    pub fn reversed_dim(&self, dim: usize) -> Self {
        let (l2, ofs) = self.layout.reversed_dim(dim);
        MultiArrayRef {
            layout: l2,
            data: unsafe { self.data.offset(ofs) },
            _m: PhantomData,
        }
    }

    /// Create a shared view where one given dimension is subsampled by a given factor
    pub fn subsampled_dim(&self, dim: usize, factor: usize) -> Self {
        MultiArrayRef {
            layout: self.layout.subsampled_dim(dim, factor),
            data: self.data,
            _m: PhantomData,
        }
    }

    /// Create a shared view where one given dimension is sliced
    pub fn sliced_dim<R: AnyRange<usize>>(&self, dim: usize, range: R) -> Self {
        let (l2, ofs) = self.layout.sliced_dim(dim, range);
        MultiArrayRef {
            layout: l2,
            data: unsafe { self.data.offset(ofs) },
            _m: PhantomData,
        }
    }

    /// Create a shared view where the order of two dimensions are swapped
    pub fn swapped_dims(&self, d1: usize, d2: usize) -> Self {
        MultiArrayRef {
            layout: self.layout.swapped_dims(d1, d2),
            data: self.data,
            _m: PhantomData,
        }
    }
}

impl<'a, T, A> MultiArrayRef<'a, T, A> where A: LayoutHelperExt {
    /// Create a lower-dimensional shared view where one dimension
    /// is fixed at the given coordinate.
    pub fn eliminated_dim(&self, dim: usize, coord: usize) -> MultiArrayRef<'a, T, A::OneLess> {
        let (l2, ofs) = self.layout.eliminated_dim(dim, coord);
        MultiArrayRef {
            layout: l2,
            data: unsafe { self.data.offset(ofs) },
            _m: PhantomData,
        }
    }
}


impl<'a, T, A> MultiArrayRefMut<'a, T, A> where A: LayoutHelper {
    /// get the array's extents (one item per dimension)
    pub fn extents(&self) -> &[usize] { &self.layout.extents() }

    /// Create a shared view where one given dimension is reversed
    pub fn reversed_dim(self, dim: usize) -> Self {
        let (l2, ofs) = self.layout.reversed_dim(dim);
        MultiArrayRefMut {
            layout: l2,
            data: unsafe { self.data.offset(ofs) },
            _m: PhantomData,
        }
    }

    /// Create a mutable view where one given dimension is subsampled by a given factor
    pub fn subsampled_dim(self, dim: usize, factor: usize) -> Self {
        MultiArrayRefMut {
            layout: self.layout.subsampled_dim(dim, factor),
            data: self.data,
            _m: PhantomData,
        }
    }

    /// Create a mutable view where one given dimension is sliced
    pub fn sliced_dim<R: AnyRange<usize>>(self, dim: usize, range: R) -> Self {
        let (l2, ofs) = self.layout.sliced_dim(dim, range);
        MultiArrayRefMut {
            layout: l2,
            data: unsafe { self.data.offset(ofs) },
            _m: PhantomData,
        }
    }

    /// Create a mutable view where the order of two dimensions are swapped
    pub fn swapped_dims(self, d1: usize, d2: usize) -> Self {
        MultiArrayRefMut {
            layout: self.layout.swapped_dims(d1, d2),
            data: self.data,
            _m: PhantomData,
        }
    }
}

impl<'a, T, A> MultiArrayRefMut<'a, T, A> where A: LayoutHelperExt {
    /// Create a lower-dimensional mutable view where one dimension
    /// is fixed at the given coordinate.
    pub fn eliminated_dim(self, dim: usize, coord: usize) -> MultiArrayRefMut<'a, T, A::OneLess> {
        let (l2, ofs) = self.layout.eliminated_dim(dim, coord);
        MultiArrayRefMut {
            layout: l2,
            data: unsafe { self.data.offset(ofs) },
            _m: PhantomData,
        }
    }
}

macro_rules! declare_iterator {
    ($name:ident, $itemtype:ty, $rawptrname:ident, $borrow_expr:expr ) => {
        impl<'a, T> Iterator for $name<'a, T, Dim1> {
            type Item = $itemtype;

            fn next(&mut self) -> Option<Self::Item> {
                if self.layout.extents()[0] == 0 {
                    None
                } else {
                    unsafe {
                        let $rawptrname = self.data;
                        self.data = self.data.offset(self.layout.steps()[0]);
                        self.layout.extents.as_mut()[0] -= 1;
                        Some($borrow_expr)
                    }
                }
                
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                let x = self.layout.extents()[0];
                (x, Some(x))
            }
        }

        impl<'a, T> ExactSizeIterator for $name<'a, T, Dim1> { }

        impl<'a, T> DoubleEndedIterator for $name<'a, T, Dim1> {
            fn next_back(&mut self) -> Option<Self::Item> {
                if self.layout.extents()[0] == 0 {
                    None
                } else {
                    unsafe {
                        let rx = &mut self.layout.extents.as_mut()[0];
                        let oldsize = *rx;
                        *rx = oldsize - 1;
                        let ofs = oldsize as isize * self.layout.steps.as_ref()[0];
                        let $rawptrname = self.data.offset(ofs);
                        Some($borrow_expr)
                    }
                }
            }
        }
    };
}

declare_iterator! { MultiArrayRef, &'a T, ptr, &*ptr }
declare_iterator! { MultiArrayRefMut, &'a mut T, ptr, &mut *ptr }

#[cfg(test)]
mod test {

    use super::*;

    fn show_matrix(mat: Array2DRef<f64>) {
        let num_rows = mat.extents()[0];
        let num_cols = mat.extents()[1];
        for row_index in 0..num_rows {
            let row_view = mat.eliminated_dim(0, row_index);
            for col_index in 0..num_cols {
                print!(" {}", row_view[[col_index]]);
            }
            println!("");
        }
    }

    #[test]
    fn test1() {
        let mut voxel = Array3D::new([3, 4, 5], 0.0);
        voxel[[1,0,0]] = 100.0;
        voxel[[0,1,0]] =  10.0;
        voxel[[0,0,1]] =   1.0;
        println!("\nextents: {:?}", voxel.extents());
        println!("[0] ="); show_matrix(voxel.eliminated_dim(0, 0));
        println!("[1] ="); show_matrix(voxel.eliminated_dim(0, 1));
        println!("[2] ="); show_matrix(voxel.eliminated_dim(0, 2));
    }

}

