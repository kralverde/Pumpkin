use crate::noise_router::density_function_ast::{BinaryData, LinearData, UnaryData};

use super::{
    ChunkNoiseFunctionRange, IndexToNoisePos, NoisePos,
    StaticIndependentChunkNoiseFunctionComponentImpl,
};

#[derive(Clone)]
pub struct Constant {
    value: f64,
}

impl Constant {
    pub fn new(value: f64) -> Self {
        Self { value }
    }
}

impl ChunkNoiseFunctionRange for Constant {
    fn min(&self) -> f64 {
        self.value
    }

    fn max(&self) -> f64 {
        self.value
    }
}

impl StaticIndependentChunkNoiseFunctionComponentImpl for Constant {
    fn sample(&self, _pos: &impl NoisePos) -> f64 {
        self.value
    }

    fn fill(&self, array: &mut [f64], _mapper: &impl IndexToNoisePos) {
        array.fill(self.value);
    }
}

#[derive(Clone)]
pub struct Linear<'a> {
    pub(crate) arg_index: usize,
    pub(crate) data: &'a LinearData,
}

impl ChunkNoiseFunctionRange for Linear<'_> {
    fn min(&self) -> f64 {
        *self.data.min_value()
    }

    fn max(&self) -> f64 {
        *self.data.max_value()
    }
}

impl<'a> Linear<'a> {
    pub fn new(arg_index: usize, data: &'a LinearData) -> Self {
        Self { arg_index, data }
    }
}

#[derive(Clone)]
pub struct Binary<'a> {
    pub(crate) arg1_index: usize,
    pub(crate) arg2_index: usize,
    pub(crate) data: &'a BinaryData,
}

impl ChunkNoiseFunctionRange for Binary<'_> {
    fn min(&self) -> f64 {
        *self.data.min_value()
    }

    fn max(&self) -> f64 {
        *self.data.max_value()
    }
}

impl<'a> Binary<'a> {
    pub fn new(arg1_index: usize, arg2_index: usize, data: &'a BinaryData) -> Self {
        Self {
            arg1_index,
            arg2_index,
            data,
        }
    }
}

#[derive(Clone)]
pub struct Unary<'a> {
    pub(crate) arg_index: usize,
    pub(crate) data: &'a UnaryData,
}

impl ChunkNoiseFunctionRange for Unary<'_> {
    fn min(&self) -> f64 {
        *self.data.min_value()
    }

    fn max(&self) -> f64 {
        *self.data.max_value()
    }
}

impl<'a> Unary<'a> {
    pub fn new(arg_index: usize, data: &'a UnaryData) -> Self {
        Self { arg_index, data }
    }
}
