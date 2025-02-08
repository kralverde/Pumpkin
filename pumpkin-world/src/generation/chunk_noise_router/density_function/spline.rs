use crate::noise_router::density_function_ast::SplineData;

use super::ChunkNoiseFunctionRange;

#[derive(Clone)]
pub enum SplineValue {
    Spline(Spline),
    Fixed(f32),
}

#[derive(Clone)]
pub struct SplinePoint {
    pub(crate) location: f32,
    pub(crate) value: SplineValue,
    pub(crate) derivative: f32,
}

impl SplinePoint {
    pub fn new(location: f32, value: SplineValue, derivative: f32) -> Self {
        Self {
            location,
            value,
            derivative,
        }
    }

    pub fn sample_outside_range(&self, sample_location: f32, last_known_sample: f32) -> f32 {
        if self.derivative == 0f32 {
            last_known_sample
        } else {
            self.derivative * (sample_location - self.location) + last_known_sample
        }
    }
}

/// Returns the smallest usize between min..max that does not match the predicate
fn binary_walk(min: usize, max: usize, pred: impl Fn(usize) -> bool) -> usize {
    let mut i = max - min;
    let mut min = min;
    while i > 0 {
        let j = i / 2;
        let k = min + j;
        if pred(k) {
            i = j;
        } else {
            min = k + 1;
            i -= j + 1;
        }
    }
    min
}

pub enum Range {
    In(usize),
    Below,
}

#[derive(Clone)]
pub struct Spline {
    pub(crate) input_index: usize,
    pub(crate) points: Box<[SplinePoint]>,
}

impl Spline {
    pub fn new(input_index: usize, points: Box<[SplinePoint]>) -> Self {
        Self {
            input_index,
            points,
        }
    }

    pub fn find_index_for_location(&self, loc: f32) -> Range {
        let index_greater_than_x =
            binary_walk(0, self.points.len(), |i| loc < self.points[i].location);
        if index_greater_than_x == 0 {
            Range::Below
        } else {
            Range::In(index_greater_than_x - 1)
        }
    }
}

#[derive(Clone)]
pub struct SplineFunction<'a> {
    pub(crate) spline: Spline,
    data: &'a SplineData,
}

impl<'a> SplineFunction<'a> {
    pub fn new(spline: Spline, data: &'a SplineData) -> Self {
        Self { spline, data }
    }
}

impl ChunkNoiseFunctionRange for SplineFunction<'_> {
    #[inline]
    fn min(&self) -> f64 {
        *self.data.min_value()
    }

    #[inline]
    fn max(&self) -> f64 {
        *self.data.max_value()
    }
}
