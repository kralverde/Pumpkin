use derive_getters::Getters;
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(tag = "_type", content = "value")]
pub enum SplineRepr {
    #[serde(rename(deserialize = "standard"))]
    Standard {
        #[serde(rename(deserialize = "locationFunction"))]
        location_function: Box<DensityFunctionRepr>,
        locations: Box<[f32]>,
        values: Box<[SplineRepr]>,
        derivatives: Box<[f32]>,
    },
    #[serde(rename(deserialize = "fixed"))]
    Fixed { value: f32 },
}

#[derive(Deserialize)]
pub enum BinaryOperation {
    #[serde(rename(deserialize = "ADD"))]
    Add,
    #[serde(rename(deserialize = "MUL"))]
    Mul,
    #[serde(rename(deserialize = "MIN"))]
    Min,
    #[serde(rename(deserialize = "MAX"))]
    Max,
}

#[derive(Deserialize)]
pub enum LinearOperation {
    #[serde(rename(deserialize = "ADD"))]
    Add,
    #[serde(rename(deserialize = "MUL"))]
    Mul,
}

#[derive(Deserialize)]
pub enum UnaryOperation {
    #[serde(rename(deserialize = "ABS"))]
    Abs,
    #[serde(rename(deserialize = "SQUARE"))]
    Square,
    #[serde(rename(deserialize = "CUBE"))]
    Cube,
    #[serde(rename(deserialize = "HALF_NEGATIVE"))]
    HalfNegative,
    #[serde(rename(deserialize = "QUARTER_NEGATIVE"))]
    QuarterNegative,
    #[serde(rename(deserialize = "SQUEEZE"))]
    Squeeze,
}

#[derive(Deserialize)]
pub enum WierdScaledMapper {
    #[serde(rename(deserialize = "TYPE2"))]
    Caves,
    #[serde(rename(deserialize = "TYPE1"))]
    Tunnels,
}

impl WierdScaledMapper {
    #[inline]
    pub fn max_multiplier(&self) -> f64 {
        match self {
            Self::Tunnels => 2f64,
            Self::Caves => 3f64,
        }
    }

    #[inline]
    pub fn scale(&self, value: f64) -> f64 {
        match self {
            Self::Tunnels => {
                if value < -0.5f64 {
                    0.75f64
                } else if value < 0f64 {
                    1f64
                } else if value < 0.5f64 {
                    1.5f64
                } else {
                    2f64
                }
            }
            Self::Caves => {
                if value < -0.75f64 {
                    0.5f64
                } else if value < -0.5f64 {
                    0.75f64
                } else if value < 0.5f64 {
                    1f64
                } else if value < 0.75f64 {
                    2f64
                } else {
                    3f64
                }
            }
        }
    }
}

#[derive(Copy, Clone, Deserialize)]
pub enum WrapperType {
    Interpolated,
    #[serde(rename(deserialize = "FlatCache"))]
    CacheFlat,
    Cache2D,
    CacheOnce,
    CellCache,
}

#[derive(Deserialize, Getters)]
pub struct NoiseData {
    #[serde(rename(deserialize = "noise"))]
    noise_id: String,
    #[serde(rename(deserialize = "xzScale"))]
    xz_scale: f64,
    #[serde(rename(deserialize = "yScale"))]
    y_scale: f64,
}

#[derive(Deserialize, Getters)]
pub struct ShiftedNoiseData {
    #[serde(rename(deserialize = "xzScale"))]
    xz_scale: f64,
    #[serde(rename(deserialize = "yScale"))]
    y_scale: f64,
    #[serde(rename(deserialize = "noise"))]
    noise_id: String,
}

#[derive(Deserialize, Getters)]
pub struct WeirdScaledData {
    #[serde(rename(deserialize = "noise"))]
    noise_id: String,
    #[serde(rename(deserialize = "rarityValueMapper"))]
    mapper: WierdScaledMapper,
}

#[derive(Deserialize, Getters)]
pub struct InterpolatedNoiseSamplerData {
    #[serde(rename(deserialize = "scaledXzScale"))]
    scaled_xz_scale: f64,
    #[serde(rename(deserialize = "scaledYScale"))]
    scaled_y_scale: f64,
    #[serde(rename(deserialize = "xzFactor"))]
    xz_factor: f64,
    #[serde(rename(deserialize = "yFactor"))]
    y_factor: f64,
    #[serde(rename(deserialize = "smearScaleMultiplier"))]
    smear_scale_multiplier: f64,
    #[serde(rename(deserialize = "maxValue"))]
    max_value: f64,
    // These are unused currently
    //#[serde(rename(deserialize = "xzScale"))]
    //xz_scale: f64,
    //#[serde(rename(deserialize = "yScale"))]
    //y_scale: f64,
}

#[derive(Deserialize, Getters)]
pub struct ClampedYGradientData {
    #[serde(rename(deserialize = "fromY"))]
    from_y: i32,
    #[serde(rename(deserialize = "toY"))]
    to_y: i32,
    #[serde(rename(deserialize = "fromValue"))]
    from_value: f64,
    #[serde(rename(deserialize = "toValue"))]
    to_value: f64,
}

#[derive(Deserialize, Getters)]
pub struct BinaryData {
    #[serde(rename(deserialize = "type"))]
    pub(crate) operation: BinaryOperation,
    #[serde(rename(deserialize = "minValue"))]
    pub(crate) min_value: f64,
    #[serde(rename(deserialize = "maxValue"))]
    pub(crate) max_value: f64,
}

#[derive(Deserialize, Getters)]
pub struct LinearData {
    #[serde(rename(deserialize = "specificType"))]
    operation: LinearOperation,
    argument: f64,
    #[serde(rename(deserialize = "minValue"))]
    min_value: f64,
    #[serde(rename(deserialize = "maxValue"))]
    max_value: f64,
}

impl LinearData {
    #[inline]
    pub fn apply_density(&self, density: f64) -> f64 {
        match self.operation {
            LinearOperation::Add => density + self.argument,
            LinearOperation::Mul => density * self.argument,
        }
    }
}

#[derive(Deserialize, Getters)]
pub struct UnaryData {
    #[serde(rename(deserialize = "type"))]
    operation: UnaryOperation,
    #[serde(rename(deserialize = "minValue"))]
    min_value: f64,
    #[serde(rename(deserialize = "maxValue"))]
    max_value: f64,
}

impl UnaryData {
    #[inline]
    pub fn apply_density(&self, density: f64) -> f64 {
        match self.operation {
            UnaryOperation::Abs => density.abs(),
            UnaryOperation::Square => density * density,
            UnaryOperation::Cube => density * density * density,
            UnaryOperation::HalfNegative => {
                if density > 0.0 {
                    density
                } else {
                    density * 0.5
                }
            }
            UnaryOperation::QuarterNegative => {
                if density > 0.0 {
                    density
                } else {
                    density * 0.25
                }
            }
            UnaryOperation::Squeeze => {
                let clamped = density.clamp(-1.0, 1.0);
                clamped / 2.0 - clamped * clamped * clamped / 24.0
            }
        }
    }
}

#[derive(Deserialize, Getters)]
pub struct ClampData {
    #[serde(rename(deserialize = "minValue"))]
    min_value: f64,
    #[serde(rename(deserialize = "maxValue"))]
    max_value: f64,
}

impl ClampData {
    #[inline]
    pub fn apply_density(&self, density: f64) -> f64 {
        density.clamp(self.min_value, self.max_value)
    }
}

#[derive(Deserialize, Getters)]
pub struct RangeChoiceData {
    #[serde(rename(deserialize = "minInclusive"))]
    min_inclusive: f64,
    #[serde(rename(deserialize = "maxExclusive"))]
    max_exclusive: f64,
}

#[derive(Deserialize, Getters)]
pub struct SplineData {
    #[serde(rename(deserialize = "minValue"))]
    min_value: f64,
    #[serde(rename(deserialize = "maxValue"))]
    max_value: f64,
}

#[derive(Deserialize)]
#[serde(tag = "_class", content = "value")]
pub enum DensityFunctionRepr {
    // This is a placeholder for leaving space for world structures
    Beardifier,
    // These functions is initialized by a seed at runtime
    BlendAlpha,
    BlendOffset,
    BlendDensity {
        input: Box<DensityFunctionRepr>,
    },
    EndIslands,
    Noise {
        #[serde(flatten)]
        data: NoiseData,
    },
    ShiftA {
        #[serde(rename(deserialize = "offsetNoise"))]
        noise_id: String,
    },
    ShiftB {
        #[serde(rename(deserialize = "offsetNoise"))]
        noise_id: String,
    },
    ShiftedNoise {
        #[serde(rename(deserialize = "shiftX"))]
        shift_x: Box<DensityFunctionRepr>,
        #[serde(rename(deserialize = "shiftY"))]
        shift_y: Box<DensityFunctionRepr>,
        #[serde(rename(deserialize = "shiftZ"))]
        shift_z: Box<DensityFunctionRepr>,
        #[serde(flatten)]
        data: ShiftedNoiseData,
    },
    InterpolatedNoiseSampler {
        #[serde(flatten)]
        data: InterpolatedNoiseSamplerData,
    },
    #[serde(rename(deserialize = "WeirdScaledSampler"))]
    WeirdScaled {
        input: Box<DensityFunctionRepr>,
        #[serde(flatten)]
        data: WeirdScaledData,
    },
    // The wrapped function is wrapped in a new wrapper at runtime
    #[serde(rename(deserialize = "Wrapping"))]
    Wrapper {
        #[serde(rename(deserialize = "wrapped"))]
        input: Box<DensityFunctionRepr>,
        #[serde(rename(deserialize = "type"))]
        wrapper: WrapperType,
    },
    // These functions are unchanged except possibly for internal functions
    Constant {
        value: f64,
    },
    #[serde(rename(deserialize = "YClampedGradient"))]
    ClampedYGradient {
        #[serde(flatten)]
        data: ClampedYGradientData,
    },
    #[serde(rename(deserialize = "BinaryOperation"))]
    Binary {
        argument1: Box<DensityFunctionRepr>,
        argument2: Box<DensityFunctionRepr>,
        #[serde(flatten)]
        data: BinaryData,
    },
    #[serde(rename(deserialize = "LinearOperation"))]
    Linear {
        input: Box<DensityFunctionRepr>,
        #[serde(flatten)]
        data: LinearData,
    },
    #[serde(rename(deserialize = "UnaryOperation"))]
    Unary {
        input: Box<DensityFunctionRepr>,
        #[serde(flatten)]
        data: UnaryData,
    },
    Clamp {
        input: Box<DensityFunctionRepr>,
        #[serde(flatten)]
        data: ClampData,
    },
    RangeChoice {
        input: Box<DensityFunctionRepr>,
        #[serde(rename(deserialize = "whenInRange"))]
        when_in_range: Box<DensityFunctionRepr>,
        #[serde(rename(deserialize = "whenOutOfRange"))]
        when_out_range: Box<DensityFunctionRepr>,
        #[serde(flatten)]
        data: RangeChoiceData,
    },
    Spline {
        spline: SplineRepr,
        #[serde(flatten)]
        data: SplineData,
    },
}

/*
impl DensityFunctionRepr {
    #[allow(unused_variables)]
    pub fn as_str(&self) -> &str {
        match self {
            DensityFunctionRepr::BlendAlpha => "BlendAlpha",
            DensityFunctionRepr::Linear { input, data } => "Linear",
            DensityFunctionRepr::ClampedYGradient { data } => "ClampedYGradient",
            DensityFunctionRepr::Constant { value } => "Constant",
            DensityFunctionRepr::Wrapper { input, wrapper } => "Wrapper",
            DensityFunctionRepr::Unary { input, data } => "Unary",
            DensityFunctionRepr::RangeChoice {
                input,
                when_in_range,
                when_out_range,
                data,
            } => "RangeChoice",
            DensityFunctionRepr::Clamp { input, data } => "Clamp",
            DensityFunctionRepr::Spline { spline, data } => "Spline",
            DensityFunctionRepr::WeirdScaled { input, data } => "WeirdScaled",
            DensityFunctionRepr::Binary {
                argument1,
                argument2,
                data,
            } => "Binary",
            DensityFunctionRepr::ShiftedNoise {
                shift_x,
                shift_y,
                shift_z,
                data,
            } => "ShiftedNoise",
            DensityFunctionRepr::BlendDensity { input } => "BlendDensity",
            DensityFunctionRepr::BlendOffset => "BlendOffset",
            DensityFunctionRepr::InterpolatedNoiseSampler { data } => "InterpolatedNoiseSampler",
            DensityFunctionRepr::Noise { data } => "Noise",
            DensityFunctionRepr::EndIslands => "EndIslands",
            DensityFunctionRepr::ShiftA { noise_id } => "ShiftA",
            DensityFunctionRepr::ShiftB { noise_id } => "ShiftB",
        }
    }
}
*/
