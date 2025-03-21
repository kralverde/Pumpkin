use std::cmp::Ordering;

use itertools::Itertools;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

pub fn to_long(float: f32) -> i64 {
    (float * 10000.0) as i64
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq)]
pub struct NoiseValuePoint {
    pub temperature: i64,
    pub humidity: i64,
    pub continentalness: i64,
    pub erosion: i64,
    pub depth: i64,
    pub weirdness: i64,
}

#[derive(Clone, Deserialize)]
pub struct NoiseHypercube {
    pub temperature: ParameterRange,
    pub erosion: ParameterRange,
    pub depth: ParameterRange,
    pub continentalness: ParameterRange,
    pub weirdness: ParameterRange,
    pub humidity: ParameterRange,
    pub offset: f32,
}

impl NoiseHypercube {
    pub fn to_parameters(&self) -> [ParameterRange; 7] {
        [
            self.temperature,
            self.humidity,
            self.continentalness,
            self.erosion,
            self.depth,
            self.weirdness,
            ParameterRange {
                min: to_long(self.offset),
                max: to_long(self.offset),
            },
        ]
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ParameterRange {
    pub min: i64,
    pub max: i64,
}

impl<'de> Deserialize<'de> for ParameterRange {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value: Value = Deserialize::deserialize(deserializer)?;

        match value {
            Value::Array(arr) if arr.len() == 2 => {
                let min = arr[0]
                    .as_f64()
                    .ok_or_else(|| serde::de::Error::custom("Expected float"))?
                    as f32;
                assert!(min >= -2.0);
                let max = arr[1]
                    .as_f64()
                    .ok_or_else(|| serde::de::Error::custom("Expected float"))?
                    as f32;
                assert!(max <= 2.0);
                assert!(min < max, "min is more max");
                Ok(ParameterRange {
                    min: to_long(min),
                    max: to_long(max),
                })
            }
            Value::Number(num) if num.is_f64() => {
                let val = num
                    .as_f64()
                    .ok_or_else(|| serde::de::Error::custom("Expected float"))?
                    as f32;
                let converted_val = to_long(val);
                Ok(ParameterRange {
                    min: converted_val,
                    max: converted_val,
                })
            }
            _ => Err(serde::de::Error::custom(
                "Expected array of two floats or a single float",
            )),
        }
    }
}

impl ParameterRange {
    fn get_distance(&self, noise: i64) -> i64 {
        let l = noise - self.max;
        let m = self.min - noise;
        if l > 0 { l } else { m.max(0) }
    }

    pub fn combine(&self, other: &Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }
}

#[derive(Clone)]
/// T = Biome
pub struct SearchTree<T: Clone> {
    pub root: TreeNode<T>,
}

impl<T: Clone> SearchTree<T> {
    pub fn create(entries: Vec<(T, &NoiseHypercube)>) -> Self {
        assert!(!entries.is_empty(), "entries cannot be empty");

        let leaves: Vec<TreeNode<T>> = entries
            .into_iter()
            .map(|(value, hypercube)| TreeNode::new_leaf(value, hypercube.to_parameters()))
            .collect();

        SearchTree {
            root: create_node(leaves),
        }
    }

    pub fn get(
        &self,
        point: &NoiseValuePoint,
        last_result_node: &mut Option<TreeLeafNode<T>>,
    ) -> Option<T> {
        let point = &[
            point.temperature,
            point.humidity,
            point.continentalness,
            point.erosion,
            point.depth,
            point.weirdness,
            0,
        ];
        let result_node = self.root.get_node(point, last_result_node);
        let result = result_node.clone().map(|node| node.value);
        *last_result_node = result_node;
        result
    }
}

fn create_node<T: Clone>(mut sub_tree: Vec<TreeNode<T>>) -> TreeNode<T> {
    assert!(
        !sub_tree.is_empty(),
        "Need at least one child to build a node"
    );

    if sub_tree.len() == 1 {
        return sub_tree[0].clone();
    }

    if sub_tree.len() <= 6 {
        let mut sorted_sub_tree = sub_tree;
        sorted_sub_tree.sort_by_key(|a| calculate_parameters_average_sum(a.parameters()));
        let bounds = get_enclosing_parameters(&sorted_sub_tree);
        return TreeNode::Branch {
            children: sorted_sub_tree,
            bounds,
        };
    }

    let mut best_range_sum = i64::MAX;
    let mut best_parameter_offset = 0;
    let mut best_batched = Vec::new();

    let parameter_count = sub_tree[0].parameters().len();
    for parameter_offset in 0..parameter_count {
        sort_tree(&mut sub_tree, parameter_offset, false);
        let batched_tree = get_batched_tree(sub_tree.clone());
        let range_sum: i64 = batched_tree
            .iter()
            .map(|node| get_range_length_sum(node.parameters()))
            .sum();

        if best_range_sum > range_sum {
            best_range_sum = range_sum;
            best_parameter_offset = parameter_offset;
            best_batched = batched_tree;
        }
    }

    sort_tree(&mut best_batched, best_parameter_offset, true);

    let children: Vec<TreeNode<T>> = best_batched
        .into_iter()
        .map(|batch| create_node(batch.children()))
        .collect();

    let bounds = get_enclosing_parameters(&children);
    TreeNode::Branch { children, bounds }
}

fn sort_tree<T: Clone>(
    sub_tree: &mut [TreeNode<T>],
    current_parameter: usize,
    absolute_value: bool,
) {
    sub_tree.sort_by(|a, b| {
        let parameter_count = a.parameters().len();
        for parameter_offset in 0..parameter_count {
            let current_index = (current_parameter + parameter_offset) % parameter_count;

            let a_avg = a.bound_average(current_index, absolute_value);
            let b_avg = b.bound_average(current_index, absolute_value);

            let comp = a_avg.cmp(&b_avg);
            if comp != Ordering::Equal {
                return comp;
            }
        }

        Ordering::Equal
    });
}

fn get_batched_tree<T: Clone>(nodes: Vec<TreeNode<T>>) -> Vec<TreeNode<T>> {
    let mut result = Vec::new();
    let mut current_batch = Vec::new();

    // Calculate batch size based on the formula
    let node_count = nodes.len();
    let logged_size_div = (node_count as f64 - 0.01).ln() / 6.0f64.ln();
    let batch_size = 6.0f64.powf(logged_size_div.floor()) as i32 as usize;

    for node in nodes {
        current_batch.push(node);

        if current_batch.len() >= batch_size {
            result.push(TreeNode::Branch {
                children: current_batch.clone(),
                bounds: get_enclosing_parameters(&current_batch),
            });
            current_batch.clear();
        }
    }

    // Add the remaining nodes as the final batch
    if !current_batch.is_empty() {
        result.push(TreeNode::Branch {
            children: current_batch.clone(),
            bounds: get_enclosing_parameters(&current_batch),
        });
    }

    result
}

fn get_enclosing_parameters<T: Clone>(nodes: &[TreeNode<T>]) -> [ParameterRange; 7] {
    assert!(!nodes.is_empty(), "SubTree needs at least one child");
    let mut parameters_acc = *nodes[0].parameters();
    for node in nodes.iter().skip(1) {
        for (parameter_acc, node_parameter) in parameters_acc.iter_mut().zip_eq(node.parameters()) {
            *parameter_acc = node_parameter.combine(parameter_acc);
        }
    }
    parameters_acc
}

fn get_range_length_sum(bounds: &[ParameterRange]) -> i64 {
    bounds
        .iter()
        .map(|range| (range.max - range.min).abs())
        .sum()
}

fn calculate_parameters_average_sum(parameters: &[ParameterRange]) -> i64 {
    parameters
        .iter()
        .map(|range| ((range.min + range.max) / 2).abs())
        .sum()
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum TreeNode<T: Clone> {
    Leaf(TreeLeafNode<T>),
    Branch {
        children: Vec<TreeNode<T>>,
        bounds: [ParameterRange; 7],
    },
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TreeLeafNode<T: Clone> {
    value: T,
    point: [ParameterRange; 7],
}

impl<T: Clone> TreeNode<T> {
    pub fn new_leaf(value: T, point: [ParameterRange; 7]) -> Self {
        TreeNode::Leaf(TreeLeafNode { value, point })
    }

    // pub fn new_branch(children: Vec<TreeNode<T>>, bounds: [ParameterRange; 7]) -> Self {
    //     TreeNode::Branch { children, bounds }
    // }
    fn is_leaf(&self, node: &TreeLeafNode<T>) -> bool {
        match self {
            TreeNode::Leaf(leaf) => leaf.point == node.point,
            TreeNode::Branch { .. } => false,
        }
    }

    pub fn get_node(
        &self,
        point: &[i64; 7],
        alternative: &Option<TreeLeafNode<T>>,
    ) -> Option<TreeLeafNode<T>> {
        match self {
            Self::Leaf(node) => Some(node.clone()),
            Self::Branch { children, .. } => {
                let mut min = alternative
                    .as_ref()
                    .map(|node| squared_distance(&node.point, point))
                    .unwrap_or(i64::MAX);
                let mut tree_leaf_node = alternative.clone();
                for node in children {
                    let distance = squared_distance(node.parameters(), point);
                    if min > distance {
                        let tree_leaf_node2 = node
                            .get_node(point, &tree_leaf_node)
                            .expect("get_node should always return a value on a non empty tree");
                        let distance = if node.is_leaf(&tree_leaf_node2) {
                            distance
                        } else {
                            squared_distance(&tree_leaf_node2.point, point)
                        };

                        if min > distance {
                            min = distance;
                            tree_leaf_node = Some(tree_leaf_node2);
                        }
                    }
                }
                tree_leaf_node
            }
        }
    }

    pub fn parameters(&self) -> &[ParameterRange; 7] {
        match self {
            TreeNode::Leaf(TreeLeafNode { point, .. }) => point,
            TreeNode::Branch { bounds, .. } => bounds,
        }
    }

    pub fn children(self) -> Vec<TreeNode<T>> {
        match self {
            TreeNode::Leaf(TreeLeafNode { .. }) => vec![],
            TreeNode::Branch { children, .. } => children,
        }
    }

    pub fn bound_average(&self, parameter_index: usize, absolute_value: bool) -> i64 {
        let parameter = self.parameters()[parameter_index];
        let average = (parameter.min + parameter.max) / 2;
        if absolute_value {
            average.abs()
        } else {
            average
        }
    }
}

fn squared_distance(a: &[ParameterRange; 7], b: &[i64; 7]) -> i64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| {
            let distance = a.get_distance(*b);
            distance * distance
        })
        .sum()
}

#[cfg(test)]
mod test {
    use pumpkin_util::math::vector2::Vector2;

    use crate::{
        GENERATION_SETTINGS, GeneratorSetting, GlobalProtoNoiseRouter, GlobalRandomConfig,
        NOISE_ROUTER_ASTS, ProtoChunk,
        biome::multi_noise::{TreeNode, create_node},
        read_data_from_file,
    };

    use super::{NoiseHypercube, ParameterRange};

    #[test]
    fn test_create_node_single_leaf() {
        let hypercube = NoiseHypercube {
            temperature: ParameterRange { min: 0, max: 10 },
            humidity: ParameterRange { min: 0, max: 10 },
            continentalness: ParameterRange { min: 0, max: 10 },
            erosion: ParameterRange { min: 0, max: 10 },
            depth: ParameterRange { min: 0, max: 10 },
            weirdness: ParameterRange { min: 0, max: 10 },
            offset: 0.0,
        };
        let leaves = vec![TreeNode::new_leaf(1, hypercube.to_parameters())];
        let node = create_node(leaves.clone());
        assert_eq!(node, leaves[0]);
    }

    #[test]
    fn test_create_node_multiple_leaves_small() {
        let hypercube1 = NoiseHypercube {
            temperature: ParameterRange { min: 0, max: 10 },
            humidity: ParameterRange { min: 0, max: 10 },
            continentalness: ParameterRange { min: 0, max: 10 },
            erosion: ParameterRange { min: 0, max: 10 },
            depth: ParameterRange { min: 0, max: 10 },
            weirdness: ParameterRange { min: 0, max: 10 },
            offset: 0.0,
        };
        let hypercube2 = NoiseHypercube {
            temperature: ParameterRange { min: 10, max: 20 },
            humidity: ParameterRange { min: 10, max: 20 },
            continentalness: ParameterRange { min: 10, max: 20 },
            erosion: ParameterRange { min: 10, max: 20 },
            depth: ParameterRange { min: 10, max: 20 },
            weirdness: ParameterRange { min: 10, max: 20 },
            offset: 0.0,
        };
        let leaves = vec![
            TreeNode::new_leaf(1, hypercube1.to_parameters()),
            TreeNode::new_leaf(2, hypercube2.to_parameters()),
        ];
        let node = create_node(leaves.clone());
        if let TreeNode::Branch { children, .. } = node {
            assert_eq!(children.len(), 2);
            assert_eq!(children[0], leaves[0]);
            assert_eq!(children[1], leaves[1]);
        } else {
            panic!("Expected a branch node");
        }
    }

    #[test]
    fn test_sample_chunk() {
        type PosToPoint = (i32, i32, i32, i64, i64, i64, i64, i64, i64);
        let expected_data: Vec<PosToPoint> =
            read_data_from_file!("../../assets/multi_noise_sample_no_blend_no_beard_0_0_0.json");

        let seed = 0;
        let chunk_pos = Vector2::new(0, 0);
        let random_config = GlobalRandomConfig::new(seed, false);
        let noise_rounter =
            GlobalProtoNoiseRouter::generate(&NOISE_ROUTER_ASTS.overworld, &random_config);

        let surface_config = GENERATION_SETTINGS
            .get(&GeneratorSetting::Overworld)
            .unwrap();

        let mut chunk = ProtoChunk::new(chunk_pos, &noise_rounter, &random_config, surface_config);

        for (x, y, z, tem, hum, con, ero, dep, wei) in expected_data.into_iter() {
            let point = chunk.multi_noise_sampler.sample(x, y, z);
            assert_eq!(point.temperature, tem);
            assert_eq!(point.humidity, hum);
            assert_eq!(point.continentalness, con);
            assert_eq!(point.erosion, ero);
            assert_eq!(point.depth, dep);
            assert_eq!(point.weirdness, wei);
        }
    }
}
