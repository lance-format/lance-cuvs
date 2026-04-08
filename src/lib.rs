// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! cuVS-backed vector index build helpers for Lance.
//!
//! This crate is intentionally narrow:
//! - it trains IVF_PQ models with cuVS,
//! - it encodes a Lance dataset into a partition-local artifact,
//! - and it stops before Lance's canonical finalize step.
//!
//! Callers are expected to pass the returned artifact and training outputs back
//! to Lance's own index creation APIs.

mod backend;
mod cuda;
#[cfg(feature = "python")]
mod python;

pub use backend::{
    CuvsVectorBuildBackend, IvfPqBuildParams, PartitionArtifactBuildOutput, TrainedIvfPqIndex,
    VectorBuildBackend, VectorIndexBuildOutput, VectorIndexBuildParams, VectorIndexKind,
    assign_ivf_pq_to_artifact, build_vector_index, train_ivf_pq,
};
