// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow_array::{Array, FixedSizeListArray, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, FieldRef, Schema};
use lance::Dataset;
use lance::dataset::{WriteMode, WriteParams};
use lance_arrow::FixedSizeListArrayExt;
use lance_cuvs::{IvfPqBuildParams, VectorIndexBuildParams, VectorIndexKind, build_vector_index};
use lance_linalg::distance::DistanceType;

fn unique_path(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{nanos}"))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    const DIM: i32 = 16;
    const ROWS: usize = 4096;

    let dataset_uri = unique_path("lance-cuvs-smoke-dataset");
    let artifact_uri = unique_path("lance-cuvs-smoke-artifact");

    let schema = Arc::new(Schema::new(vec![Field::new(
        "vector",
        DataType::FixedSizeList(
            FieldRef::new(Field::new("item", DataType::Float32, false)),
            DIM,
        ),
        false,
    )]));

    let values: Vec<f32> = (0..ROWS)
        .flat_map(|row| (0..DIM as usize).map(move |dim| row as f32 + dim as f32 / 100.0))
        .collect();
    let vectors =
        FixedSizeListArray::try_new_from_values(arrow_array::Float32Array::from(values), DIM)?;
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)])?;
    let reader = RecordBatchIterator::new([Ok(batch)], schema);

    Dataset::write(
        reader,
        dataset_uri.to_str().expect("dataset uri utf8"),
        Some(WriteParams {
            mode: WriteMode::Create,
            max_rows_per_file: ROWS,
            max_rows_per_group: 1024,
            ..Default::default()
        }),
    )
    .await?;

    let dataset = Dataset::open(dataset_uri.to_str().expect("dataset uri utf8")).await?;
    let output = build_vector_index(
        &dataset,
        VectorIndexBuildParams {
            column: "vector".to_string(),
            kind: VectorIndexKind::IvfPq(IvfPqBuildParams {
                num_partitions: 8,
                metric_type: DistanceType::L2,
                num_sub_vectors: 4,
                sample_rate: 8,
                max_iters: 20,
                num_bits: 8,
            }),
            artifact_uri: artifact_uri
                .to_str()
                .expect("artifact uri utf8")
                .to_string(),
            batch_size: 1024,
            filter_nan: false,
        },
    )
    .await?;

    println!("artifact_uri={}", output.artifact_uri());
    println!("files={}", output.files().len());
    println!("ivf_centroids={}", output.ivf_centroids().len());
    println!("pq_codebook={}", output.pq_codebook().len());

    Ok(())
}
