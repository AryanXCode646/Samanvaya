# Independent Checkpoint Format

Real benchmark checkpoints must be supplied as CSV or strict JSON with these fields:

```csv
point_id,source_x,source_y,reference_x,reference_y,provenance,quality,notes
cp_001,100.25,80.50,98.75,81.00,independent_control_network,A,curated overlap point
```

Required:

- unique `point_id`
- finite source/reference pixel coordinates
- non-empty `provenance`
- quality class `A`, `B`, or `C`
- coordinates inside source and reference image bounds

Optional geographic fields:

- `source_lat`, `source_lon`
- `reference_lat`, `reference_lon`

The benchmark runner rejects missing provenance, duplicate IDs, invalid coordinates, unsupported quality classes, and out-of-bounds points. Checkpoint metrics are evaluated against the executed transform and are separate from fitting-point reprojection residuals.
