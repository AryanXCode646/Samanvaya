# Real Data Accuracy Status

This document records the current scientific status of Samanvaya's real-data validation claims.

## Current status

- Synthetic validation: demonstrated within the project benchmark suite
- Real mission validation: not yet executed with checked-in, independently evaluated mission data
- Ground truth: must come from an independent source and is not inferred from the pipeline's own reprojection residuals

## Accurate wording

The project should say:

> "Samanvaya includes a real-data registration framework and conservative validation logic, but full real-mission scientific validation remains pending acquisition of checked-in mission products and independent ground-truth control points."

It should not say:

> "Samanvaya achieves real lunar mission registration accuracy of X px" without measured independent reference data.

## Required next step

When a verified mission pair and independent ground-truth control points are supplied, the registration pipeline should be executed and the resulting RMSE, P95 error, inlier count, and failure rate reported under the real-data evidence section rather than the synthetic benchmark section.
