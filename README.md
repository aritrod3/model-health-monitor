# Model Health Monitor

A production-oriented system for monitoring deployed tabular classification models. Tracks data drift, performance decay, and provides decision-level explanations over time.

## Overview

The system provides continuous monitoring of a deployed classifier through:

- **Drift Detection**: Statistical tests to identify shifts in feature distributions between training and production data
- **Performance Monitoring**: Tracks model performance metrics over time to detect decay
- **Explanation Generation**: Provides interpretable explanations for model decisions
- **Alerting**: Configurable alerts when drift or performance issues are detected

## Architecture

- **`model/`**: Training, evaluation, and explanation modules
- **`service/`**: FastAPI service for inference and monitoring endpoints
- **`monitoring/`**: Drift detection, metrics collection, and alerting logic
- **`storage/`**: Database interface for storing predictions, metrics, and drift signals
- **`scripts/`**: Local execution utilities

## Usage

Data should be uploaded to `data/raw/` in CSV format. The system will process it through the monitoring pipeline once configured.

See individual module docstrings for interface specifications.
