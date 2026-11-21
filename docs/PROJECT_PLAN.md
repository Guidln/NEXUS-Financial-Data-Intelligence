# Project plan and design notes

## Design principles
- Reproducibility: random seed controls synthetic data.
- Separation: raw records are retained in Bronze; downstream stages do not mutate source.
- Quality: validations are emitted as artifacts.
- Explainability: model metrics and score outputs are persisted.
- Safety: no real PII or banking integrations.

## Proposed extensions
1. Streaming ingestion: Kafka producer emits transaction events; consumer writes to object storage.
2. Spark: implement equivalent batch and Structured Streaming transformations.
3. Orchestration: Airflow DAG for generation, pipeline, tests, model training, and publishing.
4. dbt: staging, intermediate, marts; schema and relationship tests.
5. Observability: Prometheus metrics and Grafana panels for throughput, latency, errors, quality, drift.
6. CI: lint, unit tests, build image, dependency scanning.
7. Cloud: S3 + Glue + EMR + Redshift or an equivalent lakehouse stack.

## Caveats
Fraud labels are rule-generated. The model may learn the generator's rules, so use this dataset to demonstrate workflow, not to claim real predictive performance. A production design would require governance, privacy controls, independent validation, temporal evaluation, cost-sensitive thresholds, human review, and ongoing drift monitoring.
