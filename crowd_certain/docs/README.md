# Crowd-Certain

[← Back to Main README](../README.md) | [← Back to Documentation Index](index.md) | [Next: Installation Guide →](INSTALLATION.md)

---

A Python library for crowd-sourced label aggregation with uncertainty estimation and confidence scoring.

## Overview

Crowd-Certain is a comprehensive framework for aggregating labels from multiple annotators (crowd workers) while estimating the uncertainty and confidence in the aggregated labels. The library implements various techniques for:

- Calculating worker weights based on their consistency and reliability
- Estimating uncertainty in crowd-sourced labels using multiple techniques
- Generating confidence scores for aggregated labels
- Benchmarking against established crowd-sourcing techniques
- Analyzing the relationship between worker strength and label quality

The framework is particularly useful for researchers and practitioners working with crowd-sourced data, where understanding the reliability of aggregated labels is crucial.

## Key Features

- **Multiple Uncertainty Estimation Techniques**: Standard deviation, entropy, coefficient of variation, prediction interval, and confidence interval
- **Consistency Calculation Methods**: Convert uncertainties to consistency scores using different techniques
- **Worker Weight Calculation**: Calculate weights for workers based on their consistency and reliability
- **Confidence Scoring**: Generate confidence scores for aggregated labels using frequency-based and beta distribution-based strategies
- **Benchmarking**: Compare against established crowd-sourcing techniques like MACE, MajorityVote, MMSR, Wawa, ZeroBasedSkill, GLAD, and DawidSkene
- **Evaluation Metrics**: Calculate AUC, accuracy, and F1 score for different aggregation methods
- **Visualization Tools**: Analyze and visualize the relationship between worker strength and label quality

## Applications

Crowd-Certain is designed for applications where multiple annotators provide labels for the same data, such as:

- Medical image annotation
- Natural language processing tasks
- Computer vision datasets
- Any crowd-sourced labeling task where quality assessment is important

## Documentation

- [Installation Guide](INSTALLATION.md)
- [Usage Examples](USAGE.md)
- [API Reference](API.md)

## Citation

If you use Crowd-Certain in your research, please cite:

```
@software{crowd_certain,
  author = {Majdi, Artin},
  title = {Crowd-Certain: Crowd-Sourced Label Aggregation with Uncertainty Estimation},
  url = {https://github.com/artinmajdi/taxonomy},
  version = {1.0.0},
  year = {2023},
}
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Contact

Artin Majdi - msm2024@gmail.com

---
