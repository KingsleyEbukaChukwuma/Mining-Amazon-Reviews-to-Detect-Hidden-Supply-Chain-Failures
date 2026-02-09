# Amazon Review Supply Chain Intelligence Pipeline

## Overview

This project analyzes large-scale Amazon electronics reviews to detect hidden operational and supply chain failures using NLP and data analytics.

Customer reviews are treated as **unstructured operational signals**. Instead of sentiment analysis to detect operational failures, the pipeline identifies concrete failure categories such as:

- delivery delays  
- damaged packaging  
- defective products  
- wrong items shipped  
- missing accessories  

The system selects 400,000 reviews from ~44 million of amazon reviews using streaming ingestion, balanced sampling, rule-based NLP labeling, TF-IDF contrast modeling, and trend analytics to generate interpretable operational insights.

The result is a reproducible analytics pipeline that transforms raw text reviews into measurable supply chain intelligence.


## Why This Matters

Organizations often overlook customer review text as a structured source of operational diagnostics.

This pipeline demonstrates how:

- logistics failures appear in natural language
- defect patterns correlate with customer dissatisfaction
- operational risk can be quantified from text

The approach mirrors how a real world analytics team could monitor product quality and fulfillment performance at scale.


## Dataset

Source: Amazon Reviews 2023 (Electronics subset)

Characteristics:

- ~44 million raw reviews (streamed)
- balanced reservoir sample (400k reviews)
- time-filtered to recent years
- rating stratification to avoid bias

The pipeline is designed to operate without loading the entire dataset into memory.


## Pipeline Architecture

```
Raw JSONL Reviews
        ↓
Streaming ingestion
        ↓
Balanced reservoir sampling
        ↓
Preprocessing & feature engineering
        ↓
Rule-based issue detection
        ↓
TF-IDF contrast modeling
        ↓
Trend & co-occurrence analysis
        ↓
Composite supply chain risk index
```


## Project Structure

```
amazon-review-ops-analysis/

pipeline/
    ingestion.py
    sampling.py
    preprocessing.py

labeling/
    rule_engine.py

analysis/
    tfidf_contrast.py
    trend_analysis.py
    risk_index.py

visualization/
    networks.py

data/
outputs/

main.py
requirements.txt
README.md
```

Each module is designed with a single responsibility to mimic production analytics workflows.

## Installation

Clone the repository:

```bash
git clone <repo-url>
cd amazon-review-ops-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

Download the dataset Electronics.jsonl.gz from https://amazon-reviews-2023.github.io/ and place the dataset file in the project root:

```
Electronics.jsonl.gz
```

Run:

```bash
python main.py
```

The pipeline will:

- stream and sample reviews
- preprocess text
- label supply chain issues
- compute TF-IDF contrasts
- generate analytics datasets

Outputs are written to:

```
outputs/
```


## Core Methods

### Balanced Reservoir Sampling
Maintains rating balance while reducing computational load.

### Rule-Based NLP Labeling
Regex driven classification of operational issues using domain logic.

### TF-IDF Contrast Modeling
Identifies distinguishing language patterns for each issue type.

### Trend Analytics
Tracks issue frequency over time and rating buckets.

### Co-occurrence Modeling
Detects multi failure patterns.

### Composite Risk Index
Weighted metric representing operational failure intensity.


## Key Findings

- Defective products dominate operational complaints
- Issue frequency strongly correlates with low ratings
- Supply chain failures show stable long term patterns


## Outputs

The pipeline produces:

- labeled datasets
- TF-IDF term rankings
- issue trend metrics
- risk index summaries
- network visualizations



## Future Extensions

- machine learning classification models
- anomaly detection on issue spikes
- dashboard integration
- cross-category comparison


## Author

Kingsley Chukwuma  


## License

Educational / portfolio use.
