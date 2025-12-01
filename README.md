# End-to-End Crash Analytics Pipeline (Bronze → Silver → Gold + ML + Monitoring)

This project is a fully containerized **data engineering + machine learning pipeline** built using Go, Python, DuckDB, Streamlit, Prometheus, and Grafana.
It ingests raw Chicago crash data, processes it through a multi-stage ETL system, stores it in a DuckDB warehouse, and exposes a Streamlit app for model inference and monitoring.

---

## 1. Pipeline Overview 

This pipeline solves the problem of **building a complete, production-style ETL + ML system** using real data from the **Chicago Data Portal**:

* **APIs Used:**

  * *Crashes dataset*
  * *Vehicles dataset*
  * *People dataset*
    URL: [https://data.cityofchicago.org](https://data.cityofchicago.org)

* **Purpose:**
  Create a fully automated system that:

  1. Fetches real crash data
  2. Cleans and standardizes it
  3. Stores it in a structured warehouse (DuckDB)
  4. Loads trained ML artifacts
  5. Predicts injury severity of traffic crashes
  6. Monitors the health of every pipeline step

* **ML Prediction Task:**
  **Binary Classification**
  Predict whether a crash is:

  * **1 = serious injury / tow**
  * **0 = non-serious**

* **Model Features Used:**

  ```
  crash_date
  weather_condition
  lighting_condition
  posted_speed_limit
  road_defect
  roadway_surface_cond
  trafficway_type
  ```

* **Streamlit App Capabilities (8 Pages):**

  * Home
  * Data Management (MinIO & DuckDB admin)
  * Data Fetcher
  * Scheduler
  * EDA
  * Reports
  * Model (predictions)
  * Monitoring (Prometheus metrics)

Training happens **only in a notebook**, and Streamlit **loads artifacts**:

* model.pkl
* threshold.txt
* labels.json

---

## 2. Component-by-Component Walkthrough

### **Extractor (Go → Bronze Layer)**

Pulls raw crashes/vehicles/people data from the Chicago API.
Stores raw JSON or JSON.gz into **MinIO** under structured folder prefixes.
Exposes Prometheus metrics such as job success/fail counts and rows processed.

### **Transformer (Python → Silver Layer)**

Combines all raw datasets by `crash_record_id`.
Performs normalization, aggregation, and merging.
Outputs a cleaned `merged.csv` for each correlation ID.
Tracks metrics like rows_in, rows_out, latency, and number of Silver objects created.

### **Cleaner (Python → Gold Layer)**

Applies final cleaning rules and upserts into **DuckDB Gold tables**.
Ensures idempotent operations using a merge/upsert pattern.
Provides Prometheus metrics including stage timings, rows processed, and errors.

### **Streamlit App**

Loads DuckDB data, model artifacts, and exposes:

* EDA
* Data preview
* Prediction
* Metrics dashboard
* ETL pipeline monitoring
  Uses Prometheus hooks to report prediction count, latency, accuracy, precision, recall.

### **Docker Compose**

Launches your entire platform:

* MinIO (object storage)
* RabbitMQ (message broker)
* Extractor, Transformer, Cleaner
* Prometheus (scraper)
* Grafana (dashboards)
* Streamlit (UI)
  Automatically creates runtime folders the first time you run the system.

### **Monitoring**

Prometheus scrapes all services:

* Extractor
* Transformer
* Cleaner
* Streamlit
* RabbitMQ
* MinIO

Grafana visualizes latency, row counts, object counts, queue depth, failures, and ML metrics.

---

## 3. Screenshots (Placeholders Required)

Replace the placeholders after you take screenshots:

```
***SCREENSHOT OF EXTRACTOR RUNNING HERE***
***SCREENSHOT OF TRANSFORMER RUNNING HERE***
***SCREENSHOT OF CLEANER RUNNING HERE***
***SCREENSHOT OF STREAMLIT HOME PAGE HERE***
***SCREENSHOT OF STREAMLIT MODEL PAGE HERE***
***SCREENSHOT OF STREAMLIT PREDICTION PAGE HERE***
***SCREENSHOT OF DUCKDB TABLES IN CLI/NOTEBOOK HERE***
***SCREENSHOT OF GRAFANA DASHBOARDS HERE***
***SCREENSHOT OF PROMETHEUS TARGET LIST HERE***
```

---

## 4. Architecture Diagram

```
Chicago API
     ↓
Extractor (Go)
     ↓
MinIO (Bronze)
     ↓
Transformer (Python)
     ↓
MinIO (Silver)
     ↓
Cleaner (Python)
     ↓
DuckDB (Gold)
     ↓
Streamlit App ----→ Predictions + Monitoring
                      ↑                 ↓
                Prometheus  ←-------- Grafana
```

---

## 5. How to Run the Pipeline (Step-by-Step)

### **1. Clone the repository**

```bash
git clone https://github.com/yourname/pipeline-project.git
cd pipeline-project
```

### **2. Create your `.env`**

Start from the sample file:

```bash
cp .env.sample .env
```

Fill in:

* MinIO credentials
* RabbitMQ URL
* Bucket names
* Chicago API tokens if needed
* Gold database paths

### **3. Ensure required folders exist**

The following folders are created automatically by Docker **if missing**:

```
minio-data/
grafana_data/
prometheus_data/
duckdb-data/
```

But you can pre-create them manually:

```bash
mkdir -p minio-data grafana_data prometheus_data duckdb-data
```

### **4. Launch everything**

```bash
docker compose up -d --build
```

### **5. Access all services**

| Service                          | URL                                              |
| -------------------------------- | ------------------------------------------------ |
| Streamlit                        | [http://localhost:8501](http://localhost:8501)   |
| Grafana                          | [http://localhost:3000](http://localhost:3000)   |
| Prometheus                       | [http://localhost:9090](http://localhost:9090)   |
| MinIO Console                    | [http://localhost:9001](http://localhost:9001)   |
| RabbitMQ Management (if enabled) | [http://localhost:15672](http://localhost:15672) |

### **6. Trigger the pipeline**

Run extractor → transformer → cleaner automatically via queue events.

---

## 6. Extra Features Added

These improvements go beyond the base project:

* **Custom ML metrics:** accuracy, precision, recall, latency
* **Custom ETL metrics:** object counts per layer, row throughput
* **8 Streamlit pages** (more than required)
* **Grafana dashboards:** 10+ visuals
* **Prometheus metrics for all services** including ML model
* **Gold database built using DuckDB instead of Postgres/MySQL**
* **Real-time inference inside Streamlit**
* **Silver object counters for MinIO**
* **Last-trained timestamp + model quality metrics**

---

## 7. Lessons Learned & Challenges

* Setting up Go + Python + Streamlit + Docker + Prometheus in one pipeline was complex
* Learned how to structure a Bronze/Silver/Gold ETL architecture
* Debugging Prometheus exporters and metrics was tricky
* MinIO bucket structure needed careful prefix design
* Would improve next time:

  * Better visuals in Grafana
  * Running more pipeline cycles for real metrics
  * Automated retraining
  * More dataset feature engineering

---

## 8. Repository Structure (Final)

This is the **final folder layout** reflecting your actual project and automatic runtime folders:

```
Pipeline/
│
├── extractor/               # Go Extractor (Bronze) – pulls crash data → MinIO
├── transformer/             # Python Transformer (Silver) – merges JSON → CSV
├── cleaner/                 # Python Cleaner (Gold) – writes to DuckDB
│
├── streamlit-app/           # Full Streamlit UI (Model, EDA, Monitoring, etc.)
│
├── docker-compose.yaml      # Launches entire platform (ETL + MinIO + Grafana)
├── .env.sample              # Template for environment variables
├── .gitignore               # Ensures no runtime data is committed
│
├── backfill.json            # Local-only config 
├── streaming.json           # Local-only config 
│
├── minio-data/              # (Auto-created) MinIO object storage volume
├── prometheus_data/         # (Auto-created) Prometheus TSDB storage
├── grafana_data/            # (Auto-created) Grafana dashboards / settings
├── duckdb-data/             # (Auto-created) Gold database files
│
└── README.md                # This documentation
```
