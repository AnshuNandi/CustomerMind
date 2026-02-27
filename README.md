
# CustomerMind

An intelligent machine learning system that segments and predicts customer personality types. Built with FastAPI, scikit-learn, and MongoDB.

**🚀 [Live Demo](https://customermind.onrender.com)** | **📊 [GitHub](https://github.com/AnshuNandi/CustomerMind)**

## Overview

CustomerMind is a complete ML pipeline that identifies distinct customer segments and predicts which segment new customers belong to. The system learns customer personalities from behavioral and demographic data through unsupervised clustering and supervised classification.

**Dataset**: Customer marketing data with ~2,240 records and 21 features (demographics, spending patterns, purchase channels)

## Tech Stack

- **Backend**: FastAPI
- **ML**: Scikit-learn (K-Means, Logistic Regression), XGBoost
- **Data**: Pandas, NumPy
- **Database**: MongoDB Atlas
- **Deployment**: Docker, Render.com
- **CI/CD**: GitHub Actions

## What I Built

- **ML Pipeline**: Data validation → preprocessing → PCA dimensionality reduction → K-Means clustering → Logistic Regression classification
- **REST API**: FastAPI backend with training and prediction endpoints
- **Web Interface**: HTML frontend for interactive predictions
- **Model Optimization**: GridSearchCV hyperparameter tuning for both models
- **Data Pipeline**: MongoDB integration with custom data ingestion and validation
- **Notebook Analysis**: EDA, feature engineering, and model evaluation in Jupyter notebooks
- **CI/CD**: Automated deployment with GitHub Actions
- **Docker**: Containerized application for reproducible deployments  

## Results

- **Clustering**: K-Means identified 4 distinct customer segments
- **Classification**: Logistic Regression trained to predict segment membership
- **Evaluation**: Model validation on held-out test data

## Project Structure

```
src/
├── components/         # ML pipeline stages
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── data_transformation.py
│   ├── data_clustering.py
│   ├── model_trainer.py
│   └── model_evaluation.py
├── pipeline/          # Training and prediction pipelines
├── configuration/     # MongoDB and AWS connections
└── utils/            # Helper functions

notebooks/            # EDA and analysis
├── EDA.ipynb
├── Feature_engineering_and_clustering.ipynb
└── Feature_Selection_and_classification.ipynb

app.py               # FastAPI application
templates/           # HTML frontend
```

## Running Locally

```bash
# Setup
conda create -p venv python=3.8
conda activate venv/
pip install -r requirements.txt

# Configure MongoDB connection
export MONGODB_URL=<your_mongodb_url>

# Run
python app.py
```

Access the app at `http://localhost:5000`

## Deployment

Deployed on **Render.com** with Docker containerization. Application automatically deploys on every GitHub push.

**🌐 Live Application**: [https://customermind.onrender.com](https://customermind.onrender.com)

### Deployment Stack
- **Container**: Docker
- **Platform**: Render.com (free tier)
- **Database**: MongoDB Atlas (free tier)
- **Environment**: Python 3.8, FastAPI, Uvicorn

## Machine Learning Models

### K-Means Clustering
- Unsupervised algorithm for identifying customer personality clusters
- Optimized via GridSearchCV for optimal cluster count
- Dimensionality reduction with PCA prior to clustering

### Logistic Regression
- Supervised classifier for customer segment prediction
- Binary cross-entropy loss optimization
- Feature scaling applied for optimal convergence

## Custom Utilities

## Implementation Details

- Custom logger and exception handling throughout the codebase
- Pydantic-based data validation
- YAML configuration files for model parameters
- GridSearchCV hyperparameter optimization for both clustering and classification models

## API Endpoints

- `GET /` - Web interface for predictions  
- `POST /` - Submit customer data for prediction
- `GET /train` - Trigger model training pipeline




