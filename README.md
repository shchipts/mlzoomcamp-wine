# Forecast wine quality
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-REST%20API-black?logo=flask)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

This project predicts red wine quality based on its physicochemical properties using a Random Forest Regressor, served through a Flask REST API and containerized with Docker.

---

## Table of Contents

1. [Technologies Used](#2-technologies-used)
2. [Model Development](#4-model-development)
3. [Model Performance](#5-model-performance)
4. [Model Training](#6-model-training)
5. [Prediction API](#7-prediction-api)
6. [Running Locally](#8-running-locally)
7. [Testing the API](#9-testing-the-api)
8. [Docker Deployment](#10-docker-deployment)

## 1. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Scikit-learn | Machine Learning and model evaluation |
| Flask | REST API |
| Joblib | Model serialization |
| Gunicorn | Production WSGI server |
| Waitress | Local server on Windows |
| Docker | Application containerization |

## 2. Model Development

The project uses a Random Forest Regression model to predict red wine quality.

The model development process in `notebook.ipynb` includes:

1. Exploratory Data Analysis (EDA).
2. Feature standardization.
3. Training and evaluation of multiple regression models:
   - Linear Regression
   - Ridge Regression
   - Lasso Regression
   - Random Forest Regression
4. Hyperparameter optimization for Ridge, Lasso, and Random Forest models
5. Final Random Forest model training using the selected hyperparameters.

The optimization objective was minimizing:

- **Mean Squared Error (MSE)**

The following evaluation metrics were considered:

- **MSE (Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **R² (Coefficient of Determination)**

## 3. Model Performance

The final Random Forest model achieved the following performance:

| Metric | Value |
|---|---:|
| MSE | 0.321870 |
| MAE | 0.410852 |
| R² | 0.505701 |

## 4. Model Training

The final Random Forest model is trained using `train.py`.

The training script:

1. Loads the dataset.
2. Separates features and target variable.
3. Creates a preprocessing and regression pipeline.
4. Trains the Random Forest model using the selected hyperparameters.
5. Saves the fitted pipeline and feature metadata using Joblib.

Run the training script:

```bash
python train.py
```

The trained model is saved to:

```text
models/random_forest.joblib
```

## 5. Prediction API

The prediction service is implemented using Flask in `predict.py`.

The API loads the previously trained model and provides predictions through HTTP requests.

### Available Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/predict` | POST | Generate wine quality prediction |

### Prediction Request

Send a POST request to `/predict` with a JSON object containing the required 11 input features.

Example request:

```json
{
    "fixed acidity": 7.4,
    "volatile acidity": 0.7,
    "citric acid": 0,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11,
    "total sulfur dioxide": 34,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
}
```

### Prediction Response

Example response:

```json
{
    "prediction": 5.0195715053763434
}
```

The prediction represents the estimated wine quality score.

## 6. Running Locally

### Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/mlzoomcamp-wine.git

cd mlzoomcamp-wine
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python train.py
```

### Start the prediction service

For local development:

```bash
python predict.py
```

Alternatively, on Windows, use Waitress:

```bash
waitress-serve --host=0.0.0.0 --port=5000 predict:app
```

The API will be available at:

```text
http://127.0.0.1:5000
```

## 7. Testing the API

The API can be tested using Python and the Requests library.

Run:

```bash
python test_api.py
```

Alternatively, use Postman to send POST requests to the `/predict` endpoint.

## 8. Docker Deployment

The application is containerized using Docker.

### Build the Docker image

```bash
docker build -t random-forest-api .
```

### Run the Docker container

```bash
docker run -d -p 5000:5000 --name rf-api random-forest-api
```

The API will be accessible at:

```text
http://localhost:5000
```

### Test the Dockerized API

```bash
python test_api.py
```

### Check container logs

```bash
docker logs rf-api
```

### Stop the container

```bash
docker stop rf-api
```

### Remove the container

```bash
docker rm rf-api
```
