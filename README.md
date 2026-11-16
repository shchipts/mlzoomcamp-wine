# mlzoomcamp-wine
Red Wine Quality Prediction API

1. Project Overview

This project develops and deploys a ML regression model to predict red wine quality based on its physicochemical properties.

The Random Forest Regressor model is trained using Scikit-learn and served through a REST API built with Flask.

The application is containerized using Docker to ensure reproducibility and simplify deployment.

2. Technologies Used
Python
Pandas and NumPy
Scikit-learn
Flask
Joblib
Gunicorn
Docker

4. Project Structure
text
mlzoomcamp-wine/
│
├── datasets/
│   └── winequality-red.csv
│
├── models/
│   └── random_forest.joblib
│
├── notebook.ipynb
├── eda.py
├── train.py
├── predict.py
├── test_api.py
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md

6. Model Development

The project uses a Random Forest Regression model to predict wine quality.

The model development process includes:
Exploratory Data Analysis (EDA).
Feature standardization.
Model training and evaluation: * Linear regression * Ridge, Lasso * Random Forest
Hyperparameter optimization for Ridge, Lasso and Random Forest models
Final Random Forest model training using the selected hyperparameters.

Optimization objectives were considered:

Minimizing Mean Squared Error (MSE).

The model was evaluated using:

Mean Squared Error (MSE).
Mean Absolute Error (MAE).
R² (coefficient of determination).

Final model performance:

Metric	Value

MSE	0.32187
MAE	0.410852
R²	0.505701

5. Model Training

The final Random Forest model is trained using train.py.

The training script:

Loads the dataset.
Separates features and target variable.
Creates a preprocessing and regression pipeline.
Trains the Random Forest model using the selected hyperparameters.
Saves the fitted pipeline and feature metadata using Joblib.

Run the training script:

bash
python train.py


The trained model is saved to:

text
models/random_forest.joblib

6. Prediction API

The prediction service is implemented using Flask in predict.py.

The API loads the previously trained model and provides predictions through HTTP requests.

Available Endpoints
Endpoint	Method	Description
/	GET	API information
/health	GET	Health check
/predict	POST	Generate wine quality prediction
Prediction Request

Send a POST request to /predict with a JSON object containing the required input features.

Example:

json
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

Prediction Response

Example response:

json
{
    "prediction": 5.0195715053763434
}

7. Running Locally
Install dependencies
bash
pip install -r requirements.txt

Train the model
bash
python train.py

Start the prediction service

For local development:

bash
python predict.py


Alternatively, on Windows, use Waitress:

bash
waitress-serve --host=0.0.0.0 --port=5000 predict:app


The API will be available at:

text
http://127.0.0.1:5000

8. Testing the API

The API can be tested using Python and the Requests library.

Run:

bash
python test_api.py


Alternatively, use Postman to send POST requests to the /predict endpoint.

9. Docker Deployment

The application is containerized using Docker.

Build the Docker image
bash
docker build -t random-forest-api .

Run the Docker container
bash
docker run -d -p 5000:5000 --name rf-api random-forest-api


The API will be accessible at:

text
http://localhost:5000

Stop the container
bash
docker stop rf-api
