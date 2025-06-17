import kfp
import kfp.dsl as dsl
from kfp.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Metrics,
    Artifact,
)
from kfp import kubernetes


@component(base_image='python:3.9', packages_to_install=["openpyxl", "dask[dataframe]==2024.8.0", "s3fs==2025.2.0", "pandas==2.2.3"])
def fetch_data(
    data_location: str,
    dataset: Output[Dataset],
):
    """
    Fetches data from URL
    """

    import pandas as pd
    import yaml
    import os

    data = pd.read_excel(data_location)

    dataset.path += ".csv"
    dataset.metadata = {"origin": data_location}
    data.to_csv(dataset.path, index=False, header=True)


@component(base_image="tensorflow/tensorflow:2.15.0", packages_to_install=[ "pandas==2.2.3", "scikit-learn==1.6.1"])
def train_model(
    data: Input[Dataset],
    trained_model: Output[Artifact],
    metrics: Output[Metrics],
):
    """
    Trains a simple dense TensorFlow model using a single input dataset.
    Performs train/val splitting internally.
    """
    import pandas as pd
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from keras.layers import Dense, Concatenate, Input as KerasInput
    from tensorflow.keras.models import Model as KerasModel
    import numpy as np

    # Reproducibility
    SEED = 42
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    # Load data
    df = pd.read_csv(data.path)

    # Drop rows with missing Demand
    df = df.dropna(subset=["Demand"])

    # Separate features and label
    y = df[["Demand"]]
    X = df.drop(columns=["Demand"])

    # Convert non-numeric columns
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.factorize(X[col])[0]

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # Build model
    inputs = [KerasInput(shape=(1,), name=name) for name in X.columns]
    x = Concatenate(name="input")(inputs)
    x = Dense(64, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(1)(x)

    model = KerasModel(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Prepare datasets
    train_features = [X_train[[name]].to_numpy() for name in X.columns]
    val_features = [X_val[[name]].to_numpy() for name in X.columns]

    train_dataset = tf.data.Dataset.zip((
        tf.data.Dataset.zip(tuple(tf.data.Dataset.from_tensor_slices(f) for f in train_features)),
        tf.data.Dataset.from_tensor_slices(y_train)
    )).shuffle(len(y_train), seed=SEED).batch(32)

    val_dataset = tf.data.Dataset.zip((
        tf.data.Dataset.zip(tuple(tf.data.Dataset.from_tensor_slices(f) for f in val_features)),
        tf.data.Dataset.from_tensor_slices(y_val)
    )).batch(32)

    # Train the model
    model.fit(train_dataset, validation_data=val_dataset, epochs=1, verbose=True)

    # Evaluate the model
    val_inputs = {name: X_val[[name]].to_numpy() for name in X.columns}
    y_pred = model.predict(val_inputs).flatten()
    y_true = y_val.values.flatten()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    metrics.log_metric("MAE", float(mae))
    metrics.log_metric("RMSE", float(rmse))

    # Save model
    model.save(trained_model.path + ".keras")

@dsl.pipeline(
  name='simple-training-pipeline',
)
def training_pipeline(data_location: str):

    fetch_task = fetch_data(data_location = data_location)

    train_model(data = fetch_task.outputs["dataset"])


if __name__ == '__main__':
    COMPILE=False

    if COMPILE:
        kfp.compiler.Compiler().compile(training_pipeline, 'simple-training-pipeline.yaml')
    else:
        metadata = {
            "data_location": 'https://raw.githubusercontent.com/RHRolun/simple-training-pipeline/main/data/demand_qty_item_loc.xlsx'
        }
            
        namespace_file_path =\
            '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
        with open(namespace_file_path, 'r') as namespace_file:
            namespace = namespace_file.read()
    
        kubeflow_endpoint =\
            f'https://ds-pipeline-dspa.{namespace}.svc:8443'
    
        sa_token_file_path = '/var/run/secrets/kubernetes.io/serviceaccount/token'
        with open(sa_token_file_path, 'r') as token_file:
            bearer_token = token_file.read()
    
        ssl_ca_cert =\
            '/var/run/secrets/kubernetes.io/serviceaccount/service-ca.crt'
    
        print(f'Connecting to Data Science Pipelines: {kubeflow_endpoint}')
        client = kfp.Client(
            host=kubeflow_endpoint,
            existing_token=bearer_token,
            ssl_ca_cert=ssl_ca_cert
        )
    
        client.create_run_from_pipeline_func(
            training_pipeline,
            arguments=metadata,
            experiment_name="simple-training-pipeline",
            enable_caching=False
        )
