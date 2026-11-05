import numpy as np
import pandas as pd
import keras
import ml_edu.experiment
import ml_edu.results

# 1. Load and Select Data
movie_dataset = pd.read_csv("Untitled document-Movies.csv")
training_df = movie_dataset[
    ['PRODUCTION_BUDGET', 'MARKETING_SPEND', 'TOTAL_BOX_OFFICE', 'STUDIO', 'GENRE', 'IMDB_RATING']].copy()

# 2. Preprocess Data (Scale down massive currency numbers to "Millions" for neural network stability)
training_df['BUDGET_MILLIONS'] = training_df['PRODUCTION_BUDGET'] / 1000000
training_df['MARKETING_MILLIONS'] = training_df['MARKETING_SPEND'] / 1000000
training_df['BOX_OFFICE_MILLIONS'] = training_df['TOTAL_BOX_OFFICE'] / 1000000


# 3. Define Machine Learning Functions
def create_model(settings, metrics):
    inputs = {name: keras.Input(shape=(1,), name=name) for name in settings.input_features}
    concatenated_inputs = keras.layers.Concatenate()(list(inputs.values()))
    outputs = keras.layers.Dense(units=1)(concatenated_inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=settings.learning_rate),
                  loss="mean_squared_error",
                  metrics=metrics)
    return model


def train_model(experiment_name, model, dataset, label_name, settings):
    features = {name: dataset[name].values for name in settings.input_features}
    label = dataset[label_name].values

    history = model.fit(x=features, y=label, batch_size=settings.batch_size, epochs=settings.number_epochs, verbose=0)

    return ml_edu.experiment.Experiment(
        name=experiment_name,
        settings=settings,
        model=model,
        epochs=history.epoch,
        metrics_history=pd.DataFrame(history.history),
    )


# 4. Configure and Train the Final Model
settings = ml_edu.experiment.ExperimentSettings(
    learning_rate=0.1,
    number_epochs=20,
    batch_size=50,
    input_features=['BUDGET_MILLIONS', 'MARKETING_MILLIONS']
)

model = create_model(settings, [keras.metrics.RootMeanSquaredError(name='rmse')])
experiment = train_model('movie_revenue_predictor', model, training_df, 'BOX_OFFICE_MILLIONS', settings)

# 5. Visualize Model Performance
ml_edu.results.plot_experiment_metrics(experiment, ['rmse'])
ml_edu.results.plot_model_predictions(experiment, training_df, 'BOX_OFFICE_MILLIONS')


# 6. Make and Format Predictions
def predict_revenue(model, df, features, label, batch_size=10):
    batch = df.sample(n=batch_size).copy().reset_index(drop=True)
    predicted_values = model.predict_on_batch(x={name: batch[name].values for name in features})

    data = {"PREDICTED_REVENUE": [], "OBSERVED_REVENUE": [], "ERROR_MARGIN": []}
    for name in features:
        data[name] = []

    for i in range(batch_size):
        predicted = predicted_values[i][0]
        observed = batch.at[i, label]
        data["PREDICTED_REVENUE"].append(f"${predicted:.2f}M")
        data["OBSERVED_REVENUE"].append(f"${observed:.2f}M")
        data["ERROR_MARGIN"].append(f"${abs(observed - predicted):.2f}M")
        for name in features:
            data[name].append(batch.at[i, name])

    return pd.DataFrame(data)


print("\n--- SAMPLE REVENUE PREDICTIONS ---")
predictions_df = predict_revenue(experiment.model, training_df, settings.input_features, 'BOX_OFFICE_MILLIONS')
print(predictions_df.to_string())

# Add this to the very bottom of main.py and run it once
experiment.model.save('box_office_predictor.keras')

weights, bias = experiment.model.layers[-1].get_weights()

print("\nWeights:")
print(weights)

print("\nBias:")
print(bias)


print(experiment.metrics_history)