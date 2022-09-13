import flwr as fl
from tensorflow import keras as ks

from utils import load_testing_data

DEFAULT_SERVER_ADDRESS = "[::]:8080"
IMG_SIZE = 160


# define an evaluation function for server side:
def get_eval_fn(model):
    X_test, y_test = load_testing_data()

    def evaluate(weights: fl.common.Weights):
        model.set_weights(weights)
        loss, accuracy = model.evaluate(X_test, y_test)
        print("****** CENTRALIZED ACCURACY: ", accuracy, " ******")
        return loss, accuracy

    return evaluate


# load our model on the server side
model = ks.Sequential([
    ks.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE)),
    ks.layers.Dense(128, activation='relu'),
    ks.layers.Dense(4)
])

model.compile(
    optimizer='adam',
    loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# define our strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.75,  # what percentage of clients we sample from in the next round
    min_available_clients=4,  # wait for 4 clients to connect before starting
    eval_fn=get_eval_fn(model),
)

if __name__ == '__main__':
    fl.server.start_server(DEFAULT_SERVER_ADDRESS, strategy=strategy, config={"num_rounds": 10})
