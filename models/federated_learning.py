import tensorflow as tf
from utils.federated_utils import scale_model_weights, aggregate_scaled_weights

class FederatedLearning:
    def __init__(self, global_model, num_clients=5, optimizer=None, loss_function='categorical_crossentropy', metrics=['categorical_accuracy']):
        self.global_model = global_model
        self.num_clients = num_clients
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics

    def train(self, train_generators, validation_generators, epochs_per_round=1, num_comm_rounds=60):
        for round_num in range(num_comm_rounds):
            global_weights = self.global_model.get_weights()
            scaled_weights_per_client = []

            for client_idx in range(self.num_clients):
                # Build and compile the local model for each client
                local_model = TransferLearningModel().build()
                local_model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=self.metrics)
                local_model.set_weights(global_weights)

                # Fit the model with client's data
                local_model.fit(train_generators[client_idx], epochs=epochs_per_round, steps_per_epoch=100, validation_data=validation_generators[client_idx], validation_steps=20)

                # Scale and collect client weights
                scaled_weights = scale_model_weights(local_model.get_weights(), 1 / self.num_clients)
                scaled_weights_per_client.append(scaled_weights)

            # Aggregate weights and update the global model
            averaged_weights = aggregate_scaled_weights(scaled_weights_per_client)
            self.global_model.set_weights(averaged_weights)
