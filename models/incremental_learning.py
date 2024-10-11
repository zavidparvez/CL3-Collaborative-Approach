import tensorflow as tf

class IncrementalLearning:
    def __init__(self, model, threshold=0.1):
        self.model = model
        self.threshold = threshold
        self.previous_weights = None

    def check_drift(self, current_loss):
        """ Concept drift detection based on loss changes. """
        if self.previous_weights is None:
            self.previous_weights = self.model.get_weights()
            return False
        
        drift_detected = abs(self.previous_loss - current_loss) > self.threshold
        self.previous_loss = current_loss
        return drift_detected

    def update_model(self, new_data, epochs=1, steps_per_epoch=100):
        """ Perform incremental learning when concept drift is detected. """
        history = self.model.fit(new_data, epochs=epochs, steps_per_epoch=steps_per_epoch)
        return history.history['loss'][-1]
