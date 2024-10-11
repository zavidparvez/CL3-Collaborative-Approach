class IncrementalLearning:
    def __init__(self, model, drift_threshold=0.1):
        self.model = model
        self.drift_threshold = drift_threshold
        self.previous_loss = None

    def detect_drift(self, current_loss):
        if self.previous_loss is None:
            self.previous_loss = current_loss
            return False
        return abs(current_loss - self.previous_loss) > self.drift_threshold

    def update_model(self, data_generator, steps_per_epoch=100):
        history = self.model.fit(data_generator, epochs=1, steps_per_epoch=steps_per_epoch)
        current_loss = history.history['loss'][-1]
        if self.detect_drift(current_loss):
            print("Concept drift detected! Updating model...")
        self.previous_loss = current_loss
