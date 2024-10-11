import tensorflow as tf

def scale_model_weights(weights, scale_factor):
    return [scale_factor * weight for weight in weights]

def aggregate_scaled_weights(scaled_weights):
    return [tf.reduce_sum(weight, axis=0) for weight in zip(*scaled_weights)]
