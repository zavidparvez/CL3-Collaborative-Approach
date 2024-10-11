import efficientnet.keras as efn
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential, Model

class TransferLearningModel:
    def __init__(self, input_shape=(224, 224, 3), num_output_classes=2):
        self.input_shape = input_shape
        self.num_output_classes = num_output_classes
    
    def build(self):
        base_model = efn.EfficientNetB7(input_shape=self.input_shape, include_top=False, weights='imagenet')
        transfer_layer = base_model.get_layer('top_activation')
        conv_model = Model(inputs=base_model.input, outputs=transfer_layer.output)

        model = Sequential()
        model.add(conv_model)
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(self.num_output_classes, activation='softmax'))
        return model
