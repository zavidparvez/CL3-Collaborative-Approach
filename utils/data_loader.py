from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(data_paths, image_size=(224, 224), batch_size=16):
    datagens = []
    for path in data_paths:
        datagen = ImageDataGenerator(preprocessing_function=None)
        train_gen = datagen.flow_from_directory(path, target_size=image_size, batch_size=batch_size, class_mode='categorical')
        datagens.append(train_gen)
    return datagens
