import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Define the ImageDataGenerators for training and testing data
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

# Function to create a model


def create_model(conv_layers, dense_layers, dropout_rate):
    model = Sequential()
    for i in range(conv_layers):
        if i == 0:
            model.add(Conv2D(32*(2**i), (3, 3),
                      activation='relu', input_shape=(64, 64, 3)))
        else:
            model.add(Conv2D(32*(2**i), (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout_rate))
    model.add(Flatten())
    for i in range(dense_layers):
        model.add(Dense(128*(2**i), activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    return model


# Define a list of models
models = []
for conv_layers in [1, 2]:
    for dense_layers in [1, 2]:
        for dropout_rate in [0.25, 0.5]:
            models.append(create_model(
                conv_layers, dense_layers, dropout_rate))

# Train each model and keep track of their performance
best_model = None
best_val_loss = float('inf')
best_val_acc = 0
best_epoch_count = 0
for i, model in enumerate(models):
    print(f'Training model {i+1}')
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=3, verbose=1)
    history = model.fit(train_generator, epochs=10, steps_per_epoch=50,
                        validation_data=test_generator, callbacks=[early_stopping, reduce_lr])
    val_loss = min(history.history['val_loss'])
    val_acc = max(history.history['val_accuracy'])
    epoch_count = sum(
        acc == val_acc for acc in history.history['val_accuracy'])

    if val_loss < best_val_loss and val_acc >= best_val_acc and epoch_count >= best_epoch_count:
        best_val_loss = val_loss
        best_val_acc = val_acc
        best_epoch_count = epoch_count
        best_model = model

# Save the best model
best_model.save('pcos_detection_model.h5')
