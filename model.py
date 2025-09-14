#
# #If u made a model from scratch then use it here and save it as a pkl file
#
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from huggingface_hub import hf_hub_download
# import pickle
#
# model_path = hf_hub_download(
#     repo_id="jack-perlo/Lenet5-Mnist",   # repo name
#     filename="lenet5_fp32_mnist.keras"   # file inside repo
# )
#
# model = tf.keras.models.load_model(model_path)
# model.save('main_save.h5') #conversion to h5
#
# model = load_model('main_save.h5')
#
# with open('main_pickle_save.pkl', 'wb') as f:
#     pickle.dump(model, f)
#


""" 28 x 28 LeNet-5 Model"""

# model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pickle

# ----------------------------
# 1. Load MNIST dataset
# ----------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)  # (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)    # (10000, 28, 28, 1)

# ----------------------------
# 2. Define LeNet-5 architecture
# ----------------------------
def build_lenet5(input_shape=(28, 28, 1), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(6, kernel_size=5, strides=1, activation="tanh", input_shape=input_shape, padding="same"),
        layers.AveragePooling2D(pool_size=2, strides=2),

        layers.Conv2D(16, kernel_size=5, strides=1, activation="tanh"),
        layers.AveragePooling2D(pool_size=2, strides=2),

        layers.Conv2D(120, kernel_size=5, strides=1, activation="tanh"),
        layers.Flatten(),

        layers.Dense(84, activation="tanh"),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

model = build_lenet5()

# ----------------------------
# 3. Compile model
# ----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ----------------------------
# 4. Train model
# ----------------------------
model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=5,
    validation_split=0.1,
    verbose=2
)

# ----------------------------
# 5. Evaluate
# ----------------------------
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {acc:.4f}")

# ----------------------------
# 6. Save model
# ----------------------------
model.save("main_save.h5")

# ----------------------------
# 7. Pickle for Flask
# ----------------------------
with open("main_pickle_save.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as main_save.h5 & main_pickle_save.pkl")
