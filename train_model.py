import tensorflow as tf
from tensorflow as keras

# MODEL DEFINITION
import tensorflow as tf

# Named function that can be fetched during training
def my_l2_normalize(x, tf=tf):
    return tf.linalg.l2_normalize(x, axis=1)

num_outputs = 32
tf.random.set_seed(1)

# USER_NN
input_user = tf.keras.Input(shape=(user_train.shape[1],))
l1 = tf.keras.layers.Dense(256, activation='relu')(input_user)
l2 = tf.keras.layers.Dense(128, activation='relu')(l1)
vu = tf.keras.layers.Dense(num_outputs)(l2)
# Use the named function in Lambda layer
vu = tf.keras.layers.Lambda(my_l2_normalize, output_shape=lambda input_shape: input_shape)(vu)

# MOVIE_NN
input_movie = tf.keras.Input(shape=(movie_train.shape[1],))
L1 = tf.keras.layers.Dense(512, activation='relu')(input_movie)
L2 = tf.keras.layers.Dense(256, activation='relu')(L1)
vm = tf.keras.layers.Dense(num_outputs)(L2)
# Use the same named function in Lambda layer
vm = tf.keras.layers.Lambda(my_l2_normalize, output_shape=lambda input_shape: input_shape)(vm)

# Computing the cosine similarity of two vectors
output = tf.keras.layers.Dot(axes=1)([vu, vm])

# Creating the Functional API model
model = tf.keras.Model([input_user, input_movie], output)
model.summary()

# MODEL COMPILATION
cost_fn = tf.keras.losses.MeanSquaredError()
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt,loss=cost_fn)

# MODEL FITTING AND TRAINING
history=model.fit([user_train, movie_train], y_train, epochs=30)

# MODEL EVALUATION 
model.evaluate([user_test, movie_test], y_test)
