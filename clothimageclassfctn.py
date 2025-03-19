# IMPORT LIBRARIES..Import and configure logging
import logging
import google.cloud.logging as cloud_logging
from google.cloud.logging.handlers import CloudLoggingHandler
from google.cloud.logging_v2.handlers import setup_logging

cloud_logger = logging.getLogger('cloudLogger')
cloud_logger.setLevel(logging.INFO)
cloud_logger.addHandler(CloudLoggingHandler(cloud_logging.Client()))
cloud_logger.addHandler(logging.StreamHandler())
# Import TensorFlow
import tensorflow as tf
# Import numpy
import numpy as np
# Import tensorflow_datasets
import tensorflow_datasets as tfds

#LOAD N PREPROCESS DATA
# Define, load and configure data
(ds_train, ds_test), info = tfds.load('fashion_mnist', split=['train', 'test'], with_info=True, as_supervised=True)
#supervsd true gives tuple (input, label)
# Values before normalization
image_batch, labels_batch = next(iter(ds_train))
print("Before normalization ->", np.min(image_batch[0]), np.max(image_batch[0]))
#wiil be 0 and say 227
#DATA PREPROCESSING
# Define batch size
BATCH_SIZE = 32
# Normalize and batch process the dataset
#map() normalizes it
ds_train = ds_train.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y)).batch(BATCH_SIZE)
ds_test = ds_test.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y)).batch(BATCH_SIZE)
#pixel values r of int8 type convert it to float using cast
#batch converts it to
# Examine the min and max values of the batch after normalization
image_batch, labels_batch = next(iter(ds_train))
print("After normalization ->", np.min(image_batch[0]), np.max(image_batch[0]))

#DESIGN, COMPILE, TRAIN THE MODEL
# Define the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(64, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
#flattten converts ip 78*78 square matrix into a 1D
#the rule of thumb that the first layer in your network should be the same shape as your data. Right now, the input images are of shape 28x28, and 28 layers of 28 neurons would be infeasible. So, it makes more sense to flatten that 28,28 into a 784x1.
Another rule of thumb -- the number of neurons in the last layer should match the number of classes you are classifying for. In this case, it's the digits 0-9,for each category if dress type, so there are 10 of them, and hence you should have 10 neurons in your final layer.
#Softmax picks the biggest one.the output of the last layer looks like [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05], it returns [0,0,0,0,1,0,0,0,0].
#An Optimizer is an algorithm that modifies the attributes of the neural network like weights and learning rate. This helps in reducing the loss and improving accuracy.
#metrics= parameter. This allows TensorFlow to report on the accuracy of the training after each epoch by checking the predicted results against the known answers(labels). It basically reports back on how effectively the training is progressing.

# Compile the model
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

model.fit(ds_train, epochs=5)
#above gives 5 epoch with accuracy

#Evaluate model performance on unseen data
cloud_logger.info(model.evaluate(ds_test))
#gives new acc

# Save and load the model
#An entire model can be saved in two different file formats (SavedModel and Keras). The TensorFlow SavedModel format is the default file format in TF2.x. However, models can be saved in Keras format. You will learn more on saving models in the two file formats.

# Save the entire model as a Keras model using .keras format
model.save('saved_model.keras')

# Load the model using custom_objects to handle the custom activation function
new_model = tf.keras.models.load_model('saved_model.keras', custom_objects={'softmax_v2': tf.keras.activations.softmax})

# Summary of loaded SavedModel
new_model.summary()

# Save the entire model to a keras file.
model.save('my_model.keras')

# Recreate the exact same model, including its weights and the optimizer
new_model_keras = tf.keras.models.load_model('my_model.keras', custom_objects={'softmax_v2': tf.keras.activations.softmax})

# Summary of loaded keras model
new_model_keras.summary()

#if u want to stop training  at asay 82% use callback
#CALLBACKS
# Import and configure logging
import logging
import google.cloud.logging as cloud_logging
from google.cloud.logging.handlers import CloudLoggingHandler
from google.cloud.logging_v2.handlers import setup_logging
exp_logger = logging.getLogger('expLogger')
exp_logger.setLevel(logging.INFO)
exp_logger.addHandler(CloudLoggingHandler(cloud_logging.Client(), name="callback"))

# Import tensorflow_datasets
import tensorflow_datasets as tfds
# Import numpy
import numpy as np
# Import TensorFlow
import tensorflow as tf
# Define Callback
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('sparse_categorical_accuracy')>0.82):
      exp_logger.info("\nReached 84% accuracy so cancelling training!")
      self.model.stop_training = True
callbacks = myCallback()
# Define, load and configure data
(ds_train, ds_test), info = tfds.load('fashion_mnist', split=['train', 'test'], with_info=True, as_supervised=True)
# Define batch size
BATCH_SIZE = 32
# Normalizing and batch processing of data
ds_train = ds_train.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y)).batch(BATCH_SIZE)
ds_test = ds_test.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y)).batch(BATCH_SIZE)
# Define the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(64, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
# Compile data
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
model.fit(ds_train, epochs=5, callbacks=[callbacks])
