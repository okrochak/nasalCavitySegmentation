from functools import partial
import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tensorflow_datasets as tfds 
from tensorflow.keras.applications.resnet50 import ResNet50
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.keras import TuneReportCallback


# transform functions for data preprocessing
def train_transform(inputs):
    i = inputs["image"]
    i = tf.cast(i, tf.float32)
    i = tf.image.resize(i, size=[256,256])
    i = tf.image.random_crop(i, size=[224,224,3])
    i = tf.image.random_flip_left_right(i)
    i = tf.keras.applications.resnet50.preprocess_input(i)
    i = i / 255.0
    return (i, inputs["label"])

def val_transform(inputs):                        
    i = inputs["image"]
    i = tf.cast(i, tf.float32)
    i = tf.image.resize(i, size=[256,256])
    i = tf.image.central_crop(i, 224/256)
    i = tf.keras.applications.resnet50.preprocess_input(i)
    i = i / 255.0
    return (i, inputs["label"])   

# main train function
def train_cifar(config, data_dir=None):

    strategy = tf.distribute.MirroredStrategy()

    # load data
    train_ds, test_ds = tfds.load('cifar10', split=['train','test'], data_dir=data_dir, download=False)

    with strategy.scope():
        # prepare data and load model
        train_ds=train_ds.map(train_transform).batch(config["batch_size"])
        test_ds=test_ds.map(val_transform).batch(config["batch_size"])

        model = ResNet50(weights=None)


        # compile and run model 
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

        history = model.fit(train_ds,validation_data=test_ds, epochs=10, verbose=2, callbacks=[TuneReportCallback({"loss": "loss"})])
    
    
def main(num_samples=10, max_num_epochs=10, gpus_per_trial=4):
    ray.init(address='auto')
    
    
    config = {
        "batch_size": tune.choice([32, 64, 128, 256])    
    }
    
    result = tune.run(
        partial(train_cifar, data_dir='/p/project/raise-ctp2/tensorflow_datasets/'),
        local_dir=os.path.join(os.path.abspath(os.getcwd()), "ray_results"),
        resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=None)


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=4)
