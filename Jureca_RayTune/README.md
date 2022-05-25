# Simple Ray Tune script working with cifar10 dataset on JURECA-DC

Steps:
- create environment by running *create_jureca_env.sh* (or use your own env)
- run startscript *jureca_run_ray.sh*

Also includes a TensorFlow version (cifar_tune_tf.py) with TFMirroredStrategy for data-parallelism on a node-level
