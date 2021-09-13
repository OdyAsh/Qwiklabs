#  model_initial.py
```python
import tensorflow as tf
CSV_COLUMNS  = \
('ontime,dep_delay,taxiout,distance,avg_dep_delay,avg_arr_delay,' + \
 'carrier,dep_lat,dep_lon,arr_lat,arr_lon,origin,dest').split(',')
LABEL_COLUMN = 'ontime'
DEFAULTS     = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],\
                ['na'],[0.0],[0.0],[0.0],[0.0],['na'],['na']]
def read_dataset(filename, mode=tf.contrib.learn.ModeKeys.EVAL,
                 batch_size=512, num_training_epochs=10):
      # This is double indented to make a later edit simpler
      if mode == tf.contrib.learn.ModeKeys.TRAIN:
         num_epochs = num_training_epochs
      else:
         num_epochs = 1
      # could be a path to one file or a file pattern.
      input_file_names = tf.train.match_filenames_once(filename)
      filename_queue = tf.train.string_input_producer(
          input_file_names, num_epochs=num_epochs, shuffle=True)
      # Read in and parse the CSV
      reader = tf.TextLineReader()
      _, value = reader.read_up_to(
          filename_queue, num_records=batch_size)
      value_column = tf.expand_dims(value, -1)
      columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)
      features = dict(zip(CSV_COLUMNS, columns))
      label = features.pop(LABEL_COLUMN)
      return features, label
```

#  task_initial.py
```python
import argparse
import model
# import trainer.model as model
import tensorflow as tf
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--traindata',
      help='Training data file(s)',
      required=True
  )
  # parse args
  args = parser.parse_args()
  arguments = args.__dict__
  traindata = arguments.pop('traindata')
# Call read_dataset from model.py
feats, label = model.read_dataset(traindata)
# Find the average of all the labels that were read in
avg = tf.reduce_mean(label)
print(avg)
```

#  model_after_task_1.py
```python
import tensorflow.estimator as tflearn
import tensorflow.contrib.layers as tflayers
import tensorflow.contrib.metrics as tfmetrics
import numpy as np
import tensorflow as tf
CSV_COLUMNS  = \
('ontime,dep_delay,taxiout,distance,avg_dep_delay,avg_arr_delay,' + \
 'carrier,dep_lat,dep_lon,arr_lat,arr_lon,origin,dest').split(',')
LABEL_COLUMN = 'ontime'
DEFAULTS     = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],\
                ['na'],[0.0],[0.0],[0.0],[0.0],['na'],['na']]
def read_dataset(filename, mode=tf.contrib.learn.ModeKeys.EVAL,
                 batch_size=512, num_training_epochs=10):
      # the actual input function passed to TensorFlow
      def _input_fn():
      # This is double indented to make a later edit simpler
        if mode == tf.contrib.learn.ModeKeys.TRAIN:
         num_epochs = num_training_epochs
        else:
         num_epochs = 1
         # could be a path to one file or a file pattern.
        input_file_names = tf.train.match_filenames_once(filename)
        filename_queue = tf.train.string_input_producer(
          input_file_names, num_epochs=num_epochs, shuffle=True)
        # Read in and parse the CSV
        reader = tf.TextLineReader()
        _, value = reader.read_up_to(
         filename_queue, num_records=batch_size)
        value_column = tf.expand_dims(value, -1)
        columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)
        features = dict(zip(CSV_COLUMNS, columns))
        label = features.pop(LABEL_COLUMN)
        return features, label
      # return input function callback.
      return _input_fn
def get_features():
    # Using three basic inputs
    real = {
      colname : tflayers.real_valued_column(colname) \
          for colname in \
            ('dep_delay,taxiout,distance').split(',')
    }
    sparse = {}
    return real, sparse
def linear_model(output_dir):
    real, sparse = get_features()
    all = {}
    all.update(real)
    all.update(sparse)
    estimator = tflearn.LinearClassifier(model_dir=output_dir, feature_columns=all.values())
    return estimator
def serving_input_fn():
    real, sparse = get_features()
    feature_placeholders = {
      key : tf.placeholder(tf.float32, [None]) \
        for key in real.keys()
    }
    feature_placeholders.update( {
      key : tf.placeholder(tf.string, [None]) \
        for key in sparse.keys()
    } )
    features = {
      # tf.expand_dims will insert a dimension 1 into tensor shape
      # This will make the input tensor a batch of 1
      key: tf.expand_dims(tensor, -1)
         for key, tensor in feature_placeholders.items()
    }
    return tf.estimator.export.ServingInputReceiver(
      features,
      feature_placeholders)
def run_experiment(traindata,evaldata,output_dir):
  train_input = read_dataset(traindata,\
                 mode=tf.contrib.learn.ModeKeys.TRAIN)
  # Don't shuffle evaluation data
  eval_input = read_dataset(evaldata)
  train_spec = tf.estimator.TrainSpec(train_input, max_steps=1000)
  eval_spec  = tf.estimator.EvalSpec(eval_input)
  run_config = tf.estimator.RunConfig()
  run_config = run_config.replace(model_dir=output_dir)
  print('model dir {}'.format(run_config.model_dir))
  estimator = linear_model(output_dir)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
```

#  task_after_task_1.py
```python
import argparse
import model #put '#' here if you see an error
# import trainer.model as model #and remove '#' from here
import tensorflow as tf
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--traindata',
      help='Training data file(s)',
      required=True
  )
  parser.add_argument(
      '--evaldata',
      help='Training data can have wildcards',
      required=True
   )
  parser.add_argument(
      '--output_dir',
      help='Output directory',
      required=True
   )
  parser.add_argument(
      '--job-dir',
      help='required by gcloud',
      default='./junk'
   )
  # parse args
  args = parser.parse_args()
  arguments = args.__dict__
  traindata = arguments.pop('traindata')
  evaldata =  arguments.pop('evaldata')
  output_dir = arguments.pop('output_dir')
tf.logging.set_verbosity(tf.logging.INFO)
model.run_experiment(traindata,evaldata,output_dir)
```

#  model_after_task_2_part_1.py
```python
import tensorflow.estimator as tflearn
import tensorflow.contrib.layers as tflayers
import tensorflow.contrib.metrics as tfmetrics
import numpy as np
import tensorflow as tf
CSV_COLUMNS  = \
('ontime,dep_delay,taxiout,distance,avg_dep_delay,avg_arr_delay,' + \
 'carrier,dep_lat,dep_lon,arr_lat,arr_lon,origin,dest').split(',')
LABEL_COLUMN = 'ontime'
DEFAULTS     = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],\
                ['na'],[0.0],[0.0],[0.0],[0.0],['na'],['na']]
def read_dataset(filename, mode=tf.contrib.learn.ModeKeys.EVAL,
                 batch_size=512, num_training_epochs=10):
      # the actual input function passed to TensorFlow
      def _input_fn():
      # This is double indented to make a later edit simpler
        if mode == tf.contrib.learn.ModeKeys.TRAIN:
         num_epochs = num_training_epochs
        else:
         num_epochs = 1
         # could be a path to one file or a file pattern.
        input_file_names = tf.train.match_filenames_once(filename)
        filename_queue = tf.train.string_input_producer(
          input_file_names, num_epochs=num_epochs, shuffle=True)
        # Read in and parse the CSV
        reader = tf.TextLineReader()
        _, value = reader.read_up_to(
         filename_queue, num_records=batch_size)
        value_column = tf.expand_dims(value, -1)
        columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)
        features = dict(zip(CSV_COLUMNS, columns))
        label = features.pop(LABEL_COLUMN)
        return features, label
      # return input function callback.
      return _input_fn
def get_features():
    # Using three basic inputs
    real = {
      colname : tflayers.real_valued_column(colname) \
          for colname in \
            ('dep_delay,taxiout,distance').split(',')
    }
    sparse = {}
    return real, sparse
def my_rmse(labels,predictions):
    predicted_classes = predictions['probabilities'][:,1]
    custom_metric = tf.metrics.root_mean_squared_error(labels, predicted_classes,name="rmse")
    return {'rmse':custom_metric}
def linear_model(output_dir):
    real, sparse = get_features()
    all = {}
    all.update(real)
    all.update(sparse)
    estimator = tflearn.LinearClassifier(model_dir=output_dir, feature_columns=all.values())
    estimator = tf.contrib.estimator.add_metrics(estimator, my_rmse)
    return estimator
def serving_input_fn():
    real, sparse = get_features()
    feature_placeholders = {
      key : tf.placeholder(tf.float32, [None]) \
        for key in real.keys()
    }
    feature_placeholders.update( {
      key : tf.placeholder(tf.string, [None]) \
        for key in sparse.keys()
    } )
    features = {
      # tf.expand_dims will insert a dimension 1 into tensor shape
      # This will make the input tensor a batch of 1
      key: tf.expand_dims(tensor, -1)
         for key, tensor in feature_placeholders.items()
    }
    return tf.estimator.export.ServingInputReceiver(
      features,
      feature_placeholders)
def run_experiment(traindata,evaldata,output_dir):
  train_input = read_dataset(traindata,\
                 mode=tf.contrib.learn.ModeKeys.TRAIN)
  # Don't shuffle evaluation data
  eval_input = read_dataset(evaldata)
  train_spec = tf.estimator.TrainSpec(train_input, max_steps=1000)
  eval_spec  = tf.estimator.EvalSpec(eval_input)
  run_config = tf.estimator.RunConfig()
  run_config = run_config.replace(model_dir=output_dir)
  print('model dir {}'.format(run_config.model_dir))
  estimator = linear_model(output_dir)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
```

#  model_after_task_2_part_2.py
```python
import tensorflow.estimator as tflearn
import tensorflow.contrib.layers as tflayers
import tensorflow.contrib.metrics as tfmetrics
import numpy as np
import tensorflow as tf
CSV_COLUMNS  = \
('ontime,dep_delay,taxiout,distance,avg_dep_delay,avg_arr_delay,' + \
 'carrier,dep_lat,dep_lon,arr_lat,arr_lon,origin,dest').split(',')
LABEL_COLUMN = 'ontime'
DEFAULTS     = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],\
                ['na'],[0.0],[0.0],[0.0],[0.0],['na'],['na']]
def read_dataset(filename, mode=tf.contrib.learn.ModeKeys.EVAL,
                 batch_size=512, num_training_epochs=10):
      # the actual input function passed to TensorFlow
      def _input_fn():
      # This is double indented to make a later edit simpler
        if mode == tf.contrib.learn.ModeKeys.TRAIN:
         num_epochs = num_training_epochs
        else:
         num_epochs = 1
         # could be a path to one file or a file pattern.
        input_file_names = tf.train.match_filenames_once(filename)
        filename_queue = tf.train.string_input_producer(
          input_file_names, num_epochs=num_epochs, shuffle=True)
        # Read in and parse the CSV
        reader = tf.TextLineReader()
        _, value = reader.read_up_to(
         filename_queue, num_records=batch_size)
        value_column = tf.expand_dims(value, -1)
        columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)
        features = dict(zip(CSV_COLUMNS, columns))
        label = features.pop(LABEL_COLUMN)
        return features, label
      # return input function callback.
      return _input_fn
def get_features_raw():
    real = {
      colname : tflayers.real_valued_column(colname) \
          for colname in \
            ('dep_delay,taxiout,distance,avg_dep_delay,avg_arr_delay' +
             ',dep_lat,dep_lon,arr_lat,arr_lon').split(',')
    }
    sparse = {
      'carrier': tflayers.sparse_column_with_keys('carrier',
                 keys='AS,VX,F9,UA,US,WN,HA,EV,MQ,DL,OO,B6,NK,AA'.split(',')),
      'origin' : tflayers.sparse_column_with_hash_bucket('origin',
                 hash_bucket_size=1000),
      'dest'   : tflayers.sparse_column_with_hash_bucket('dest',
                 hash_bucket_size=1000)
    }
    return real, sparse
def get_features_ch7():
    # Using three basic inputs
    real = {
      colname : tflayers.real_valued_column(colname) \
          for colname in \
            ('dep_delay,taxiout,distance').split(',')
    }
    sparse = {}
    return real, sparse
def get_features_ch8():
    # Using the basic three inputs plus calculated time averages
    real = {
      colname : tflayers.real_valued_column(colname) \
          for colname in \
            ('dep_delay,taxiout,distance,avg_dep_delay,avg_arr_delay').split(',')
    }
    sparse = {}
    return real, sparse
def get_features():
    # Select the active get_feature function
    #return get_features_raw()
    #return get_features_ch7()
    return get_features_ch8()
def my_rmse(labels,predictions):
    predicted_classes = predictions['probabilities'][:,1]
    custom_metric = tf.metrics.root_mean_squared_error(labels, predicted_classes,name="rmse")
    return {'rmse':custom_metric}
def linear_model(output_dir):
    real, sparse = get_features()
    all = {}
    all.update(real)
    all.update(sparse)
    estimator = tflearn.LinearClassifier(model_dir=output_dir, feature_columns=all.values())
    estimator = tf.contrib.estimator.add_metrics(estimator, my_rmse)
    return estimator
def serving_input_fn():
    real, sparse = get_features()
    feature_placeholders = {
      key : tf.placeholder(tf.float32, [None]) \
        for key in real.keys()
    }
    feature_placeholders.update( {
      key : tf.placeholder(tf.string, [None]) \
        for key in sparse.keys()
    } )
    features = {
      # tf.expand_dims will insert a dimension 1 into tensor shape
      # This will make the input tensor a batch of 1
      key: tf.expand_dims(tensor, -1)
         for key, tensor in feature_placeholders.items()
    }
    return tf.estimator.export.ServingInputReceiver(
      features,
      feature_placeholders)
def run_experiment(traindata,evaldata,output_dir):
  train_input = read_dataset(traindata,\
                 mode=tf.contrib.learn.ModeKeys.TRAIN)
  # Don't shuffle evaluation data
  eval_input = read_dataset(evaldata)
  train_spec = tf.estimator.TrainSpec(train_input, max_steps=1000)
  eval_spec  = tf.estimator.EvalSpec(eval_input)
  run_config = tf.estimator.RunConfig()
  run_config = run_config.replace(model_dir=output_dir)
  print('model dir {}'.format(run_config.model_dir))
  estimator = linear_model(output_dir)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
```

#  model_after_task_2_part_3.py
```python
import tensorflow.estimator as tflearn
import tensorflow.contrib.layers as tflayers
import tensorflow.contrib.metrics as tfmetrics
import numpy as np
import tensorflow as tf
CSV_COLUMNS  = \
('ontime,dep_delay,taxiout,distance,avg_dep_delay,avg_arr_delay,' + \
 'carrier,dep_lat,dep_lon,arr_lat,arr_lon,origin,dest').split(',')
LABEL_COLUMN = 'ontime'
DEFAULTS     = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],\
                ['na'],[0.0],[0.0],[0.0],[0.0],['na'],['na']]
def read_dataset(filename, mode=tf.contrib.learn.ModeKeys.EVAL,
                 batch_size=512, num_training_epochs=10):
      # the actual input function passed to TensorFlow
      def _input_fn():
      # This is double indented to make a later edit simpler
        if mode == tf.contrib.learn.ModeKeys.TRAIN:
         num_epochs = num_training_epochs
        else:
         num_epochs = 1
         # could be a path to one file or a file pattern.
        input_file_names = tf.train.match_filenames_once(filename)
        filename_queue = tf.train.string_input_producer(
          input_file_names, num_epochs=num_epochs, shuffle=True)
        # Read in and parse the CSV
        reader = tf.TextLineReader()
        _, value = reader.read_up_to(
         filename_queue, num_records=batch_size)
        value_column = tf.expand_dims(value, -1)
        columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)
        features = dict(zip(CSV_COLUMNS, columns))
        label = features.pop(LABEL_COLUMN)
        return features, label
      # return input function callback.
      return _input_fn
def get_features_raw():
    real = {
      colname : tflayers.real_valued_column(colname) \
          for colname in \
            ('dep_delay,taxiout,distance,avg_dep_delay,avg_arr_delay' +
             ',dep_lat,dep_lon,arr_lat,arr_lon').split(',')
    }
    sparse = {
      'carrier': tflayers.sparse_column_with_keys('carrier',
                 keys='AS,VX,F9,UA,US,WN,HA,EV,MQ,DL,OO,B6,NK,AA'.split(',')),
      'origin' : tflayers.sparse_column_with_hash_bucket('origin',
                 hash_bucket_size=1000),
      'dest'   : tflayers.sparse_column_with_hash_bucket('dest',
                 hash_bucket_size=1000)
    }
    return real, sparse
def get_features_ch7():
    # Using three basic inputs
    real = {
      colname : tflayers.real_valued_column(colname) \
          for colname in \
            ('dep_delay,taxiout,distance').split(',')
    }
    sparse = {}
    return real, sparse
def get_features_ch8():
    # Using the basic three inputs plus calculated time averages
    real = {
      colname : tflayers.real_valued_column(colname) \
          for colname in \
            ('dep_delay,taxiout,distance,avg_dep_delay,avg_arr_delay').split(',')
    }
    sparse = {}
    return real, sparse
def get_features():
    # Select the active get_feature function
    return get_features_raw()
    #return get_features_ch7()
    #return get_features_ch8()
def my_rmse(labels,predictions):
    predicted_classes = predictions['probabilities'][:,1]
    custom_metric = tf.metrics.root_mean_squared_error(labels, predicted_classes,name="rmse")
    return {'rmse':custom_metric}
def linear_model(output_dir):
    real, sparse = get_features()
    all = {}
    all.update(real)
    all.update(sparse)
    estimator = tflearn.LinearClassifier(model_dir=output_dir, feature_columns=all.values())
    estimator = tf.contrib.estimator.add_metrics(estimator, my_rmse)
    return estimator
def serving_input_fn():
    real, sparse = get_features()
    feature_placeholders = {
      key : tf.placeholder(tf.float32, [None]) \
        for key in real.keys()
    }
    feature_placeholders.update( {
      key : tf.placeholder(tf.string, [None]) \
        for key in sparse.keys()
    } )
    features = {
      # tf.expand_dims will insert a dimension 1 into tensor shape
      # This will make the input tensor a batch of 1
      key: tf.expand_dims(tensor, -1)
         for key, tensor in feature_placeholders.items()
    }
    return tf.estimator.export.ServingInputReceiver(
      features,
      feature_placeholders)
def run_experiment(traindata,evaldata,output_dir):
  train_input = read_dataset(traindata,\
                 mode=tf.contrib.learn.ModeKeys.TRAIN)
  # Don't shuffle evaluation data
  eval_input = read_dataset(evaldata)
  train_spec = tf.estimator.TrainSpec(train_input, max_steps=1000)
  eval_spec  = tf.estimator.EvalSpec(eval_input)
  run_config = tf.estimator.RunConfig()
  run_config = run_config.replace(model_dir=output_dir)
  print('model dir {}'.format(run_config.model_dir))
  estimator = linear_model(output_dir)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
```

#  setup.py
```python
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from setuptools import find_packages
from setuptools import setup
REQUIRED_PACKAGES = [
   'tensorflow>=1.7'
]
setup(
    name='flights',
    version='0.1',
    author = 'V Lakshmanan',
    author_email = 'lak@vlakshman.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Flight Delay Prediction using TensorFlow in Cloud ML Engine (part of OReilly book Data Science on$
    requires=[]
)
```




