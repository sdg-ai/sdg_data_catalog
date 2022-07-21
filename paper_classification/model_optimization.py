import os
os.system('pip install -q tf-models-official==2.7.0')
os.system("pip install -q -U 'tensorflow-text==2.8.*")
os.system("pip install tensorflow-hub")
os.system("pip install pyyaml h5py")

import shutil
import tensorflow as tf
from tensorflow import keras 

import matplotlib.pyplot as plt
import pickle
import pandas as pd 
from sklearn.metrics import accuracy_score
import numpy as np
from utils import *
import argparse
import tensorflow_datasets as tfds 
import tensorflow_hub as hub
import tensorflow_text as text




# __________________________________ Training Params __________________________________
meta_step_size = 0.3
learning_rate = 0.001
inner_batch_size = 50
eval_batch_size = 50
meta_iters = 1000
eval_iters = 5
inner_iters = 10
eval_interval = 1
train_shots = 50
shots = 5
classes = 2
optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

# __________________________________Model__________________________________
bert_model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8'
tfhub_handle_encoder = map_name_to_handle[bert_model_name]
tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]


bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
def build_classifier_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(256, activation='relu', name='Dense1')(net)
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(128, activation='relu', name='Dense2')(net)
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(64, activation='relu', name='Dense3')(net)
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(2, activation='softmax', name='classifier')(net)
    return tf.keras.Model(text_input, net)


# __________________________________ Training __________________________________

if __name__=='__main__':

    """
    Serialization of few shot learning takes 4 inputs:
    - text_path_training : default path to access the directory that contains abstract of the annotations
    - labels_path_training: default path to access the directory that contains SDG Goals annotations
    - text_path_test : default path to access the directory that contains abstract of the annotations
    - labels_path_test: default path to access the directory that contains SDG Goals annotations
    """

    # ___________________________________ Data Parser ___________________________________
    OR_PATH = os.getcwd()
    DATA_DIR = OR_PATH + os.path.sep + 'binary_class_data' + os.path.sep
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text_path_training", 
        help= 'Path to access the directory with abstracts',
        default=DATA_DIR+'X_train.pkl', 
        type=str, 
        required=True
    )
    parser.add_argument(
        "--labels_path_training", 
        help= 'Path to access the directory with annotations',
        default=DATA_DIR+'y_train.pkl', 
        type=str, 
        required=True
    )
    
    parser.add_argument(
        "--text_path_test", 
        help= 'Path to access the directory with abstracts',
        default=DATA_DIR+'X_test.pkl', 
        type=str, 
        required=True
    )

    parser.add_argument(
        "--labels_path_test", 
        help= 'Path to access the directory with abstracts',
        default=DATA_DIR+'y_test.pkl', 
        type=str, 
        required=True
    )

    prs = parser.parse_args()

    train_dataset = Dataset(prs.text_path_training, prs.text_path_training,training=True)
    test_dataset = Dataset(prs.text_path_test, prs.labels_path_test, training=False)
    text = pd.read_pickle(DATA_DIR+'X_train.pkl')
    labels = pd.read_pickle(DATA_DIR+'y_train.pkl')
    text_test = pd.read_pickle(DATA_DIR+'X_test.pkl')
    labels_test = pd.read_pickle(DATA_DIR+'y_test.pkl')
    model = build_classifier_model()
    bert_raw_result = model(tf.constant(text))
    dataset, test_images, test_labels = test_dataset.get_mini_dataset(
        batch_size=2, 
        repetitions=2, 
        shots=2, 
        num_classes=2, 
        split=True)
    training = []
    testing = []
    score_history = []
    trial_num = []
    best_accuracy = 0.5
    for meta_iter in range(meta_iters):
        frac_done = meta_iter / meta_iters
        cur_meta_step_size = (1 - frac_done) * meta_step_size
        # Temporarily save the weights from the model.
        old_vars = model.get_weights()
        # Get a sample from the full dataset.
        mini_dataset = train_dataset.get_mini_dataset(
            inner_batch_size, inner_iters, train_shots, classes
        )
        for images, labels in mini_dataset:
            with tf.GradientTape() as tape:
                preds = model(images)
                loss = keras.losses.sparse_categorical_crossentropy(labels, preds)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        new_vars = model.get_weights()
        # Perform SGD for the meta step.
        for var in range(len(new_vars)):
        # for some reason the 71'st element has a boolean layer so we skip updating
        # it to avoid errors. 
            if var !=71:
                new_vars[var] = old_vars[var] + (
                    (new_vars[var] - old_vars[var]) * cur_meta_step_size
                )
        # After the meta-learning step, reload the newly-trained weights into the model.
        model.set_weights(new_vars)
        # Evaluation loop
        if meta_iter % eval_interval == 0:
            accuracies = []
            for dataset in (train_dataset, test_dataset):
                # Sample a mini dataset from the full dataset.
                train_set, test_images, test_labels = dataset.get_mini_dataset(
                    eval_batch_size, eval_iters, shots, classes, split=True
                )
                old_vars = model.get_weights()
                # Train on the samples and get the resulting accuracies.
                for images, labels in train_set:
                    with tf.GradientTape() as tape:
                        preds = model(images)
                        loss = keras.losses.sparse_categorical_crossentropy(labels, preds)
                    grads = tape.gradient(loss, model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))
                test_preds = model.predict(test_images)
                test_preds = tf.argmax(test_preds, axis=1).numpy()
                num_correct = (test_preds == test_labels).sum()
                # Reset the weights after getting the evaluation accuracies.
                model.set_weights(old_vars)
                accuracies.append(num_correct / classes)
            training.append(accuracies[0])
            testing.append(accuracies[1])
            if meta_iter % 10 == 0:
                print(
                    "batch %d: train=%f test=%f" % (meta_iter, accuracies[0], accuracies[1])
                )
                scores = accuracy_score(labels_test, model.predict(text_test).argmax(axis=-1))
                if scores >= best_accuracy:
                    model.save(OR_PATH+'/model_best_accuracy.h5')
                #print("Test scores Accuracy: ",scores)
                score_history.append(scores)
                trial_num.append(meta_iter)
            print('Model training completed')
            print(f'Best model accuracy : {max(score_history)}')
