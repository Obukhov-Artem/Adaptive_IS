from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras import backend as K

import tensorflow as tf

from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib

import numpy as np

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


def main():
    X = [i for i in range (1,1000)]
    Y = [i+10 for i in range (1,1000)]
    X = np.array(X)
    Y = np.array(Y)

    model = Sequential()
    model.add(Dense(80, input_dim=1, activation='linear'))
    model.add(Dense(80,  activation='linear'))
    model.add(Dense(100, activation='linear'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer="adam")
    model.fit(X, Y, batch_size=10, nb_epoch=4)
    model.summary()

    frozen_graph = freeze_session(K.get_session(),
                                  output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, "some_directory", "my_model.pb", as_text=False)

if __name__=='__main__':
    main()
