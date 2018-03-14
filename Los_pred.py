import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

patient_data_df = pd.read_csv('Hospital_Inpatient_Discharges__SPARCS_De-Identified___2015.csv')
X_Full = patient_data_df[['Age Group', 'Gender', 'Race', 'Ethnicity', 'Type of Admission','CCS Diagnosis Code','CCS Procedure Code',  'APR DRG Code', 'APR MDC Code','APR Severity of Illness Code','APR Risk of Mortality']]
X_Full=pd.get_dummies(X_Full, columns=["Age Group"])
X_Full=pd.get_dummies(X_Full, columns=["Race"])
X_Full=pd.get_dummies(X_Full, columns=["Ethnicity"])
X_Full=pd.get_dummies(X_Full, columns=["Type of Admission"])
X_Full=pd.get_dummies(X_Full, columns=["Gender"])
X_Full=pd.get_dummies(X_Full, columns=["APR Risk of Mortality"])
list(X_Full)



Y_Full = patient_data_df[['Length of Stay']]
msk = np.random.rand(len(patient_data_df)) < 0.8
X_training = X_Full[msk]
X_testing = X_Full[~msk]

Y_training = Y_Full[msk]
Y_testing = Y_Full[~msk]

# print (Y_testing)
# print (Y_training)
# print ("types= ", Y_testing['Length of Stay'].dtypes)
# print ("types= ",Y_training['Length of Stay'].dtypes)
Y_training = Y_training.replace('120 +',120)
# print ((Y_training.loc[Y_training['Length of Stay'].isin(['120 +'])]))
Y_testing = Y_testing.replace('120 +',120)

Y_testing['Length of Stay'] = Y_testing['Length of Stay'].apply(pd.to_numeric)
Y_training['Length of Stay'] = Y_training['Length of Stay'].apply(pd.to_numeric)
# print ("types= ", Y_testing['Length of Stay'].dtypes)
# print ("types= ",Y_training['Length of Stay'].dtypes)

# print("Training testing data loaded")

X_training
#if ((Y_training.loc[Y_training['Length of Stay'].isin(['120 +'])])):

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# tf.reset_default_graph()
X_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaler = MinMaxScaler(feature_range=(0, 1))

# Scale both the training inputs and outputs
X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)

# It's very important that the training and test data are scaled with the same scaler.
X_scaled_testing = X_scaler.transform(X_testing)
Y_scaled_testing = Y_scaler.transform(Y_testing)


def neuralTraing(learning_rate = 0.001,training_epochs = 100, layer_1_nodes = 100, layer_2_nodes = 250, layer_3_nodes = 250 ,layer_4_nodes = 100):

    # Define how many inputs and outputs are in our neural network
    number_of_inputs = 31
    number_of_outputs = 1

    # Define how many neurons we want in each layer of our neural network

    # Section One: Define the layers of the neural network itself

    # Input Layer
    with tf.variable_scope('input'):
        X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))

    # Layer 1
    with tf.variable_scope('layer_1'):
        weights = tf.get_variable("weights1", shape=[number_of_inputs, layer_1_nodes],
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name="biases1", shape=[layer_1_nodes], initializer=tf.zeros_initializer())
        layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)

    # Layer 2
    with tf.variable_scope('layer_2'):
        weights = tf.get_variable("weights2", shape=[layer_1_nodes, layer_2_nodes],
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name="biases2", shape=[layer_2_nodes], initializer=tf.zeros_initializer())
        layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

    # Layer 3
    with tf.variable_scope('layer_3'):
        weights = tf.get_variable("weights3", shape=[layer_2_nodes, layer_3_nodes],
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name="biases3", shape=[layer_3_nodes], initializer=tf.zeros_initializer())
        layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

    # Layer 4
    with tf.variable_scope('layer_4'):
        weights = tf.get_variable("weights4", shape=[layer_3_nodes, layer_4_nodes],
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name="biases4", shape=[layer_4_nodes], initializer=tf.zeros_initializer())
        layer_4_output = tf.nn.relu(tf.matmul(layer_3_output, weights) + biases)

    # Output Layer
    with tf.variable_scope('output'):
        weights = tf.get_variable("weights5", shape=[layer_4_nodes, number_of_outputs],
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name="biases5", shape=[number_of_outputs], initializer=tf.zeros_initializer())
        prediction = tf.matmul(layer_4_output, weights) + biases

    # Section Two: Define the cost function of the neural network that will measure prediction accuracy during training

    with tf.variable_scope('cost'):
        Y = tf.placeholder(tf.float32, shape=(None, 1))
        cost = tf.reduce_mean(tf.squared_difference(prediction, Y))

    # Section Three: Define the optimizer function that will be run to optimize the neural network

    with tf.variable_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Create a summary operation to log the progress of the network
    with tf.variable_scope('logging'):
        tf.summary.scalar('current_cost', cost)
        summary = tf.summary.merge_all()

    rmsds= []
    arr = np.asarray(Y_testing['Length of Stay'])
    arr = np.transpose(np.asmatrix(arr))

    saver = tf.train.Saver()
    with tf.Session() as session:
        # When loading from a checkpoint, don't initialize the variables!
        session.run(tf.global_variables_initializer())
        for epoch in range(training_epochs):

            # Feed in the training data and do one step of neural network training
            session.run(optimizer, feed_dict={X: X_scaled_training, Y: Y_scaled_training})

            # Every 5 training steps, log our progress
            if epoch % 3 == 0:
                # Get the current accuracy scores by running the "cost" operation on the training and test data sets
                training_cost, training_summary = session.run([cost, summary], feed_dict={X: X_scaled_training, Y:Y_scaled_training})
                testing_cost, testing_summary = session.run([cost, summary], feed_dict={X: X_scaled_testing, Y:Y_scaled_testing})

                Y_predicted_scaled = session.run(prediction, feed_dict={X: X_scaled_testing})

                Y_predicted = Y_scaler.inverse_transform(Y_predicted_scaled)

                rmsd = np.sqrt(np.mean(np.asarray((arr - Y_predicted)) ** 2))
                Y_predicted = Y._predicted.astype(np.int64, copy=False)


                # Print the current training status to the screen
                print("Epoch: {} - Training Cost: {}  Testing Cost: {} RMSD: {}".format(epoch, training_cost, testing_cost, rmsd))

            # Training is now complete!

            # Get the final accuracy scores by running the "cost" operation on the training and test data sets
            final_training_cost = session.run(cost, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
            final_testing_cost = session.run(cost, feed_dict={X: X_scaled_testing, Y: Y_scaled_testing})

        print("Final Training cost: {}".format(final_training_cost))
        print("Final Testing cost: {}".format(final_testing_cost))

        # print(arr)
        # print(Y_predicted)
        import matplotlib.pyplot as plt

        # np.histogram(arr-Y_predicted)
        plt.hist(arr - Y_predicted)
        plt.show()


neuralTraing(learning_rate = 0.01,training_epochs = 100, layer_1_nodes = 100, layer_2_nodes = 150, layer_3_nodes = 150 ,layer_4_nodes = 100)
