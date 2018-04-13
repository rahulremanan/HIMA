import tensorflow as tf
tf.set_random_seed(1)

from keras.models import Model
from keras.layers import Input, merge
from keras.optimizers import Adam

ALPHA = 0.2  # Triplet Loss Parameter

# Source: https://github.com/davidsandberg/facenet/blob/master/src/facenet.py
def triplet_loss(x):
    anchor, positive, negative = x

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), ALPHA)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    return loss

# Builds an embedding for each example (i.e., positive, negative, anchor)
# Then calculates the triplet loss between their embedding.
# Then applies identity loss on the triplet loss value to minimize it on training.
def build_model(input_shape):
    # Standardizing the input shape order
    K.set_image_dim_ordering('th')

    positive_example = Input(shape=input_shape)
    negative_example = Input(shape=input_shape)
    anchor_example = Input(shape=input_shape)

    # Create Common network to share the weights along different examples (+/-/Anchor)
    embedding_network = create_embedding_network(input_shape)

    positive_embedding = embedding_network(positive_example)
    negative_embedding = embedding_network(negative_example)
    anchor_embedding = embedding_network(anchor_example)

    loss = merge([anchor_embedding, positive_embedding, negative_embedding],
                 mode=triplet_loss, output_shape=(1,))
    model = Model(inputs=[anchor_example, positive_example, negative_example],
                  outputs=loss)
    model.compile(loss='mean_absolute_error', optimizer=Adam())

    return model

# When fitting the model (i.e., model.fit()); use as an input [anchor_example,
# positive_example, negative_example] in that order and as an output zero.
# The reason to use the output as zero is that you are trying to minimize the 
# triplet loss as much as possible and the minimum value of the loss is zero.