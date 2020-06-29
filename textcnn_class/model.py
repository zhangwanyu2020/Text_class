import tensorflow as tf

def text_cnn(sequence_length, vocab_size, embedding_size, num_classes, num_filters, filter_sizes,
             regularizers_lambda=0.0, dropout_rate=0.1, embedding_matrix=None):
    inputs = tf.keras.Input(shape=(sequence_length,), name='input_x')

    if embedding_matrix is None:
        embedding_initer = tf.keras.initializers.RandomUniform(minval=-1, maxval=1)
        embedding = tf.keras.layers.Embedding(vocab_size,
                                              embedding_size,
                                              embeddings_initializer=embedding_initer,
                                              input_length=sequence_length,
                                              name='embedding')(inputs)
    else:
        embedding = tf.keras.layers.Embedding(vocab_size,
                                              embedding_size,
                                              weights=[embedding_matrix],
                                              trainable=False,
                                              name='embedding')(inputs)
    embedding = tf.keras.layers.Reshape((sequence_length, embedding_size, 1), name='add_channel')(embedding)

    pooled_outputs = []
    # filter_sizes=[2,3,4]
    filter_sizes = [int(i) for i in filter_sizes.split(',')]
    for filter_size in filter_sizes:
        conv = tf.keras.layers.Conv2D(filters=num_filters,
                                      kernel_size=[filter_size, embedding_size],
                                      strides=(1, 1),
                                      padding='valid',
                                      data_format='channels_last',
                                      activation='relu',
                                      kernel_initializer='glorot_normal',
                                      bias_initializer=tf.keras.initializers.constant(0.1),
                                      name='convolution_{:d}'.format(filter_size)
                                      )(embedding)
        pool = tf.keras.layers.MaxPool2D(pool_size=[sequence_length - filter_size + 1, 1],
                                         strides=(1, 1),
                                         padding='valid',
                                         data_format='channels_last',
                                         name='max_pooling_{:d}'.format(filter_size)
                                         )(conv)
        pooled_outputs.append(pool)

    pool_outputs = tf.keras.layers.concatenate(pooled_outputs, axis=-1, name='concatenate')
    pool_outputs = tf.keras.layers.Flatten(data_format='channels_last', name='flatten')(pool_outputs)
    pool_outputs = tf.keras.layers.Dropout(dropout_rate, name='dropout')(pool_outputs)

    outputs = tf.keras.layers.Dense(num_classes,
                                    activation='softmax',
                                    kernel_initializer='glorot_normal',
                                    bias_initializer=tf.keras.initializers.constant(0.1),
                                    kernel_regularizer=tf.keras.regularizers.l2(regularizers_lambda),
                                    bias_regularizer=tf.keras.regularizers.l2(regularizers_lambda),
                                    name='dense'
                                    )(pool_outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model