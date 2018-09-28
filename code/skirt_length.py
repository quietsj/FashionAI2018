import tensorflow as tf
import numpy as np
import random

is_test = True
AttrKey = 'skirt_length'
key_dict = {'skirt_length': (6, 1755), 'coat_length': (8, 2147), 'collar_design': (5, 1591), 'lapel_design': (5, 1343),
            'neck_design': (5, 1091), 'neckline_design': (10, 3147), 'pant_length': (6, 1434), 'sleeve_length': (9, 2534)}
images = 19333
train_size = 18500

image_size = 64
iterations = 100
batch_size = 250
total_step = 30000
weight_decay = 0.0003
dropout_rate = 0.5
momentum_rate = 0.9
class_num = key_dict[AttrKey][0]
test_num = key_dict[AttrKey][1]
x_path = 'oss://compi.oss-cn-shanghai-internal.aliyuncs.com/FAI2018/{0}_x.csv'.format(AttrKey)
y_path = 'oss://compi.oss-cn-shanghai-internal.aliyuncs.com/FAI2018/{0}_y.csv'.format(AttrKey)
model_save_path = 'oss://compi.oss-cn-shanghai-internal.aliyuncs.com/FAI2018/model/{0}.ckpt'.format(AttrKey)
log_save_path = 'oss://compi.oss-cn-shanghai-internal.aliyuncs.com/FAI2018/{0}_logs'.format(AttrKey)
result_path = 'oss://compi.oss-cn-shanghai-internal.aliyuncs.com/FAI2018/result/{0}.csv'.format(AttrKey)
test_x_path = 'oss://compi.oss-cn-shanghai-internal.aliyuncs.com/FAI2018/{0}_test.csv'.format(AttrKey)


def conv2d(x, out_channls, kernel_size, stride_size, name):
    return tf.layers.conv2d(x, out_channls, kernel_size, stride_size, 'same',
                            kernel_initializer=tf.keras.initializers.he_normal(),
                            bias_initializer=tf.keras.initializers.Constant(0.1), name=name)


def max_pool(x, pool_size, stride_size, name):
    return tf.layers.max_pooling2d(x, pool_size, stride_size, 'same', name=name)


def batch_norm(x):
    return tf.contrib.layers.batch_norm(x, decay=0.9, center=True, scale=True, epsilon=1e-3, is_training=train_flag,
                                        updates_collections=None)


def dense(x, units_size, name):
    return tf.layers.dense(x, units_size,
                           kernel_initializer=tf.keras.initializers.he_normal(),
                           bias_initializer=tf.keras.initializers.Constant(0.1), name=name)


def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch


def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


def data_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    x_test[:, :, :, 0] = (x_test[:, :, :, 0] - np.mean(x_test[:, :, :, 0])) / np.std(x_test[:, :, :, 0])
    return x_train, x_test


def learning_rate_schedule(step):
    if step < 4200:
        return 0.1
    elif step < 17000:
        return 0.01
    else:
        return 0.001


def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [64, 64], 8)
    return batch


def run_testing(sess, step):
    loss, acc = sess.run([cross_entropy, accuracy],
                         feed_dict={X: vari_x, y_: vari_y, keep_prob: 1.0, train_flag: False})
    summary = sess.run(merge_summary_op, feed_dict={X: vari_x, y_: vari_y, keep_prob: 1.0, 
                                                    train_flag: False, learning_rate: learning_rate_schedule(step)})
    return acc, loss, summary


if __name__ == '__main__':

    X = tf.placeholder(tf.float32, [None, image_size, image_size, 1])
    y_ = tf.placeholder(tf.float32, [None, class_num])
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)

    # build_network
    x = tf.nn.relu(batch_norm(conv2d(X, 64, (3, 3), (1, 1), 'conv1_1')))
    x = tf.nn.relu(batch_norm(conv2d(x, 64, (3, 3), (1, 1), 'conv1_2')))
    x = max_pool(x, (2, 2), (2, 2), "pool1")
    x = tf.nn.relu(batch_norm(conv2d(x, 128, (3, 3), (1, 1), 'conv2_1')))
    x = tf.nn.relu(batch_norm(conv2d(x, 128, (3, 3), (1, 1), 'conv2_2')))
    x = max_pool(x, (2, 2), (2, 2), "pool2")
    x = tf.nn.relu(batch_norm(conv2d(x, 256, (3, 3), (1, 1), 'conv3_1')))
    x = tf.nn.relu(batch_norm(conv2d(x, 256, (3, 3), (1, 1), 'conv3_2')))
    x = tf.nn.relu(batch_norm(conv2d(x, 256, (3, 3), (1, 1), 'conv3_3')))
    x = tf.nn.relu(batch_norm(conv2d(x, 256, (3, 3), (1, 1), 'conv3_4')))
    x = max_pool(x, (2, 2), (2, 2), "pool3")
    x = tf.nn.relu(batch_norm(conv2d(x, 512, (3, 3), (1, 1), 'conv4_1')))
    x = tf.nn.relu(batch_norm(conv2d(x, 512, (3, 3), (1, 1), 'conv4_2')))
    x = tf.nn.relu(batch_norm(conv2d(x, 512, (3, 3), (1, 1), 'conv4_3')))
    x = tf.nn.relu(batch_norm(conv2d(x, 512, (3, 3), (1, 1), 'conv4_4')))
    x = max_pool(x, (2, 2), (2, 2), "pool4")
    x = tf.nn.relu(batch_norm(conv2d(x, 512, (3, 3), (1, 1), 'conv5_1')))
    x = tf.nn.relu(batch_norm(conv2d(x, 512, (3, 3), (1, 1), 'conv5_2')))
    x = tf.nn.relu(batch_norm(conv2d(x, 512, (3, 3), (1, 1), 'conv5_3')))
    x = tf.nn.relu(batch_norm(conv2d(x, 512, (3, 3), (1, 1), 'conv5_4')))
    x = tf.layers.flatten(x, 'flatten')
    x = tf.nn.relu(batch_norm(dense(x, 4096, 'fc1')))
    x = tf.nn.dropout(x, keep_prob)
    x = tf.nn.relu(batch_norm(dense(x, 4096, 'fc2')))
    x = tf.nn.dropout(x, keep_prob)
    output = tf.nn.relu(batch_norm(dense(x, class_num, 'fc3')))

    # loss function: cross_entropy
    # train_step: training operation
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    train_step = tf.train.MomentumOptimizer(learning_rate, momentum_rate, use_nesterov=True).minimize(
        cross_entropy + l2 * weight_decay)
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.name_scope('number_1'):
        tf.summary.scalar('test_accracy', accuracy)
        tf.summary.scalar('loss', cross_entropy)
        tf.summary.scalar('learning_rate', learning_rate)
    # initial an saver to save model
    saver = tf.train.Saver()
    if is_test:
        with tf.Session() as sess:
            saver.restore(sess, model_save_path)
            test_x = tf.gfile.FastGFile(test_x_path, 'r').read().split('\n')
            test_x = np.array([x.split(',') for x in test_x]).astype(np.float32).reshape(test_num, image_size,
                                                                                         image_size, 1)
            test_x[:, :, :, 0] = (test_x[:, :, :, 0] - np.mean(test_x[:, :, :, 0])) / np.std(test_x[:, :, :, 0])
            result = sess.run(output, feed_dict={X: test_x, keep_prob: 1.0, train_flag: False}).astype(np.str)
            result = '\n'.join([','.join(y) for y in result])
            tf.gfile.FastGFile(result_path, 'w').write(result)
    else:
        with tf.Session() as sess:
            train_y = tf.gfile.FastGFile(y_path, 'r').read().split('\n')
            train_y = np.array([y.replace('', ',').split(',')[1:-1] for y in train_y]).astype(np.float32)
            train_x = tf.gfile.FastGFile(x_path, 'r').read().split('\n')
            train_x = np.array([x.split(',') for x in train_x]).astype(np.float32)
            features = train_x.shape[1]
            data = np.hstack((train_x, train_y))
            np.random.shuffle(data)
            train_x = data[:, :features].reshape(images, image_size, image_size, 1)
            train_y = data[:, features:]
            vari_x = train_x[train_size:]
            vari_y = train_y[train_size:]
            train_x = train_x[:train_size]
            train_y = train_y[:train_size]
            print 'train shape', train_x.shape, 'vari shape', vari_x.shape
            train_x, vari_x = data_preprocessing(train_x, vari_x)
            sess.run(tf.global_variables_initializer())
            #saver.restore(sess, model_save_path)
            merge_summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(log_save_path, sess.graph)

            batch_number = len(train_y) / batch_size
            k_batch = 0
            train_x = np.reshape(train_x, (batch_number, batch_size, image_size, image_size, 1))
            train_y = np.reshape(train_y, (batch_number, batch_size, class_num))
            for step in range(1, total_step + 1):
                lr = learning_rate_schedule(step)
                batch_x = train_x[k_batch]
                batch_y = train_y[k_batch]
                k_batch = (k_batch + 1) % batch_number
                batch_x = data_augmentation(batch_x)
                sess.run(train_step, feed_dict={X: batch_x, y_: batch_y, keep_prob: dropout_rate,
                                                learning_rate: lr, train_flag: True})
                if step % iterations == 0:
                    batch_loss, batch_acc = sess.run([cross_entropy, accuracy], feed_dict={X: batch_x, y_: batch_y,
                                                                                           keep_prob: 1.0,
                                                                                           train_flag: True})
                    val_acc, val_loss, test_summary = run_testing(sess, step)
                    summary_writer.add_summary(test_summary, step)
                    print 'step', step, 'batch loss', batch_loss, 'vari loss', val_loss, 'batch acc', batch_acc, 'vari acc', val_acc, lr, k_batch
                if step % 1000 == 0:
                    saver.save(sess, model_save_path)
                    print step
            saver.save(sess, model_save_path)


