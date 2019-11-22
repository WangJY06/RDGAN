import numpy as np
import tensorflow as tf
import tensorlayer as tl
import argparse, glob, os, random, time
from os.path import join
from model import *

PATCH_NUM = 0
PATCH_SIZE = 256
CROP_SIZE = 128

def get_arguments():
    parser = argparse.ArgumentParser(description='rdnet')
    parser.add_argument("--batch-size", type=int, default=16, help='')
    parser.add_argument("--learning-rate", type=float, default=1e-4, help='')
    parser.add_argument("--epoch-num", type=int, default=100, help='')
    parser.add_argument("--start-epoch", type=int, default=0, help='')
    parser.add_argument("--print-interval", type=int, default=10, help='')
    parser.add_argument("--data-dir", type=str, default='./data', help='')
    parser.add_argument("--ckpt-dir", type=str, default='./rd_model', help='')
    
    return parser.parse_args()

def parse_example(example):
    features = tf.parse_single_example(
        example, features={
            'input': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string)
        })
    input = tf.decode_raw(features['input'], tf.uint8)
    input = tf.reshape(input, [PATCH_SIZE, PATCH_SIZE, 3])
    input = tf.cast(input, dtype=tf.float32) / 255
    label = tf.decode_raw(features['label'], tf.uint8)
    label = tf.reshape(label, [PATCH_SIZE, PATCH_SIZE, 3])
    label = tf.cast(label, dtype=tf.float32) / 255
    
    if CROP_SIZE < PATCH_SIZE:
        row = random.randint(0, PATCH_SIZE-CROP_SIZE)
        col = random.randint(0, PATCH_SIZE-CROP_SIZE)
        input = input[row:row+CROP_SIZE, col:col+CROP_SIZE, :]
        label = label[row:row+CROP_SIZE, col:col+CROP_SIZE, :]
    
    c1 = tf.random_uniform([], 0, 1)
    c2 = tf.random_uniform([], 0, 1)
    c3 = tf.random_uniform([], 0, 1)
    input, label = tf.cond(c1<0.5, lambda: (input[::-1, :, :], label[::-1, :, :]), lambda: (input, label))
    input, label = tf.cond(c2<0.5, lambda: (input[:, ::-1, :], label[:, ::-1, :]), lambda: (input, label))
    input, label = tf.cond(c3<0.5, lambda: (tf.transpose(input, [1, 0, 2]), tf.transpose(label, [1, 0, 2])), lambda: (input, label))
    
    return input, label

def read_data(save_dir, batch_size):
    fs_paths = sorted(glob.glob(join(save_dir, '*.tfrecord')))
    dataset = tf.data.TFRecordDataset(fs_paths)
    dataset = dataset.map(parse_example, num_parallel_calls=16)
    dataset = dataset.shuffle(800)
    dataset = dataset.prefetch(400)
    dataset = dataset.batch(batch_size, drop_remainder=True).repeat()
    iterator = dataset.make_one_shot_iterator()
    input, label = iterator.get_next()
    
    return input, label

def main():
    args = get_arguments()
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    input, label = read_data(args.data_dir, args.batch_size)
    input_v = close_op(tf.reduce_max(input, axis=-1, keepdims=True))
    label_v = close_op(tf.reduce_max(label, axis=-1, keepdims=True))
    
    input_i, input_r = rdnet(input_v, input, hei=CROP_SIZE, wid=CROP_SIZE)
    input_i = stretch(input_v, input_i)
    label_i, label_r = rdnet(label_v, label, reuse=True, hei=CROP_SIZE, wid=CROP_SIZE)
    label_i = stretch(label_v, label_i)
    g_vars = tl.layers.get_variables_with_name('retinex')
    
    #ini_loss
    ini = tf.reduce_mean(tf.square(input_i - input_v)) * 10
    ini += tf.reduce_mean(tf.square(label_i - label_v)) * 10
    #wtv_loss
    wtv = wtv_loss(input_v, input_i, CROP_SIZE) * 5
    wtv += wtv_loss(label_v, label_i, CROP_SIZE) * 5
    #com_loss
    com = tf.reduce_mean(tf.square(input_i * input_r - input)) * 100
    com += tf.reduce_mean(tf.square(label_i * label_r - label)) * 100
    #err_loss
    err = tf.reduce_mean(tf.square(input_r - label_r)) * 5
    #total_loss
    loss = ini + wtv + com + err
    
    steps_per_epoch = PATCH_NUM // args.batch_size
    step_num = steps_per_epoch * args.epoch_num
    start_step = steps_per_epoch * args.start_epoch
    global_step = tf.Variable(start_step, trainable=False)
    lr = tf.train.exponential_decay(args.learning_rate, global_step, step_num//2, 0.1, staircase=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(lr).minimize(loss, var_list=g_vars, global_step=global_step)
    
    sess.run(tf.global_variables_initializer())
    g_saver = tf.train.Saver(g_vars)
    ckpt = tf.train.latest_checkpoint(args.ckpt_dir)
    if ckpt:
        print('Restoring model...')
        g_saver.restore(sess, ckpt)
    
    local_step = 0
    avg_ini = 0
    avg_wtv = 0
    avg_com = 0
    avg_err = 0
    start_time = time.time()
    for step in range(start_step+1, step_num+1):
        _, b_ini, b_wtv, b_com, b_err = sess.run([train_op, ini, wtv, com, err])
        avg_ini += b_ini
        avg_wtv += b_wtv
        avg_com += b_com
        avg_err += b_err
        
        if step % args.print_interval == 0:
            local_step += args.print_interval
            cur_epoch = step // steps_per_epoch
            if step % steps_per_epoch == 0:
                cur_step = steps_per_epoch
            else:
                cur_epoch += 1
                cur_step = step % steps_per_epoch
            
            end_time = time.time()
            duration = float(end_time - start_time)
            start_time = end_time
            text = 'epoch:%d/%d, step:%d/%d, ini: %.6f, wtv: %.6f, com: %.6f, err: %.6f, time:%.2f sec'
            print(text % (cur_epoch, args.epoch_num, cur_step, steps_per_epoch, avg_ini/local_step, 
                          avg_wtv/local_step, avg_com/local_step, avg_err/local_step, duration))
        
        if step % 1000 == 0:
            local_step = 0
            avg_ini = 0
            avg_wtv = 0
            avg_com = 0
            avg_err = 0
        
        if step % steps_per_epoch == 0:
            print('Saving model...')
            g_saver.save(sess, join(args.ckpt_dir, 're.ckpt'), global_step=step)
    sess.close()

if __name__ == '__main__':
    main()
