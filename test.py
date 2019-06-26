# CUDA_VISIBLE_DEVICES=0 python test.py
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import os, time
from model import *
from PIL import Image

in_dir = 'samples/'
out_dir = 'out/'
rd_dir = 'rd_model/'
fe_dir = 'fe_model/'

def main():
    tf.reset_default_graph()
    img_holder = tf.placeholder(tf.float32, shape=[None, None, 3])
    hei = tf.placeholder(tf.int32)
    wid = tf.placeholder(tf.int32)
    
    img = tf.expand_dims(img_holder, 0)
    img_v = tf.reduce_max(img, axis=-1, keepdims=True)
    img_v = close_op(img_v)
    
    img_i, img_r = rdnet(img_v, img, hei, wid)
    img_i = stretch(img_v, img_i)
    img_crm = CRM(img, img_i)
    
    out = fenet(img_crm, img, img_r, hei, wid)
    out = tf.clip_by_value(out, 0, 1)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    sess.run(init_op)
    
    print('Loading...')
    ckpt = tf.train.latest_checkpoint(rd_dir)
    rd_vars = tl.layers.get_variables_with_name('retinex', printable=False)
    rd_saver = tf.train.Saver(rd_vars)
    rd_saver.restore(sess, ckpt)
    
    ckpt = tf.train.latest_checkpoint(fe_dir)
    fe_vars = tl.layers.get_variables_with_name('fusion', printable=False)
    fe_saver = tf.train.Saver(fe_vars)
    fe_saver.restore(sess, ckpt)
    
    img_files = os.listdir(in_dir)
    img_num = len(img_files)
    img_id = 0
    avg_time = 0
    
    for img_file in img_files:
        img_id += 1
        in_img = Image.open(in_dir+img_file).convert("RGB")
        assert in_img is not None
        w = in_img.size[0]
        h = in_img.size[1]
        in_img = np.array(in_img) / 255
        
        start_time = time.time()
        out_img = sess.run(out, feed_dict={img_holder:in_img, hei:h, wid:w})
        out_img = tf.squeeze(out_img)
        out_img = out_img.eval(session=sess) * 255
        duration = float(time.time() - start_time)
        avg_time += duration
        
        out_name = img_file.split('.', 1)[0] + '.png'
        out_img = Image.fromarray(np.uint8(out_img))
        out_img.save(out_dir+out_name)
        print('step: %d/%d, time: %.2f sec' % (img_id, img_num, duration))
    
    print('Finish! avg_time: %.2f sec' % (avg_time / img_num))
    sess.close()

if __name__ == '__main__':
    main()
