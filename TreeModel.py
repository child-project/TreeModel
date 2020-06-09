#!python
import tensorflow as tf
import tensorflow_hub as hub
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow.compat.v1 as tfc
from glob import glob
from matplotlib import pyplot as plt
import pandas as pd
from collections import Counter
import time
import tensorflow as tf
from IPython.display import Image, display
import matplotlib.image as img
# import sys
# import codecs
# sys.stdout=codecs.getwriter("utf-8")(sys.stdout.detach())

CHANNELS = 3  # number of image channels (RGB)


def build_graph(hub_module_url, target_image_path):
    # Step 1) Prepare pre-trained model for extracting image features.
    module = hub.Module(hub_module_url)
    height, width = hub.get_expected_image_size(module)

    # Copied a method of https://github.com/GoogleCloudPlatform/cloudml-samples/blob/bf0680726/flowers/trainer/model.py#L181
    # and fixed for all type images (not only jpeg)
    def decode_and_resize(image_str_tensor):
        """Decodes jpeg string, resizes it and returns a uint8 tensor."""
        image = tf.image.decode_png(image_str_tensor, channels=CHANNELS)
        # Note resize expects a batch_size, but tf_map supresses that index,
        # thus we have to expand then squeeze.  Resize returns float32 in the
        # range [0, uint8_max]
        image = tf.expand_dims(image, 0)
        image = tf.image.resize(
            image, [height, width])
        image = tf.squeeze(image)
        image = tf.cast(image, dtype=tf.uint8)
        return image

    def to_img_feature(images):
        """Extract the feature of image vectors"""
        outputs = module(dict(images=images), signature="image_feature_vector", as_dict=True)
        return outputs['default']

    # Step 2) Extract image features of the target image.
    target_image_bytes = tf.io.gfile.GFile(target_image_path, 'rb').read()
    target_image = tf.constant(target_image_bytes, dtype=tf.string)
    target_image = decode_and_resize(target_image)
    target_image = tf.image.convert_image_dtype(target_image, dtype=tf.float32)
    target_image = tf.expand_dims(target_image, 0)
    target_image = to_img_feature(target_image)


    # Step 3) Extract image features of input images.
    tfc.disable_v2_behavior()
    # input_byte  = tf.Variable(tf.ones(shape = [None], dtype = tf.string))
    input_byte = tfc.placeholder(tf.string, shape=[None])
    input_image = tf.map_fn(decode_and_resize, input_byte, back_prop=False, dtype=tf.uint8)
    input_image = tf.image.convert_image_dtype(input_image, dtype=tf.float32)
    input_image = to_img_feature(input_image)

    # Step 4) Compare cosine_similarities of the target image and the input images.
    dot = tf.tensordot(target_image, tf.transpose(input_image), 1)
    similarity = dot / (tf.norm(target_image, axis=1) * tf.norm(input_image, axis=1))
    similarity = tf.reshape(similarity, [-1])

    return input_byte, similarity
    #return similarity

uimage = "tree/leehj.png"

def select_class(target_img_path):
    hub_module_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/1"  # @param {type:"string"}
    tclass = []
    p_class = []
    input_img_paths = glob('tree/*.png')
    for im in input_img_paths:
        tclass.append(im[5:-len('_00.png')])
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Load bytes of image files
    image_bytes = [tf.io.gfile.GFile(name, 'rb').read()
                   for name in [target_img_path] + input_img_paths]

    with tf.Graph().as_default():
        input_byte, similarity_op = build_graph(hub_module_url, target_img_path)

        with tfc.Session() as sess:
            sess.run(tfc.global_variables_initializer())
            t0 = time.time()  # for time check

            # Inference similarities
            similarities = sess.run(similarity_op, feed_dict={input_byte: image_bytes})
            print("%d images inference time: %.2f s" % (len(similarities), time.time() - t0))
    similarities = similarities[1:]
    print(len(similarities), len(input_img_paths), len(tclass))
    Result = pd.DataFrame({"S": similarities, "I": input_img_paths, "C": tclass})
    sort_Result = Result.sort_values(by=['S'], ascending=False)
    tclass.append(10)
    print(len(tclass))

    cdd_class=[]
    cdd_class.append(sort_Result.iloc[1][2])
    cdd_class.append(sort_Result.iloc[2][2])
    cdd_class.append(sort_Result.iloc[3][2])
    cdd_class.append(sort_Result.iloc[4][2])
    cdd_class.append(sort_Result.iloc[5][2])
    print(cdd_class)
    itype = Counter(cdd_class).most_common(n=1)[0][0]
    print(itype)
    plotImages(sort_Result)
    return itype


def plotImages(sort_Result):
    plt.figure(figsize=(20, 5))

    plt.subplot(161)
    plt.imshow(img.imread(sort_Result.iloc[0][1]))
    plt.title('Original')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(162)
    plt.imshow(img.imread(sort_Result.iloc[1][1]))
    plt.title('1st')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(163)
    plt.imshow(img.imread(sort_Result.iloc[2][1]))
    plt.title('2nd')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(164)
    plt.imshow(img.imread(sort_Result.iloc[3][1]))
    plt.title('3rd')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(165)
    plt.imshow(img.imread(sort_Result.iloc[4][1]))
    plt.title('4th')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(166)
    plt.imshow(img.imread(sort_Result.iloc[5][1]))
    plt.title('5th')
    plt.xticks([])
    plt.yticks([])

    plt.show()

select_class(uimage)
