import tensorflow as tf

class Trinet(tf.keras.Model):
    def __init__(self,embedding_dim,image_shape=(256,128,3)):
        super(Trinet, self).__init__()
        self.basemodel = tf.keras.applications.ResNet50(input_shape=image_shape,include_top=False,weights='imagenet')
        self.basemodel.trainable= False
        self.gpool = tf.keras.layers.GlobalAveragePooling2D()
        self.l1 = tf.keras.layers.Dense(1024)
        self.bnorm = tf.keras.layers.BatchNormalization(axis=1,momentum=0.9,epsilon=1e-5,scale=True)
        self.l2 = tf.keras.layers.Dense(embedding_dim,kernel_initializer=tf.keras.initializers.Orthogonal)
    
    def call(self,inputs):
        x = self.basemodel(inputs)
        x = self.gpool(x)
        x = self.l1(x)
        x = self.bnorm(x)
        x = self.l2(x)
        return x