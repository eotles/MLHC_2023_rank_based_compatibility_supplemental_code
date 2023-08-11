import json
import os
import tensorflow as tf

try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle


# -------------------- MODELS

def make_f_o(n_features=1000,
              C=0.01,
              optimizer='adam'
             ):
    
    inp = tf.keras.Input(shape=(n_features))
    out = tf.keras.layers.Dense(1,
                                kernel_regularizer=tf.keras.regularizers.l2(C),
                                activation='sigmoid')(inp)
    model = tf.keras.Model(inp, out)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['AUC'])
    
    return model
    
## --------------------
    
def make_approx_incompatibility_loss_fx(s=100):
 
    def approx_incompatibility_loss_fx(y, p_o, p_u):
        o_d_pm = p_o[y==1] - p_o[y==0][:, None]
        u_d_pm = p_u[y==1] - p_u[y==0][:, None]

        o_pm = tf.sigmoid(o_d_pm*s)
        u_pm = tf.sigmoid(u_d_pm*s)

        num_pm = tf.multiply(o_pm, u_pm)
        num = tf.reduce_sum(num_pm)
        den = tf.reduce_sum(o_pm)

        c = tf.math.divide_no_nan(num, den)
        return 1-c

    return approx_incompatibility_loss_fx
    
    
def make_weighted_loss(w=1.0, loss=tf.metrics.binary_crossentropy):
    w = tf.cast(w, tf.float32)
    
    def weighted_loss(*args):
        return tf.multiply(w, loss(*args))
    
    return weighted_loss
    
    
    
def make_weighted_combination(alpha=0.5,
                              metric=None,
                              compatibility=None):
    alpha = tf.cast(alpha, tf.float32)
    
    if metric is None:
        metric = tf.metrics.binary_crossentropy
    if compatibility is None:
        compatibility = make_approx_incompatibility_loss_fx()
    
    def weighted_combination(y, p_o, p_u):
        return tf.add(tf.multiply(alpha, metric(y, p_u)),
                      tf.multiply((1-alpha), compatibility(y, p_o, p_u)))
    
    return weighted_combination
    
    
    
def make_f_u(n_features=1000,
             alpha=0.5,
             s=100,
             C=0.01,
             optimizer='adam'
             ):
    _X = tf.keras.Input(shape=(n_features))
    _y = tf.keras.Input(shape=(1,))

    _p = tf.keras.layers.Dense(1,
                               kernel_regularizer=tf.keras.regularizers.l2(C),
                               activation='sigmoid')(_X)

    _p_hat_o = tf.keras.Input(shape=(1,))
    model = tf.keras.Model(inputs=[_X, _y, _p_hat_o], outputs=_p)

    ail_fx = make_approx_incompatibility_loss_fx(s=s)
    wail_fx = make_weighted_loss(w=1-alpha,
                                loss=ail_fx)

    model.add_metric(ail_fx(_y, _p_hat_o, _p),
                     name='AIL',
                     aggregation='mean')
    
    model.add_loss(wail_fx(_y, _p_hat_o, _p))

    wc_loss = make_weighted_combination(alpha=alpha,
                                        metric=tf.metrics.binary_crossentropy,
                                        compatibility=ail_fx)

    model.add_metric(wc_loss(_y, _p_hat_o, _p),
                     name='WC_Loss',
                     aggregation='mean')


    w_bce_fx = make_weighted_loss(w=alpha, loss=tf.metrics.binary_crossentropy)
    model.add_metric(w_bce_fx(_y, _p), name='WBCE', aggregation='mean')
    model.compile(loss=w_bce_fx, optimizer=optimizer,
                  metrics=['AUC', 'binary_crossentropy'])
    
    return model








