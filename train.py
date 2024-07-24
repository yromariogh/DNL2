#@title Tensorflow
from absl import app, flags
from absl.flags import FLAGS
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from real_networks import *
import scipy.io as sio
from skimage.metrics import structural_similarity as SSIM

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
        
flags.DEFINE_string('map',"15",'GPU.... ')
flags.DEFINE_float('lr',1e-3, 'Learning rate .... ')
flags.DEFINE_string('net',"C4",'new ckpt.... ')
flags.DEFINE_string('old',"table",'old ckpt.... ')
flags.DEFINE_string('new',"table",'new ckpt.... ')
flags.DEFINE_boolean('retrain',False, 'Load and train .... ')
flags.DEFINE_boolean('evaluate',False, 'Load and train .... ')
flags.DEFINE_integer('epochs',1000, 'Training epochs .... ')
flags.DEFINE_integer('batch',1, 'Training epochs .... ')
flags.DEFINE_boolean('all',False, 'Ckpt .... ')
flags.DEFINE_string('lr_type','lr', 'Learning rate .... ')
flags.DEFINE_integer('lr_steps',1, 'Training epochs .... ')
flags.DEFINE_float('lr_rate',1, 'Learning rate .... ')


def main(_argv):
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"]= "0"
    
    from matplotlib import pyplot as plt
    path = os.getcwd()
    
    #----------------------------- directory of the spectral data set -------------------------                              # for windows
    L_bands    = 1; L_imput    = 1;


    #########################################################################################

    IMG_WIDTH = 134; IMG_HEIGHT = 128;

    epochs =  FLAGS.epochs # @param {type:"number"}
    reTrain = FLAGS.retrain # @param {type:"boolean"}
    lr = FLAGS.lr# @param {type:"raw"}
    map = FLAGS.map
    Yin = 'Y'+map[0]
    Yout = 'Y'+map[1]
    old_cp_dir=path+'/Weights/'+map+'/'
    if FLAGS.lr_type=='lr':
        lr_info = ', lr='+str(lr)
        lr_info = ''
    elif FLAGS.lr_type=='lr_schedule':
        lr_info = ', scheduler=('+str(lr)+','+str(FLAGS.lr_steps)+','+str(FLAGS.lr_rate)+')'
    old_cp_path = old_cp_dir+FLAGS.net+'_'+FLAGS.old+lr_info+'.h5' # @param {type:"raw"}
    new_cp_path = old_cp_dir+FLAGS.net+'_'+FLAGS.new+lr_info+'.h5' # @param {type:"raw"}

    
    

    import numpy as np
    from scipy.io import loadmat

    def PSNR_Metric(y_true, y_pred):
        return tf.reduce_mean(tf.image.psnr(y_true,y_pred,tf.reduce_max(y_true)))
    
    def tf_SSIM_Metric(y_true, y_pred):
        return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=tf.reduce_max(y_true),filter_size=1))
    
    def MSE_Metric(y_true, y_pred):
        return tf.reduce_mean(tf.keras.losses.MSE(y_true,y_pred))




    from psutil import virtual_memory
    ram_gb = virtual_memory().total / 1e9
    print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

    tf.keras.backend.clear_session()

    import random

    def get_list_imgs(data_path):
        data_path = data_path.decode('utf8')
        Yn = data_path.split('\\')[-1]+'_'
        print(Yn)
        if 'Train' in data_path:
            list_imgs = [Yn+str(i) for i in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 29, 30]]
        elif 'Test' in data_path:
            list_imgs = [Yn+str(i) for i in [1, 21, 28]]
        random.shuffle(list_imgs)
        print(list_imgs)
        return list_imgs

    class DataGen(tf.data.Dataset):

        def _generator(path_in, path_out, images, name):
            list_imgs = get_list_imgs(path_out) # Degraded
            print(list_imgs)
            path_in=path_in.decode('utf8')
            path_out=path_out.decode('utf8')
            name=name.decode('utf8')
            for img_fn in list_imgs:
                if any(str_i.decode('utf8') in img_fn for str_i in images):
                    input = loadmat(path_in+'/'+img_fn.replace(name,Yin)+'.mat')[Yin]
                    output = loadmat(path_out+'/'+img_fn+'.mat')[name]
                    yield input, output

        def __new__(cls, path_in, path_out, images, name, input_size=(128, 134)):
            output_signature = tf.TensorSpec(shape=input_size, dtype=tf.float32)

            return tf.data.Dataset.from_generator(
                cls._generator,
                output_signature=(output_signature,output_signature),
                args=(path_in, path_out, images, name)
            )


    def get_pipeline(batch_size, path_in, path_out, images, name, input_size, buffer_size=3, cache_dir=''):

        dataset = DataGen(path_in, path_out, images, name)

        pipeline_data = (
            dataset
            .cache(cache_dir)
            .shuffle(buffer_size)  # cache_dir='' guarda el cache en RAM
            .batch(batch_size, drop_remainder=False)
            .prefetch(buffer_size)
        )

        return pipeline_data

    metrics = ['val_loss', 'val_PSNR_Metric', 'val_tf_SSIM_Metric', 'val_MSE_Metric']
    titles = ['Validation Loss', 'Validation PSNR', 'Validation SSIM', 'Validation MSE']
    save_path = os.path.join(os.getcwd(),'Metrics, ' + str(FLAGS.net) + ', ' + str(epochs) + lr_info)

    class MetricsPlotCallback(tf.keras.callbacks.Callback):
        def __init__(self, metrics, titles, save_path):
            self.metrics = metrics
            self.titles = titles
            self.fig, self.axs = None, None
            self.rows = len(metrics) // 2
            self.cols = 2
            self.save_path = save_path
            self.metric_history = {metric: [] for metric in self.metrics}

        def on_train_begin(self, logs=None):
            self.fig, self.axs = plt.subplots(self.rows, self.cols, figsize=(10, 8))
            self.fig.suptitle('Training Metrics')

        def on_epoch_end(self, epoch, logs=None):
            for metric in self.metrics:
                self.metric_history[metric].append(logs.get(metric))

            for i, metric in enumerate(self.metrics):
                row = i // self.cols
                col = i % self.cols

                ax = self.axs[row, col]
                ax.plot(range(epoch + 1), self.metric_history[metric], marker='o')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric)
                ax.set_title(self.titles[i])

            # Adjust spacing between subplots
            plt.tight_layout()
            self.fig.canvas.draw()

            # Save the combined plot as an SVG file
            plt.savefig(f'{self.save_path}.svg')
            save_path = f'{self.save_path}.png'
            plt.savefig(save_path)

            sio.savemat(f'{self.save_path}.mat', {metric: self.metric_history[metric] for metric in self.metrics})
            

    metrics_plot_callback = MetricsPlotCallback(metrics, titles, save_path)

    
    # batch = 30 # 8451

    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     lr,
    #     decay_steps=FLAGS.lr_steps,
    #     decay_rate=FLAGS.lr_rate,
    #     staircase=False,
    #     )
    
    class LearningRateMultiplier(tf.keras.callbacks.Callback):
        def __init__(self, lr_step, lr_rate):
            super(LearningRateMultiplier, self).__init__()
            self.lr_step = lr_step
            self.lr_rate = lr_rate

        def on_epoch_end(self, epoch, logs=None):
            if self.lr_rate == 1:
                return
            elif (epoch + 1) % self.lr_step == 0:
                old_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                new_lr = old_lr * self.lr_rate
                print(f'\nChanging learning rate from {old_lr:.6f} to {new_lr:.6f}')
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)

    
    loss =  'mean_squared_error'# @param {type:"string"}
    


    #########################################################################################

    # if FLAGS.lr_type=='lr':
    optimizad = tf.keras.optimizers.Adam(learning_rate=lr, amsgrad=False)
    # elif FLAGS.lr_type=='lr_schedule':
    #     optimizad = tf.keras.optimizers.Adam(learning_rate=lr_schedule, amsgrad=False)  

    scheduler = LearningRateMultiplier(FLAGS.lr_steps, FLAGS.lr_rate)
    
    #################################################################################################

    images=[Yout]

    train_path_in=os.path.join(path,'Train','Y',Yin)
    print("train_path_in:",train_path_in)
    train_path_out=os.path.join(path,'Train','Y',Yout)
    print("train_path_out:",train_path_out)

    train_ds = get_pipeline(batch_size=FLAGS.batch, path_in=train_path_in, path_out=train_path_out, images=images, name=Yout, input_size=(128,134))

    test_path_in=os.path.join(path,'Test','Y',Yin)
    print("test_path_in:",test_path_in)
    test_path_out=os.path.join(path,'Test','Y',Yout)
    print("test_path_out:",test_path_out)

    test_ds = get_pipeline(batch_size=1, path_in=test_path_in, path_out=test_path_out, images=images, name=Yout, input_size=(128,134))
    # sample = next(iter(train_ds))


    #-------------Net_model----------------------------------------------------------------
    if FLAGS.net=='C0':
        model = C0(input_size=(IMG_HEIGHT,IMG_WIDTH,L_bands))
    elif FLAGS.net=='C1':
        model = C1(input_size=(IMG_HEIGHT,IMG_WIDTH,L_bands))
    elif FLAGS.net=='C2':
        model = C2(input_size=(IMG_HEIGHT,IMG_WIDTH,L_bands))
    elif FLAGS.net=='C3':
        model = C3(input_size=(IMG_HEIGHT,IMG_WIDTH,L_bands))
    elif FLAGS.net=='C4':
        model = C4(input_size=(IMG_HEIGHT,IMG_WIDTH,L_bands))

    if reTrain:
        print('Loading previous weights: ',old_cp_path)
        model.load_weights(old_cp_path)
      
    model.compile(optimizer=optimizad, loss=loss,metrics = [PSNR_Metric, tf_SSIM_Metric, MSE_Metric])
    if FLAGS.evaluate:
        print('Loading previous weights: ',old_cp_path)
        model.load_weights(old_cp_path)
        model.evaluate(test_ds)
    else:
        if FLAGS.all:
            history = model.fit(train_ds, validation_data=test_ds, epochs=epochs, callbacks=[scheduler,metrics_plot_callback])
        else:
            history = model.fit(train_ds, validation_data=test_ds, epochs=epochs, callbacks=[scheduler,metrics_plot_callback])
        last_psnr=history.history['PSNR_Metric'][-1]
        best_psnr=np.max(history.history['PSNR_Metric'])
        iter_psnr=history.history['PSNR_Metric'].index(best_psnr)
        best_loss=np.min(history.history['loss'])
        iter_loss=history.history['loss'].index(best_loss)
        # name=#@param {type:"string"}
        name=Yout+', '+str(best_psnr)+', '+str(best_loss)+', '+str(iter_psnr)+', '+str(iter_loss)+', '+str(epochs)+lr_info+'.h5'
        print(name)
        model.save_weights(name)

        #.------------ seee the accfuracy---------------
        # Extract metric data from history

        # Create a subplot with 2 rows and 2 columns
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        for i, metric in enumerate(metrics):
            data = history.history[metric]
            row = i // 2
            col = i % 2

            axs[row, col].plot(data)
            axs[row, col].set_xlabel('Epoch')
            axs[row, col].set_ylabel(metric)
            axs[row, col].set_title(titles[i])

        # Adjust spacing between subplots
        plt.tight_layout()

        # Save the combined plot as an SVG file
        plt.savefig(f'Metrics_{FLAGS.net}_{epochs}_{lr_info}.svg')
        plt.savefig(f'Metrics_{FLAGS.net}_{epochs}_{lr_info}.png')

        # Show the plot (optional)
        # plt.show()

        #model.save_weights("model_weights_norm_inpal1.h5")

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass