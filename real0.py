
from absl import app, flags
from absl.flags import FLAGS
from real_functions import Transpose_CASSI, ForwardFunction, PSNR_Metric, PnP_proposed, parameter, Update_x, Update_x_0
from real_networks import C0, C1, C2, C3, C4
import mat73
from scipy.sparse import csr_matrix, find



flags.DEFINE_string('sX',"1",'Scene 1 to 30')
flags.DEFINE_string('denoiser',"BM3D",'Denoiser used [BM3D, RF, UNet]')
flags.DEFINE_float('l1',1.0,'Importance of Mapping network ')


flags.DEFINE_string('map',"15",'Mapping network and Y')
flags.DEFINE_string('sCA',"CA_ideal",'Filename of the Coded Aperture')
flags.DEFINE_boolean('all',False,'Show iter-recons ')
flags.DEFINE_string('init',"Transpose",'Initialization')
flags.DEFINE_boolean('interrupt',False,'Stop if finished ')
flags.DEFINE_boolean('again',False,'Run again ')


flags.DEFINE_integer('global_i',100,'Global iters of algorithm ')
flags.DEFINE_integer('internal_i',60,'Internal epochs of network ')
flags.DEFINE_boolean('show_plots',False,'Show iter-recons ')
flags.DEFINE_integer('freq',10,'Freq of saving ')
flags.DEFINE_float('lr',1/50,'Reconstruction network learning rate ')
flags.DEFINE_string('net',"C0",'Network used')
flags.DEFINE_string('norm',"MinMax",'Transpose normalization')
flags.DEFINE_string('PM',"D5",'Propagation model: [Simple,Hoover,Hans,D5,Song,Efficient,Kittle]')
flags.DEFINE_string('PMv',"new",'Propagation version: [old,new]')

def main(_argv):
    import os
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            print(e)
    import numpy as np
    import scipy.io as sio
    data_path = os.getcwd()
    

    sX = FLAGS.sX
    map = FLAGS.map
    l1 = FLAGS.l1
    lr = FLAGS.lr
    PM = FLAGS.PM
    PMv = FLAGS.PMv
    norm = FLAGS.norm

    M, N, L = 128, 128, 7
    bands = np.array([474.1935, 501.2903, 528.3871, 555.4839, 582.5806, 609.6774, 636.7742])
    ydims = [M, N + L - 1, 1]
    xdims = [M, N, L]

    if PMv == 'old':
        if PM in ['Simple']:
            sCA = 'CA_ideal'
        elif PM in ['Hans']:
            sCA = 'CA_crop'
        
        ca = np.array(sio.loadmat(os.path.join(data_path,'CA',sCA+'.mat'))['CA']);
        if ca.shape == (M, N):
            CA = np.zeros((M, N, L))
            for i in range(L):
                CA[:, :, i] = ca
            CA = np.expand_dims(CA, axis=0)
        else:
            CA = np.expand_dims(ca, axis=0)
        CA=(CA-np.min(CA))/(np.max(CA)-np.min(CA))

        if PM == 'Simple':
            info = 'CA_ideal'

    elif PMv == 'new':
        sCA = 'H_from_'+PM

        try:
            H = sio.loadmat(os.path.join(data_path,'CA',sCA+'.mat'))['H']
        except:
            H = mat73.loadmat(os.path.join(data_path,'CA',sCA+'.mat'))['H']
        try:
            Ht = sio.loadmat(os.path.join(data_path,'CA',sCA.replace('H_','Ht_')+'.mat'))['Ht']
        except:
            Ht = mat73.loadmat(os.path.join(data_path,'CA',sCA.replace('H_','Ht_')+'.mat'))['Ht']

        
        info = 'H_from_'+PM


    save_path = os.path.join(data_path, 'Results')
    listprev = []


    filename = ', '.join([sX,map,info,str(l1),str(FLAGS.global_i),str(FLAGS.internal_i),FLAGS.denoiser,str(FLAGS.all),str(lr),FLAGS.init,FLAGS.net])
    listdir = sorted(os.listdir(os.path.join(save_path,map)))
    for f in listdir:
        if f.startswith(filename) and f.endswith('.mat'):
            print(filename+' -> Already exists')
            if (FLAGS.init == "Transpose" or FLAGS.init == "GT") and not FLAGS.again:
                return
            if FLAGS.init == "Previous" or FLAGS.again:
                listprev.append(f)
    filename = filename+', '+str(len(listprev)+1)
    
    
    print(filename+' -> Starting')

    
    
    

    ## PnP ADMM ##

    xg = np.array(sio.loadmat(os.path.join(data_path,'X/GT_'+sX+'.mat'))['X'])
    xg=(xg-np.min(xg))/(np.max(xg)-np.min(xg))
    y = np.array(sio.loadmat(os.path.join(data_path,'Y/Y'+map[-1],'Y'+map[-1]+'_'+sX+'.mat'))['Y'+map[-1]])
    y = tf.expand_dims(tf.expand_dims(tf.squeeze(y), -1), 0)


    [row, col, val] = find(H)
    H0 = H.shape[0]
    H1 = H.shape[1]
    del(H)
    ind = np.asarray([row, col])
    ind = np.transpose(ind, (1, 0))

    H_s = tf.SparseTensor(indices=ind, values=val, dense_shape=[H0, H1])



    if FLAGS.init == "Transpose":
        init = Transpose_CASSI(Ht, y, norm, M,N,L, PMv)
        del(Ht)
    if FLAGS.init == "GT":
        init = xg
    elif FLAGS.init == "Previous":
        listprevPSNRs = []
        for prev in listprev:
            listprevPSNRs.append(float(prev[prev.rfind(' ')+1:prev.rfind('.mat')]))
        print(listprevPSNRs)
        if listprevPSNRs == []:
            print(filename+' -> Not a previous')
            return
        else:
            init = np.squeeze(np.array(sio.loadmat(os.path.join(save_path,map,listprev[listprevPSNRs.index(max(listprevPSNRs))]))['best_im'])).astype(np.float64)

        
    if l1 != 0.0 or FLAGS.all == True:
        old_cp_dir = os.path.join(data_path,'Weights',map)
        if FLAGS.net == "C0":
            Mx_temp = C0(input_size=ydims)
        elif FLAGS.net == "C1":
            Mx_temp = C1(input_size=ydims)
        elif FLAGS.net == "C2":
            Mx_temp = C2(input_size=ydims)
        elif FLAGS.net == "C3":
            Mx_temp = C3(input_size=ydims)
        elif FLAGS.net == "C4":
            Mx_temp = C4(input_size=ydims)
        Mx_temp.load_weights(os.path.join(old_cp_dir,FLAGS.net+'.h5'))

        Mx = Mx_temp.predict(ForwardFunction(H_s, xg, M,N,L))
        print(PSNR_Metric(y,Mx))


            
        model = Update_x(H=H_s, init=init, M=M, N=N, L=L, net=FLAGS.net, PMv=PMv)
        it = 2
        for layer in Mx_temp.layers[1:]:  # Desde la primera convolucion (0 es Input)
            it = it + 1  # Ignora la capa Input y la de Lambda-Hx
            model.layers[it].set_weights(Mx_temp.get_layer(layer.name).get_weights())
            model.layers[it].trainable = False
        optimizad = tf.keras.optimizers.Adam(learning_rate=lr, amsgrad=False) # Jugar con este lr (estaba en 1/50)
        losses = {"mapping": "mean_squared_error", "forward": "mean_squared_error","x_layer": "mean_squared_error"}
        lossWeights = {"mapping": l1, "forward": 1 - l1, "x_layer": 1}

    else:            
        model = Update_x_0(H=H_s, init=init, M=M, N=N, L=L, PMv=PMv)
        optimizad = tf.keras.optimizers.Adam(learning_rate=lr, amsgrad=False) # Este estaba en 1/20, ¿habrá que volver a correr los de 0.0?
        losses = {"forward": "mean_squared_error","x_layer": "mean_squared_error"}
        lossWeights = {"forward": 1 - l1, "x_layer": 1}

    model.compile(optimizer=optimizad, loss=losses, loss_weights=lossWeights,metrics=["accuracy"])

    pp = save_path + '/' + map + '/' + filename

    parameters = parameter(rho=1, gamma=1.05, lamnda=0.5, global_iteration=FLAGS.global_i, show_plots=FLAGS.show_plots,internal_iteration=FLAGS.internal_i, path=pp)
    best_im_psnr,best_psnr,best_iter_psnr,best_im_ssim,best_ssim,best_iter_ssim,best_im_sam,best_sam,best_iter_sam,finished = PnP_proposed(parameters, M, N,L, model, y, xg,l1, lossWeights,FLAGS.freq,FLAGS.denoiser,FLAGS.all,bands,FLAGS.interrupt)
    if finished:
        scipy.io.savemat(pp + ', ' + str(best_psnr)+ ', ' + str(best_ssim)+ ', ' + str(best_sam)  + ".mat",
                            {'best_im_psnr': best_im_psnr, 'best_psnr': best_psnr, 'best_iter_psnr': best_iter_psnr,
                            'best_im_ssim': best_im_ssim, 'best_ssim': best_ssim, 'best_iter_ssim': best_iter_ssim,
                            'best_im_sam': best_im_sam, 'best_sam': best_sam, 'best_iter_sam': best_iter_sam})
        print(filename+' -> FINISHED')
    else:
        print(filename+' -> Interrupted')

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

