from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, concatenate
import math
import numpy as np
import tensorflow as tf
import scipy
import matplotlib.pyplot as plt
from sewar.full_ref import sam as SAM
from skimage.metrics import structural_similarity as SSIM
from real_networks import *

def g(x, alpha, mu, sigma1, sigma2):
    sigma = (x < mu)*sigma1 + (x >= mu)*sigma2
    return alpha*np.exp((x-mu)**2 / (-2*(sigma**2)))


def component_x(x): return g(x, 1.056, 5998, 379, 310) + \
    g(x, 0.362, 4420, 160, 267) + g(x, -0.065, 5011, 204, 262)


def component_y(x): return g(x, 0.821, 5688, 469, 405) + \
    g(x, 0.286, 5309, 163, 311)


def component_z(x): return g(x, 1.217, 4370, 118, 360) + \
    g(x, 0.681, 4590, 260, 138)


def xyz_from_xy(x, y):
    """Return the vector (x, y, 1-x-y)."""
    return np.array((x, y, 1-x-y))


ILUMINANT = {
    'D65': xyz_from_xy(0.3127, 0.3291),
    'E':  xyz_from_xy(1/3, 1/3),
}

COLOR_SPACE = {
    'sRGB': (xyz_from_xy(0.64, 0.33),
             xyz_from_xy(0.30, 0.60),
             xyz_from_xy(0.15, 0.06),
             ILUMINANT['D65']),

    'AdobeRGB': (xyz_from_xy(0.64, 0.33),
                 xyz_from_xy(0.21, 0.71),
                 xyz_from_xy(0.15, 0.06),
                 ILUMINANT['D65']),

    'AppleRGB': (xyz_from_xy(0.625, 0.34),
                 xyz_from_xy(0.28, 0.595),
                 xyz_from_xy(0.155, 0.07),
                 ILUMINANT['D65']),

    'UHDTV': (xyz_from_xy(0.708, 0.292),
              xyz_from_xy(0.170, 0.797),
              xyz_from_xy(0.131, 0.046),
              ILUMINANT['D65']),

    'CIERGB': (xyz_from_xy(0.7347, 0.2653),
               xyz_from_xy(0.2738, 0.7174),
               xyz_from_xy(0.1666, 0.0089),
               ILUMINANT['E']),
}


class ColourSystem:

    def __init__(self, bands=np.array([0]), cs='sRGB'):

        # Chromaticities
        bands = bands*10

        self.cmf = np.array([component_x(bands),
                             component_y(bands),
                             component_z(bands)])

        self.red, self.green, self.blue, self.white = COLOR_SPACE[cs]

        # The chromaticity matrix (rgb -> xyz) and its inverse
        self.M = np.vstack((self.red, self.green, self.blue)).T
        self.MI = np.linalg.inv(self.M)

        # White scaling array
        self.wscale = self.MI.dot(self.white)

        # xyz -> rgb transformation matrix
        self.A = self.MI / self.wscale[:, np.newaxis]


    def get_transform_matrix(self):

        XYZ = self.cmf
        RGB = XYZ.T @ self.A.T
        RGB = RGB / np.sum(RGB, axis=0, keepdims=True)
        return RGB

    def spec_to_rgb(self, spec):
        """Convert a spectrum to an rgb value."""
        M = self.get_transform_matrix()
        rgb = spec @ M
        return rgb


class X_layer(Layer):
    def __init__(self, init, M=128, N=128, L=7):
        super(X_layer, self).__init__()
        self.init = tf.constant_initializer(init)
        self.X = self.add_weight(shape=(1, M, N, L), initializer=self.init, trainable=True)

    def call(self, inputs):
        return tf.multiply(self.X, inputs)

# class ForwardLayer(tf.keras.layers.Layer):
#     def __init__(self, CA, M, N, L, **kwargs):
#         super(ForwardLayer, self).__init__(**kwargs)
#         self.CA = CA
#         self.M = M
#         self.N = N
#         self.L = L
#
#     def call(self, inputs):
#         # Convert the input tensor to a dense tensor
#         x = tf.keras.backend.reshape(inputs, [-1, 1])
#
#         # Transpose the input tensor before performing the matmul
#         x_transposed = tf.keras.backend.transpose(x)
#
#         # Get the batch size from the input tensor
#         batch_size = tf.shape(inputs)[0]
#
#         # Repeat x_transposed to match the batch size of self.CA
#         x_transposed_batched = tf.keras.backend.repeat(x_transposed, batch_size)
#
#         # Convert the NumPy array CA to a TensorFlow sparse tensor
#         CA_sparse_tensor = tf.sparse.from_dense(self.CA)
#
#         # Perform the operation using sparse tensor multiplication (tf.sparse.sparse_dense_matmul)
#         Y_sparse = tf.sparse.sparse_dense_matmul(CA_sparse_tensor, x_transposed_batched)
#
#         # Convert the SparseTensor Y_sparse to a dense tensor Y
#         Y = tf.sparse.to_dense(Y_sparse, validate_indices=False)
#
#         # Reshape the output back to the desired shape
#         Y = tf.keras.backend.reshape(Y, [batch_size, self.M, self.N + self.L - 1, 1])
#
#         return Y

def ForwardFunction(H, x, M,N,L):
    Aux = tf.reshape(x, (M, N, L)) # like a Squeeze
    Aux = tf.transpose(Aux, perm=[2, 1, 0]) # like a order='F'
    Aux = tf.reshape(Aux, (M * N * L, 1)) # like a Flatten
    Aux = tf.sparse.sparse_dense_matmul(H, tf.cast(Aux,tf.float64))
    Aux = tf.reshape(tf.transpose(tf.reshape(Aux, (N+L-1, M))), (1, M, N+L-1, 1))
    return Aux

def Update_x(H, PMv, pretrained_weights=None, M=128, N=128, L=7, init=np.ones((128, 128, 7)), net="C0"):
    input_size = (M, N, L)
    inputs = Input(input_size)
    X = X_layer(init=init, M=M, N=N, L=L)(inputs)
    y_1 = Lambda(lambda x: ForwardFunction(H, x, M,N,L), name='forward')(X)
    if net == "C0":
        y_2 = C0_trained(y_1)
    elif net == "C1":
        y_2 = C1_trained(y_1)
    elif net == "C2":
        y_2 = C2_trained(y_1)
    elif net == "C3":
        y_2 = C3_trained(y_1)
    elif net == "C4":
        y_2 = C4_trained(y_1)
    elif net == "GAN":
        y_2 = GAN_trained(y_1)
    model = Model(inputs, [y_2,y_1, X])
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

def Update_x_0(H, PMv, pretrained_weights=None, M=128, N=128, L=7, init=np.ones((128, 128, 7))):
    input_size = (M, N, L)
    inputs = Input(input_size)
    X = X_layer(init=init, M=M, N=N, L=L)(inputs)
    y_1 = Lambda(lambda x: ForwardFunction(H, x, M,N,L), name='forward')(X)
    # y_1 = ForwardLayer(CA, M,N,L)(X)
    model = Model(inputs, [y_1, X])
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

def Forward_CASSI(CA, x, M,N,L, PMv="old"):
    if PMv == 'old':
        Aux1 = tf.multiply(CA, x)
        Aux1 = tf.pad(Aux1, [[0, 0], [0, 0], [0, L - 1], [0, 0]])
        Y = None
        for i in range(L):
            Tempo = tf.roll(Aux1, shift=i, axis=2)
            if Y is not None:
                Y = tf.concat([Y, tf.expand_dims(Tempo [:, :, :, i], -1)], axis=3)
            else:
                Y = tf.expand_dims(Tempo [:, :, :, i], -1)
        Y = tf.expand_dims(tf.reduce_sum(Y, 3), -1).numpy()
    elif PMv == 'new':
        ## x_np = x.numpy() if isinstance(x, tf.Tensor) else x
        # Y = CA.dot(np.reshape(x_np,[M*N*L,1],order='F')) #CA es realmente H
        Y = tf.matmul(CA, np.reshape(x,[M*N*L,1],order='F'))
        Y = np.reshape(Y, [1, M, N+L-1, 1], order='F')
    return Y

def Transpose_CASSI(CA, y, norm, M,N,L, PMv="old"):  # CASSI Transpose model (x = H'*y)
    print('CA shape: ', CA.shape)
    print('y shape: ', y.shape)
    if PMv == 'old':
        X = None
        for i in range(L):
            Tempo = tf.roll(y, shift=-i, axis=2)
            if X is not None:
                X = tf.concat([X, Tempo [:, :, 0:M]], axis=3)
            else:
                X = Tempo [:, :, 0:M]
        X = tf.multiply(CA, X)
    elif PMv == 'new':
        # X = tf.matmul(CA, np.reshape(y,[M*(N+L-1),1],order='F'))
        X = CA.dot(np.reshape(y,[M*(N+L-1),1],order='F')) #CA es realmente H
        X = np.reshape(X, [1, M, N, L], order='F')
    if norm == "MinMax":
        max = tf.get_static_value(tf.reduce_max(X))
        min = tf.get_static_value(tf.reduce_min(X))
        if max == 0:
            print('0 in Transpose')
        elif max is None:
            print('None in Transpose')
        else:
            X = (X - min) / (max - min)
    return tf.squeeze(X).numpy()

import bm3d
def wrapper_BM3D(inp, sigma):
    # print(np.shape(inp))
    out = np.zeros_like(inp)
    # print(np.shape(out))
    for i in range(np.shape(inp)[2]):
        out [:, :, i] = bm3d.bm3d(inp [:, :, i], sigma * np.ones((1, 16)))
    return out

def denoiser_TBxB(x, denoise, sigma,L):
    x = np.squeeze(x)
    mean = np.mean(x, axis=(0, 1)).reshape(1, 1, L)
    x = x - mean
    min, max = np.min(x, axis=(0, 1)).reshape(1, 1, L), np.max(x, axis=(0, 1)).reshape(1, 1, L)
    # print(np.min(min),np.mean(mean),np.max(max))
    x = (x - min) / (max - min)
    x = denoise(x, sigma)
    x = x * (max - min)
    x = x + min + mean
    # min,mean,max=np.min(x),np.mean(x),np.max(x)
    # print(np.min(min),np.mean(mean),np.max(max))
    return np.expand_dims(x,0)

def RF(img, sigma_s, sigma_r, noise_sigma, num_iterations):
    I = np.array(img).astype(float)

    J = I
    [h, w, num_joint_channels] = np.shape(J)

    # Compute the domain transform(Equation 11 of our paper).
    # Estimate horizontal and vertical partial derivatives using finite differences.

    dIcdx = np.diff(J, n=1, axis=1)
    dIcdy = np.diff(J, n=1, axis=0)

    dIdx = np.zeros([h, w])
    dIdy = np.zeros([h, w])

    # Compute the l1 - norm distance of neighbor pixels.

    for c in range(num_joint_channels):
        dIdx[:, 1:] = dIdx[:, 1:] + abs(dIcdx[:, :, c])
        dIdy[1:, :] = dIdy[1:, :] + abs(dIcdy[:, :, c])

    # XW and SC: include patch smoothing
    # choice_one
    win = 7
    filter = matlab_style_gauss2D((win, win), 2)
    dIdx = scipy.ndimage.correlate(dIdx, filter, mode='constant', origin=-1)
    dIdy = scipy.ndimage.correlate(dIdy, filter, mode='constant', origin=-1)

    # dIdx = cv2.filter2D(dIdx, -1, filter) # borderType=cv2.BORDER_REPLICATE
    # dIdy = cv2.filter2D(dIdy, -1, filter)

    # XW and SC: updated derivative by subtracting noise

    dIdx_noise_sigma = dIdx - noise_sigma
    dIdx_noise_sigma[dIdx_noise_sigma < 0] = 0
    dIdy_noise_sigma = dIdy - noise_sigma
    dIdy_noise_sigma[dIdy_noise_sigma < 0] = 0

    dHdx = (1 + sigma_s / sigma_r * dIdx_noise_sigma)
    dVdy = (1 + sigma_s / sigma_r * dIdy_noise_sigma)

    # The vertical pass is performed using a transposed image.
    dVdy = dVdy.T

    # Perform the filtering.
    N = num_iterations
    F = I

    sigma_H = sigma_s

    for i in range(num_iterations):
        # Compute the sigma value for this iteration (Equation 14 of our paper).
        sigma_H_i = sigma_H * np.sqrt(3) * (2 ** (N - (i + 1))) / np.sqrt((4 ** N) - 1)
        F = TransformedDomainRecursiveFilter_Horizontal(F, dHdx, sigma_H_i)
        F = image_transpose(F)
        F = TransformedDomainRecursiveFilter_Horizontal(F, dVdy, sigma_H_i)
        F = image_transpose(F)

    # F = cast(F,class(img)); # MATLAB

    return F

def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def TransformedDomainRecursiveFilter_Horizontal(I, D, sigma):
    # Feedback coefficient (Appendix of our paper).
    a = np.exp(-np.sqrt(2) / sigma)

    F = I
    V = a**D

    [h, w, num_channels] = np.shape(I)

    # Left -> Right filter.

    for i in range(1, w):
        for c in range(num_channels):
            F[:, i, c] = F[:, i, c] + np.multiply(V[:, i], (F[:, i - 1, c] - F[:, i, c]))

    # Right -> Left filter.
    for i in range(w-2, -1, -1):
        for c in range(num_channels):
            F[:, i, c] = F[:, i, c] + np.multiply(V[:, i+1], (F[:, i+1, c] - F[:, i, c]))
    return F

def image_transpose(I):
    [h, w, num_channels] = np.shape(I)
    T = np.zeros(([w, h, num_channels]))

    for c in range(num_channels):
        T[:, :, c] = I[:, :, c].T

    return T

def wrapper_RF(inp, sigma):
    return RF(inp, 3, sigma, sigma, 3)
    # out = RF(inp, 3, 0.8, 0.8, 3)

def PSNR_Metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, tf.reduce_max(y_true)))

class parameter:
    def __init__(self, rho=1, gamma=1.05, lamnda=0.5, global_iteration=60,show_plots=0,internal_iteration=100,path=''):
        self.rho = rho
        self.gamma = gamma
        self.lamnda = lamnda
        self.global_iteration = global_iteration
        self.show_plots = show_plots
        self.internal_iteration=internal_iteration
        self.path = path

def PnP_proposed(parameters,M,N,L,model,y,xg,l1,lossWeights,freq,denoiser,all,bands,interrupt):
    import os
    best_psnr = -math.inf
    best_ssim = -math.inf
    best_sam = math.inf
    ones_v = np.ones((1, N, M, L))
    u = tf.zeros((1, M, N, L))
    v = tf.zeros((1, M, N, L))
    best_im_psnr = np.squeeze(v)
    best_im_ssim = np.squeeze(v)
    best_im_sam = np.squeeze(v)
    l2 = parameters.lamnda
    rho = parameters.rho
    gamma = parameters.gamma
    general_loss = []
    mapping_loss = []
    forward_loss = []
    x_loss = []
    # xcoords = []
    PSNRs = []
    SSIMs = []
    SAMs = []
    if denoiser == 'RF':
        denoiser = wrapper_RF
    elif denoiser == 'BM3D':
        denoiser = wrapper_BM3D

    cs = ColourSystem(cs="sRGB", bands=bands)
    xg = np.float64(np.squeeze(xg))
    data_range=np.max(xg)-np.min(xg)
    gt = cs.spec_to_rgb(xg)

    
    for i in range(parameters.global_iteration):
        if interrupt and i>0:
            for f in os.listdir(parameters.path[:parameters.path.rfind('/')]):
                if f.startswith(parameters.path[parameters.path.rfind('/')+1:]) and f.endswith('.mat'):
                    return best_im_psnr,best_psnr,best_iter_psnr,best_im_ssim,best_ssim,best_iter_ssim,best_im_sam,best_sam,best_iter_sam,False
        print(i+1)
        # x_old = ones_v
        # v_old = v
        # u_old = u
        # step x
        if l1!=0.0 or all:
            lossNames = ["loss","mapping", "lambda","x_layer"]
            H = model.fit(x=ones_v, y=[y, y, v - u], epochs=parameters.internal_iteration, verbose=0)
            # print(H.history.keys())
            mapping_loss = mapping_loss + (H.history['mapping_loss'])
        else:
            lossNames = ["loss","lambda","x_layer"]
            H =model.fit(x=ones_v, y=[y, v - u], epochs=parameters.internal_iteration, verbose=0)
            # print(H.history.keys())

        epochs_list = np.arange(1, (i+1)*parameters.internal_iteration+1)
        # xcoords.append((i+1)*parameters.internal_iteration)

        general_loss = general_loss+(H.history['loss'])
        forward_loss = forward_loss +(H.history['forward_loss'])
        x_loss = x_loss+(H.history['x_layer_loss'])
        # plt.style.use("ggplot")

        (fig, ax) = plt.subplots(3, 4, figsize=(20, 15))

        min_loss = min(general_loss)
        idx_loss = general_loss.index(min_loss)
        ax[0,0].set_title(f'Total loss ({idx_loss} -> {round(min_loss,12)})')
        ax[0,0].set_xlabel("Epoch #")
        ax[0,0].set_ylabel("Loss")
        ax[0,0].plot(epochs_list, general_loss, label="General loss")
        # for xc in xcoords:
        #     ax[0,0].vlines(x=xc, ymin=min(general_loss), ymax=max(general_loss), colors='green', ls=':', lw=2)
        ax[0,0].legend()
        min_forward = min(forward_loss)
        idx_forward = forward_loss.index(min_forward)
        lw=lossWeights['forward']
        ax[0,1].set_title(f'Total for Forward_CASSI with {lw} ({idx_forward} -> {round(min_forward,12)})')
        ax[0,1].set_xlabel("Epoch #")
        ax[0,1].set_ylabel("Loss")
        ax[0,1].plot(epochs_list, forward_loss, label="Forward loss")
        # for xc in xcoords:
        #     ax[0,1].vlines(x=xc, ymin=min(forward_loss), ymax=max(forward_loss), colors='green', ls=':', lw=2)
        ax[0,1].legend()
        min_x_layer = min(x_loss)
        idx_x_layer = x_loss.index(min_x_layer)
        lw=lossWeights['x_layer']
        ax[0,2].set_title(f'Total for x_layer with {lw} ({idx_x_layer} -> {round(min_x_layer,12)})')
        ax[0,2].set_xlabel("Epoch #")
        ax[0,2].set_ylabel("Loss")
        ax[0,2].plot(epochs_list, x_loss, label="X loss")
        # for xc in xcoords:
        #     ax[0,2].vlines(x=xc, ymin=min(x_loss), ymax=max(x_loss), colors='green', ls=':', lw=2)
        ax[0,2].legend()
        if l1!=0.0 or all:
            min_mapping = min(mapping_loss)
            idx_mapping = mapping_loss.index(min_mapping)
            lw=lossWeights['mapping']
            ax[0,3].set_title(f'Total for mapping network with {lw} ({idx_mapping} -> {round(min_mapping,12)})')
            ax[0,3].set_xlabel("Epoch #")
            ax[0,3].set_ylabel("Loss")
            ax[0,3].plot(epochs_list, mapping_loss, label="Mapping loss")
            # for xc in xcoords:
            #     ax[0,3].vlines(x=xc, ymin=min(mapping_loss), ymax=max(mapping_loss), colors='green', ls=':', lw=2)
            ax[0,3].legend()
        # save the losses figure
        

        x = np.array(model.weights[0])

        # denoising step
        vtilde = x + u
        sigma = math.sqrt(l2 / rho)
        v = denoiser_TBxB(vtilde, denoiser, sigma, L)  # @ {type:"raw"}

        u = u + (x - v)

        vm = np.squeeze(v)
        xf = cs.spec_to_rgb(vm)
        psnr = PSNR_Metric(xg, vm).numpy()
        ssim = SSIM(xg, vm, data_range=data_range)
        sam = SAM(xg, vm)
        PSNRs.append(psnr)
        SSIMs.append(ssim)
        SAMs.append(sam)
        if psnr > best_psnr:
            best_psnr = psnr
            best_im_psnr = vm
            x_psnr = cs.spec_to_rgb(vm)
            best_iter_psnr = i+1
        if ssim > best_ssim:
            best_ssim = ssim
            best_im_ssim = vm
            x_ssim = cs.spec_to_rgb(vm)
            best_iter_ssim = i+1
        if sam < best_sam:
            best_sam = sam
            best_im_sam = vm
            x_sam = cs.spec_to_rgb(vm)
            best_iter_sam = i+1
            
        ax[2,1].set_title('PSNR')
        ax[2,1].set_xlabel("Epoch #")
        ax[2,1].set_ylim([0, 40])
        ax[2,1].plot(np.arange(1, i+2), PSNRs)

        ax[2,2].set_title('SSIM')
        ax[2,2].set_ylim([0, 1])
        ax[2,2].set_xlabel("Epoch #")
        ax[2,2].plot(np.arange(1, i+2), SSIMs)

        ax[2,3].set_title('SAM')
        ax[2,3].set_ylim([0, 1])
        ax[2,3].set_xlabel("Epoch #")
        ax[2,3].plot(np.arange(1, i+2), SAMs)
        
        ax[1,0].imshow(gt), ax[1,0].set_title('GT') # 1,2
        ax[1,1].imshow(x_psnr), ax[1,1].set_title('Best PSNR: '+str(best_iter_psnr)+'  ->  ( '+str(round(PSNRs[best_iter_psnr-1],6))+' ), '+str(round(SSIMs[best_iter_psnr-1],4))+', '+str(round(SAMs[best_iter_psnr-1],4)))
        ax[1,2].imshow(x_ssim), ax[1,2].set_title('Best SSIM: '+str(best_iter_ssim)+'  ->  '+str(round(PSNRs[best_iter_ssim-1],6))+', ( '+str(round(SSIMs[best_iter_ssim-1],4))+' ), '+str(round(SAMs[best_iter_ssim-1],4)))
        ax[1,3].imshow(x_sam), ax[1,3].set_title('Best SAM: '+str(best_iter_sam)+'  ->  '+str(round(PSNRs[best_iter_sam-1],6))+', '+str(round(SSIMs[best_iter_sam-1],4))+', ( '+str(round(SAMs[best_iter_sam-1],4))+' )')

        ax[2,0].imshow(xf), ax[2,0].set_title('Actual: '+str(i+1)+'  ->  '+str(round(psnr,7))+', '+str(round(ssim,5))+','+str(round(sam,5))) # 1,3
        
        if parameters.show_plots == 1:
            plt.show()
        plt.tight_layout()
        if (i+1)%freq==0 or i==0:
            plt.savefig(parameters.path+'.png')
        if i==parameters.global_iteration-1:
            final_name=parameters.path+', '+str(best_psnr)+', '+str(best_ssim)+', '+str(best_sam)+'.png'
            os.rename(parameters.path+'.png',final_name)
            plt.savefig(final_name)
        plt.close()

        # update rho
        rho = rho * gamma

    return best_im_psnr,best_psnr,best_iter_psnr,best_im_ssim,best_ssim,best_iter_ssim,best_im_sam,best_sam,best_iter_sam,True
