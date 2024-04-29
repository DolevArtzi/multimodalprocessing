import cv2
import numpy as np
from functools import reduce
from tqdm import tqdm, trange
from scipy.linalg import clarkson_woodruff_transform as CountSketch
import random
from math import floor
from concurrent.futures import ThreadPoolExecutor, as_completed
from matplotlib import pyplot as plt
from scipy.signal import convolve2d


""" MovieComp
A class that can work with mp4 and pngs, with the following capabilities
- mp4 <--> np (4d) <--> mtx
- png <--> np (3d)
- corrupt x% of an image or movie's RGB values
- uncorrupt by averaging neighboring RGB values
    - slow for movies, fast for images
"""
class MovieComp:
    def __init__(self):
        self.base = 'ex_vids/'
        self.vid_name = ''
        self.dims = np.array([])
        self.num_threads = 4
        print('initializing concurrent executor...')
        self.executor = ThreadPoolExecutor(max_workers=self.num_threads)

    """ get_dims
    set self.dims and get the dimensions of a video, given the path to the mp4 file
    """
    def get_dims(self,vid_name):
        cap = cv2.VideoCapture(vid_name)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.dims = np.array([f,h,w,3])
        return self.dims
    
    """ get_dims_img
    set self.dims and get the dimensions of an image, given the path to the png file
    """
    def get_dims_img(self,img_name):
        img = cv2.imread(img_name)
        h, w, _ = img.shape
        self.dims = np.array([h, w, 3])
        print("SETTING DIMS", self.dims)
        return self.dims
        
    """ mp4_to_np
    converts the video referenced by vid_name to a numpy array 

    mp4 --> (# frames, height, width, 3)
    """
    def mp4_to_np(self,vid_name):
        print('converting mp4 to 4d...')
        cap = cv2.VideoCapture(vid_name)

        # get dims from first frame
        ret, frame = cap.read()
        height, width, _ = frame.shape

        frames = np.empty((0, height, width, 3), dtype=np.uint8)
        dims = self.get_dims(vid_name)
        for _ in trange(dims[0]):
        # while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames = np.append(frames, [frame], axis=0)

        cap.release()
        print(f'Converted to numpy')
        print(f'Shape: {frames.shape}')
        # frames x height x width x bgr
        return frames

    """ og_to_mtx
    converts a video from 4d-numpy array to a 2d matrix  

    (# frames, h, w, 3) --> (3 * h * w, # frames)

    NOTE: we transpose to be in line with the standard literature on sketching, where A: n x d, and n >> d
    """
    def og_to_mtx(self,np_movie):
        print('converting 4d to 2d...')
        shape = np_movie.shape
        print(f'Original Shape: {shape}')
        res = np_movie.reshape(-1,reduce(lambda x,y: x*y, shape[1:]))
        res = res.T
        print(f'New Shape: {res.shape}')
        return res

    def og_to_mtx_img(self,np_img):
        print('converting 3d to 2d...')
        shape = np_img.shape
        # self.dims = shape
        print(f'Original Shape: {shape}')
        res = np_img.reshape(shape[1]*shape[2],shape[0])
        # res = res.T
        print(f'New Shape: {res.shape}')
        return res

    """ mtx_to_og
    converts a video in matrix form back to its original shape

    (3 * h * w, # frames) --> (# frames, h, w, 3)
    """
    def mtx_to_og(self,np_movie,original_shape):
        print('converting 2d to 4d...')
        shape = np_movie.shape
        print(f'Modified Shape: {shape}')
        res = np_movie.reshape(original_shape)
        print(f'New Shape: {res.shape}')
        return res

    """ trim_to_3s

    given an n x d matrix, trims it so n x d', where d' is closest multiple of 3 leq. d

    n x d --> n x (d//3 * 3)
    """
    def trim_to_3s(self,np_movie):
        print('converting 2d to 2d (threes)...')
        shape = np_movie.shape
        frames = shape[-1]
        diff = frames % 3 
        print(f'Trimming Movie')
        print(f'Original Shape: {shape}')
        res = np_movie[:,:shape[-1]//3 * 3]
        print(f'New Shape: {res.shape}')
        return res

    """ np_to_mp4
    converts a movie in its original (4d) form in numpy back to mp4
    """
    def np_to_mp4(self,np_movie,out_name='recovered_movie.mp4'):
        print('converting 4d to mp4...')
        num_frames, height, width, _ = np_movie.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_name, fourcc, 30, (width, height))

        for i in tqdm(range(num_frames)):
            frame = np_movie[i]
            out.write(frame)

        out.release()
        print("Video saved successfully.")

    """ png_to_np
    converts the image referenced by img_name to a numpy array 

    png --> (height, width, 3)
    """
    def png_to_np(self,img_name):
        image = cv2.imread(img_name)
        np_img = np.array(image)
        _ = self.get_dims_img(img_name)
        return np_img

    """ np_to_png
    converts an image in its original (3d) form in numpy back to png
    """
    def np_to_png(self,np_img,out_name='recovered_img.png'):
        print('converting 3d numpy array to PNG...')
        cv2.imwrite(out_name, np_img)
        print("PNG image saved successfully.")

    """ cs_transform
    wrapper for CountSketch

    input: A: n x d, rows: the number of rows to transform the matrix into
    output: A': rows x d
    """
    def cs_transform(self,A,rows):
        return CountSketch(A,rows)

    """ reduce_prod
    reduce a numerical list or numpy array with the operator being product
    """
    def reduce_prod(self,arr):
        return reduce(lambda x,y: x*y, arr)

    """ random_corrupt
    randomly choose pixels in a movie (in 4d numpy format) for which 
    we corrupt the color, for percent_corrupted percent of the total pixels

    output: the corrupted movie (4d numpy), a list of indices that were corrupted
    """
    def random_corrupt(self,np_movie,percent_corrupted=.05):
        zeroes = np.array([0,0,0])
        corrupt_list = []
        shape = np_movie.shape
        tot_size = self.reduce_prod(shape)
        print(f'Total Size: {tot_size}')
        to_corrupt = floor(percent_corrupted/100 * tot_size)
        print(f'To Corrupt: {to_corrupt} ({percent_corrupted}%)')
        r = random.randint
        for _ in tqdm(range(to_corrupt)):
            i,j,k = r(0,shape[0]-1),r(0,shape[1]-1),r(0,shape[2]-1)
            corrupt_list.append((i,j,k))
            np_movie[i][j][k] = zeroes
        return np_movie, corrupt_list

    """ random_corrupt_img
    randomly choose pixels in an image (in 3d numpy format) for which 
    we corrupt the color, for percent_corrupted percent of the total pixels

    output: the corrupted image (3d numpy), a list of indices that were corrupted
    """
    def random_corrupt_img(self,np_img,percent_corrupted=.05):
        np_img = np_img.copy()
        zeroes = np.array([0,0,0])
        corrupt_list = []
        shape = np_img.shape
        tot_size = self.reduce_prod(shape)
        print(f'Total Size: {tot_size}')
        to_corrupt = floor(percent_corrupted/100 * tot_size)
        print(f'To Corrupt: {to_corrupt} ({percent_corrupted}%)')
        r = random.randint
        for _ in tqdm(range(to_corrupt)):
            i,j = r(0,shape[0]-1),r(0,shape[1]-1),
            corrupt_list.append((i,j))
            np_img[i][j] = zeroes
        return np_img, corrupt_list

    """ set_name
    sets self.name with the ex_vids base and the video/image name
    """
    def set_name(self,vid_name):
        self.name = self.base + vid_name
        return self.name

    """ corrupt_and_write
    - corrupts the media by percent_corrupted%, and writes the output to local storage
    - works for mp4 and png, depending on whether is_vid is True/False, respectively

    output: the corrupted media (4d/3d numpy for mp4/png) and the list of corrupted pixels
    """
    def corrupt_and_write(self,media_name,percent_corrupted=.05,out_name='',is_vid=True):
        self.set_name(media_name)
        if out_name == '':
            split = self.name.split('/')
            pc = str(percent_corrupted)
            out_name = 'corrupted/' + 'corrupted_' + split[1].split('.')[0] + '_' + ('pt_' + pc.split('.')[1] if '.' in pc else pc) + '.mp4'
        print(f'corrupting {self.name} by {percent_corrupted}%...')
        if is_vid:
            np_movie = self.mp4_to_np(self.name)
            corrupted,corrupt_list = self.random_corrupt(np_movie,percent_corrupted)
            self.np_to_mp4(np_movie,out_name)
        else:
            np_img = self.png_to_np(media_name)
            corrupted,corrupt_list = self.random_corrupt_img(np_img,percent_corrupted)
            self.np_to_png(np_img,out_name)
        print(f'wrote corrupted {"movie" if is_vid else "image"} to {out_name}')
        return corrupted,corrupt_list

    """ get_neighbors
    gets a list of valid indices neighboring the given coords, for a 4d numpy array
    """
    def get_neighbors(self, coords):
        diffs = [-1, 0, 1]
        res = []
        for i in diffs:
            for j in diffs:
                for k in diffs:
                    new_coords = (coords[0] + i, coords[1] + j, coords[2] + k)
                    is_valid = True
                    for idx, dim_size in enumerate(self.dims[:-1]):
                        if not (0 <= new_coords[idx] < dim_size - 1):
                            is_valid = False
                            break
                    if is_valid and new_coords != coords:
                        res.append(new_coords)
        return res

    """ get_neighbors_img
    gets a list of valid indices neighboring the given coords, for a 3d numpy array
    """
    def get_neighbors_img(self, coords,delta=1):
        # diffs = [-1, 0, 1]
        diffs = list(range(-delta,delta+1))
        res = []

        for i in diffs:
            for j in diffs:
                    new_coords = (coords[0] + i, coords[1] + j)
                    is_valid = True
                    for idx, dim_size in enumerate(self.dims[:-1]):
                        if not (0 <= new_coords[idx] < dim_size - 1):
                            is_valid = False
                            break
                    if is_valid and new_coords != coords:
                        res.append(new_coords)
        return res

    """ neighbor_recovery
    given a corrupted video (4d numpy) and a list of 3d indices which were corrupted, attempts to 
    uncorrupt the video by averaging neighboring RGB values
    """
    def neighbor_recovery(self, corrupted, corrupt_list):
        print('recovering video by averaging neighboring RBG values...')
        for (i, j, k) in tqdm(corrupt_list):
            neighbors = self.get_neighbors((i, j, k))
            colors = np.mean([corrupted[i_n, j_n, k_n] for (i_n, j_n, k_n) in neighbors], axis=0)
            corrupted[i, j, k] = colors
        return corrupted

    """ neighbor_recovery_img
    given a corrupted png (3d numpy) and a list of 2d indices which were corrupted, attempts to 
    uncorrupt the video by averaging neighboring RGB values
    """
    def neighbor_recovery_img(self, corrupted, corrupt_list,delta=1):
        print('recovering image by averaging neighboring RBG values...')
        print('corrupted image size ',corrupted.shape)
        for (i, j) in tqdm(corrupt_list):
            neighbors = self.get_neighbors_img((i, j),delta=delta)
            colors = np.mean([corrupted[i_n, j_n] for (i_n, j_n) in neighbors], axis=0)
            corrupted[i, j] = colors
        return corrupted

    """ name_to_mtx
    given the name of an mp4, converts it to a matrix

    dimensions: same as og_to_mtx
    """
    def name_to_mtx(self,vid_name):
        np_movie = self.mp4_to_np(vid_name)
        mtx = self.og_to_mtx(np_movie)
        return mtx

    def name_to_mtx_img(self,img_name):
        self.get_dims_img(img_name)
        np_img = self.png_to_np(img_name)
        mtx = self.og_to_mtx_img(np_img)
        return mtx

    """ color_histogram
    
    given a matrix representing an mp4 video, plot a histogram of the color channels

    TODO: check if this works correctly for corrupted videos. It doesn't currently appear to
    """
    def color_histogram(self,mtx,title=''):
        colors = ['blue','green','red']
        plt.figure(figsize=(7, 3.5))
        for i, color in enumerate(colors):
            plt.subplot(1, 3, i+1)
            plt.hist(mtx[:, i], bins=256, color=color, alpha=0.7)
            plt.title(color.capitalize() + ' Channel')
            plt.xlim(0, 255)
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    """ loss_mse
    compute the mse of two arrays
    """
    def loss_mse(self,arr1, arr2):
        assert arr1.shape == arr2.shape, "arrays must have the same shape"
        squared_diff = (arr1 - arr2)**2
        mse = np.mean(squared_diff)
        return mse

    def loss_mae(self,arr1,arr2):
        return np.mean(np.abs(arr1 - arr2))
    
    def compare_to_corrupted(self,img_name,f,percent_corrupted=.05,display=True):
        np_img = self.png_to_np(img_name)
        corrupted_img, _ = self.random_corrupt_img(np_img,percent_corrupted)
        loss = f(np_img,corrupted_img)
        if display:
            print(f'Loss on {img_name} with {percent_corrupted}% corruption: {loss}')
        return loss

    def plot_loss(self,img_name,percents=[1,3,5,10,15,20,30,40,50,60,70,80,90,93,97,99],loss='mse'):
        if loss == 'mse':
            f = self.loss_mse
        elif loss == 'mae':
            f = self.loss_mae
        else:
            return
        loss_name = loss
        losses = []
        for p in percents:
            loss = self.compare_to_corrupted(img_name,f,p,False)
            losses.append(loss)
        plt.scatter(percents,losses)
        plt.xlabel('Corruption (%)')
        plt.ylabel(f'{loss_name} loss')
        plt.title(f'{loss_name} Loss vs. Percent Corruption of Image')
        plt.show()
        
    def add_padding(self,array):
        padded_array = np.pad(array, 1, mode='constant', constant_values=0)
        return padded_array

    def apply_kernel(self,image, kernel):
        padded_img = self.add_padding(image)
        return convolve2d(padded_img,kernel,mode='valid')
    
    def sharpen_img(self,img_name):
        mtx_img = self.name_to_mtx_img(img_name)
        laplacian_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        laplacian_kernel_2 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        res_mtx = self.apply_kernel(mtx_img,laplacian_kernel_2)
        return res_mtx.reshape(self.dims)
