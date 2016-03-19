from skimage import io
from skimage import transform
from skimage import filters
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

number_of_thetas = 8 #
frequencies = (0.1,0.2,0.3,0.4,0.5) #determines size of gabor filter

def show_kernels():
    len_of_figure = len(frequencies)
    for j in range(number_of_thetas):
        for i,freq in enumerate(frequencies):
            print(j,i)
            kernel = filters.gabor_kernel(frequency=freq,theta=np.pi/(number_of_thetas)*j)
            ax = plt.subplot(len_of_figure,number_of_thetas,i*number_of_thetas + j+1)
            ax.axis('off')
            plt.imshow(np.pad(kernel.real,((40-kernel.shape[0])//2,(40-kernel.shape[1])//2),'constant'),cmap =cm.Greys_r)
    plt.show()


#images_by_scales = [[],[],[],[],[]]

def process_image(path_to_image, show_image = False):
    print(path_to_image)
    #number_of_thetas = 8 #
    #frequencies = (0.1,0.2,0.3,0.4,0.5)

    #print(path_to_image)
    img = io.imread(path_to_image,as_grey=True)
    img = transform.resize(img,(64,64))

    if show_image:
        plt.figure()
        io.imshow(img)
        plt.figure()

    len_of_figure = len(frequencies)
    images = []
    for i,freq in enumerate(frequencies):
        size_images = []
        for j in range(number_of_thetas):
            real, imag = filters.gabor_filter(img,freq,np.pi/(number_of_thetas)*j)
            image = np.sqrt(imag**2+real**2)
            shape = image.shape
            image = preprocessing.scale(image.flatten()) #zscore
            image = image.reshape(shape)
            image = transform.resize(image,(8,8))
            if show_image:
                ax = plt.subplot(len_of_figure,number_of_thetas,i*number_of_thetas + j+1)
                ax.axis('off')
                plt.imshow(image)
            size_images.append(image)
        combination = np.array(size_images).flatten()
        #print(i)
        images.append(combination.flatten())
        #print(combination.shape)
    if show_image:
        plt.show()

    return images

#process_image('../data/NimStim/23M_AN_O.BMP', True)
#process_image('../data/POFA/aa1-AN-F-14.pgm',True)
#show_kernels()

def all_files_in_dir(path, end):
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(path) if isfile(join(path,f)) if f.lower().endswith(end) ]

    return onlyfiles

def processing_for_parta():
    path = '../data/NimStim/'

    from os.path import join
    onlyfiles = all_files_in_dir(path, '.bmp')
    y = np.array([f.split('_')[0] for f in onlyfiles])
    onlyfiles = [join(path,f) for f in onlyfiles]

    from multiprocessing import Pool
    p = Pool()
    print(onlyfiles)
    x = list(p.map(process_image, onlyfiles))

    print(len(x))
    x = np.array(x)
    print(x.shape)

    shape1 = x[:,0,:]
    print(shape1.shape)

    indexes = np.random.permutation(x.shape[0])
    split_index = int(x.shape[0]*0.75)
    train_ind = indexes[:split_index]
    test_ind = indexes[split_index:]

    from sklearn.decomposition import PCA

    train = []
    test = []
    for i in range(len(frequencies)):
        size_i_train = x[train_ind,i,:]
        size_i_test = x[test_ind,i,:]
        pca = PCA(n_components=8)
        pca.fit(size_i_train)
        print(pca.explained_variance_ratio_.sum())
        train.append(pca.transform(size_i_train))
        test.append(pca.transform(size_i_test))

    X_train = np.concatenate(tuple(train),axis=1)
    X_test = np.concatenate(tuple(test),axis=1)

    "final zscoring"
    scaller = StandardScaler()
    scaller.fit(X_train)
    X_train = scaller.transform(X_train)
    X_test = scaller.transform(X_test)

    y_train = y[train_ind]
    y_test = y[test_ind]
    print(y_train)

    print(X_train.shape)
    print(X_test.shape)

    np.savez('simnim',X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)

def processing_for_partb():
    path = '../data/POFA/'

    from os.path import join
    onlyfiles = all_files_in_dir(path, '.pgm')
    def spliiter(s):
        arr = s.split('-')[:2]
        arr[0] = arr[0][:2]
        return  arr
    y = np.array([spliiter(f) for f in onlyfiles])
    identities = set(y[:,0])
    print(identities)
    import random
    indexes = []
    for identity in identities:
        others =  identities - {identity}
        cv = random.sample(others,1)[0]
        train = np.logical_and(y[:,0]!=identity, y[:,0]!=cv)
        test = (y[:,0]==identity)
        cv = (y[:,0]==cv)
        train_cv = (y[:,0]!=identity)
        indexes.append([train_cv, train, test,cv])

    onlyfiles = [join(path,f) for f in onlyfiles]

    print(y)


    from multiprocessing import Pool
    p = Pool()
    print(onlyfiles)
    x = list(p.map(process_image, onlyfiles))

    print(len(x))
    x = np.array(x)
    print(x.shape)

    shape1 = x[:,0,:]
    print(shape1.shape)

    from sklearn.decomposition import PCA
    for iteration,(train_cv_ind, train_ind, test_ind,cv_ind)in enumerate(indexes):
        train_cv = []
        train = []
        test = []
        cv = []
        for i in range(len(frequencies)):
            size_i_train_cv = x[train_cv_ind,i,:]
            size_i_train = x[train_ind,i,:]
            size_i_test = x[test_ind,i,:]
            size_i_cv = x[cv_ind,i,:]
            pca = PCA(n_components=20)
            pca.fit(size_i_train_cv)
            print(pca.explained_variance_ratio_.sum())
            train.append(pca.transform(size_i_train))
            test.append(pca.transform(size_i_test))
            cv.append(pca.transform(size_i_cv))
            train_cv.append(pca.transform(size_i_train_cv))

        X_train = np.concatenate(tuple(train),axis=1)
        X_test = np.concatenate(tuple(test),axis=1)
        X_cv = np.concatenate(tuple(cv),axis=1)
        X_train_cv = np.concatenate(tuple(train_cv),axis=1)

        "final zscoring"
        scaller = StandardScaler()
        scaller.fit(X_train_cv)
        X_train = scaller.transform(X_train)
        X_test = scaller.transform(X_test)
        X_cv = scaller.transform(X_cv)

        y_train = y[train_ind,1]
        y_test = y[test_ind,1]
        y_cv = y[cv_ind,1]
        print(y_train)

        print(X_train.shape)
        print(X_test.shape)
        print(X_cv.shape)

        np.savez('pofa'+str(iteration),X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,X_cv=X_cv,y_cv=y_cv)

processing_for_partb()

