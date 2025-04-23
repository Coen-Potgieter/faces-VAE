from PIL import Image
import numpy as np
import os


def num_imgs(path):
    '''
    wowzers, 
        20248 F
        17673 M
        total = 37921
    '''
    return len(os.listdir(path))


def unique_dims(path):
    '''
    EVERY SINGLE IMAGE IS (171,186)
    LETS GOOOOO
    '''
    png_list = os.listdir(path)

    unique_dims = {}

    for png in png_list:
        with Image.open(path + f"/{png}") as img:
            dims = img.size

        if not f"{dims}" in unique_dims.keys():
            unique_dims[f"{dims}"] = 1
        else:
            unique_dims[f"{dims}"] += 1
    return unique_dims


def img_from_file(path, img_num):
    '''
    AND GRAYSCALE -> LOVELY
    '''
    png_list = os.listdir(path)
    with Image.open(path + f"/{png_list[img_num]}") as img:
        img.show()


def to_np(path, npy_path, num_f=None, num_m=None):
    '''
    saved is (example, 171,186, 1) 0-1 float16
    '''
    M_list = os.listdir(path + "/M")
    F_list = os.listdir(path + "/F")

    if num_f is None:
        num_f = len(F_list)
    else:
        F_list = F_list[:num_f]

    if num_m is None:
        num_m = len(M_list)
    else:
        M_list = M_list[:num_m]

    F = np.zeros((num_f, 171, 186, 1))
    for idx, f in enumerate(F_list):
        if idx % 100 == 0:
            print(f"Iteration: {idx}")
        with Image.open(path + f"/F/{f}").convert("L") as img:
            F[idx, :, :, 0] = np.array(img)

    M = np.zeros((num_m, 171, 186, 1))
    for idx, m in enumerate(M_list):
        if idx % 100 == 0:
            print(f"Iteration: {idx}")
        with Image.open(path + f"/M/{m}").convert("L") as img:
            M[idx, :, :, 0] = np.array(img)

    np.random.shuffle(M)
    np.random.shuffle(F)
    X = np.zeros((num_m + num_f, 171, 186, 1))

    X[:num_m, :, :, 0] = M[:, :, :, 0]
    X[num_m:, :, :, 0] = F[:, :, :, 0]

    np.random.shuffle(X)
    X = (X/255).astype(np.float16)
    num_train = 30_000
    X_train = X[:num_train, :, :, :]
    X_dev = X[num_train:, :, :, :]

    np.save(npy_path + "/X_train.npy", X_train)
    np.save(npy_path + "/X_dev.npy", X_dev)


def show_pic(mat,  img_dims):
    '''
    Takes in a 3D array only when the last dimension is of 2,3 or 4
    so, 
        (h,w,1) gets rejected
        (h,w,2) works
        (h,w,3) works
        (h,w,4) works
        (h,w,5) gets rejected
    '''

    img = Image.fromarray(np.uint8(mat))
    scaled_img = img.resize(img_dims, Image.NEAREST)
    scaled_img.show()


def make_small(npy_path, num_train, num_dev):
    X_train, X_dev = np.load(
        npy_path + "/X_train.npy"), np.load(npy_path + "/X_dev.npy")

    np.random.shuffle(X_train)
    np.random.shuffle(X_dev)

    np.save(npy_path + "/X_train.npy", X_train[:num_train])
    np.save(npy_path + "/X_dev.npy", X_dev[:num_dev])


def main():

    print(num_imgs("Assets/Data/Original/F"))
    print(num_imgs("Assets/Data/Original/M"))

    print(unique_dims("Assets/Data/Original/F"))
    print(unique_dims("Assets/Data/Original/M"))

    # img_from_file("Assets/Data/Original/F", -231)
    # img_from_file("Assets/Data/Original/M", -1)

    # to_np("Assets/Data/Original", "Assets/Data/Numpy")

    # make_small("Assets/Data/Numpy", 1000, 500)


if __name__ == "__main__":
    main()
