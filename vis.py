import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image, ImageFilter, ImageEnhance
import sys
from sklearn.decomposition import PCA

'''注：sklearn 和 PIL 库仅用于产生可视化结果和隐含层特征图，不参与任何训练和推理过程'''

def vis(train, val, hp, type='loss', show=False):
    x_axix = np.array([i+1 for i in range(len(train))], dtype=int)
    train_acys = np.array(train)
    test_acys = np.array(val)

    # 开始画图
    plt.title('Result Analysis')
    plt.plot(x_axix, train_acys, color='orange', label='training {}'.format(type))
    plt.plot(x_axix, test_acys, color='skyblue', label='validating {}'.format(type))
    plt.legend()  # 显示图例

    plt.xlabel('epoches')
    plt.ylabel(type)
    plt.grid()
    if not os.path.exists('./results/model_h-nodes{}_lr{}_w{}'.format(hp[0], hp[1], hp[2])):
        os.makedirs('./results/model_h-nodes{}_lr{}_w{}'.format(hp[0], hp[1], hp[2]))
    plt.savefig('./results/model_h-nodes{}_lr{}_w{}/{}.png'.format(hp[0], hp[1], hp[2], type), dpi=400)
    if show== True:
        plt.show()
    plt.close()

def feature(feature_map, hp, save_all=False, smooth=True):
    layer1, layer2 = feature_map
    f = layer1
    i = 0

    if save_all == True:
        print("Save all features!")
        if not os.path.exists('./results/model_h-nodes{}_lr{}_w{}/features'.format(hp[0], hp[1], hp[2])):
            os.makedirs('./results/model_h-nodes{}_lr{}_w{}/features'.format(hp[0], hp[1], hp[2]))

        for f1 in f:
            f1 = f1.reshape(28, -1)
            f1 = Image.fromarray(norm(-f1)*255.)
            f1 = f1.resize((56, 56),Image.NEAREST).resize((112, 112),Image.NEAREST).resize((784, 784),Image.NEAREST).convert('L')
            f1.save('./results/model_h-nodes{}_lr{}_w{}/features/{}.png'.format(hp[0], hp[1], hp[2], str(i)))

            i = i + 1
            r = int(i / f.shape[0] * 100)
            print("\r", end="")
            print("[{}/800]: {}%: ".format(i, r), "|*" * (r // 2), end="")
            sys.stdout.flush()

    print("Making Principal Component Analysis on hidden features.")
    pca = PCA(copy=True, n_components="mle", whiten=False)
    if f.shape[0] > 783:
        f = f[:783,:].transpose(1,0)
    else:
        f = f.transpose(1, 0)
    pca.fit(f)
    print(pca.explained_variance_ratio_)
    f = pca.transform(f).transpose(1,0)

    if not os.path.exists('./results/model_h-nodes{}_lr{}_w{}/features_pca'.format(hp[0], hp[1], hp[2])):
        os.makedirs('./results/model_h-nodes{}_lr{}_w{}/features_pca'.format(hp[0], hp[1], hp[2]))

    for f1 in f:
        f1 = f1.reshape(28, -1)
        f1 = Image.fromarray(norm(-f1)*255.)
        if smooth == True:
            f1 = up_smooth(f1)
        else:
            f1 = f1.resize((784, 784),Image.NEAREST).convert('L')
        f1.save('./results/model_h-nodes{}_lr{}_w{}/features_pca/{}.png'.format(hp[0], hp[1], hp[2], str(i)))

        i = i + 1
        r = int(i / f.shape[0] * 100)
        print("\r", end="")
        print("[{}/{}]: {}%: ".format(i,f.shape[0], r), "*|" * (r // 2), end="")
        sys.stdout.flush()


def norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def up_smooth(x):
    x = x.resize((56, 56), Image.BILINEAR).convert('L')
    x = x.filter(ImageFilter.GaussianBlur(radius=2))
    x = x.resize((112, 112), Image.BILINEAR)
    x = x.resize((784, 784), Image.BILINEAR)

    x = ImageEnhance.Brightness(x)
    x = x.enhance(0.8)
    x = ImageEnhance.Contrast(x)
    x = x.enhance(2.2)
    x = ImageEnhance.Sharpness(x)
    x = x.enhance(3)

    x = x.filter(ImageFilter.GaussianBlur(radius=2))

    return x