import argparse
import numpy as np
import matplotlib.pyplot as plt

def main(args):
    print(args)
    img = plt.imread(args.image_path)
    plt.imshow(img)
    clicks = plt.ginput(args.n, args.time_out)
    plt.close()
    np.savetxt(args.result_path, clicks, fmt="%s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, default='hw2_material/1.JPG', help='path of image')
    parser.add_argument('--result-path', type=str, default='result.txt', help='path of result')
    parser.add_argument('--n', type=int, default=8, help='click count')
    parser.add_argument('--time-out', type=int, default=300, help='click count')
    args = parser.parse_args()
    main(args)
