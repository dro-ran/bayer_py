import sys
import json
import argparse
import cv2
import numpy as np

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_image', metavar='input raw image', type=str, help='This option is must need')
    parser.add_argument('-j', '--input_json', metavar='input json config', type=str, help='This option is must need')

    args, text = parser.parse_known_args()

    if (not args.input_image or not args.input_json):
        print('-i and -j option are must needed!')
        sys.exit(1)

    return args.input_image, args.input_json

def json_parse(file_path):
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    return json_data

def sub_black(bayer_nda, black_level):
    # set signed 64bit for overflow support
    bayer_nda = np.array(bayer_nda, dtype=np.float64)
    # clip to u16bit possible value and go back uint16
    bayer_nda - black_level
    bayer_nda = np.clip(bayer_nda, 0, None)
    bayer_nda = np.array(bayer_nda, dtype=np.uint16)
    return bayer_nda

def set_dgain(nda, gain):
    # to float64 for float point calc and overflow support
    nda = np.array(nda, dtype=np.float64)
    nda *= gain

    # clip to u16bit possible value and go back uint16
    nda = np.clip(nda, None, 65535)
    nda = np.array(nda, dtype=np.uint16)

    return nda

# set bayer pattern gain, need to set 2D ndarray shape before call this function.
def set_bayer_gain(bayer_nda, bayer_pattern, b_gain, gb_gain, gr_gain, r_gain):
    # to float64 for float point calc and overflow support
    bayer_nda = np.array(bayer_nda, dtype=np.float64)

    if (bayer_pattern == 'BGGR'):
        bayer_nda[::2, ::2] *= b_gain     # B
        bayer_nda[::2, 1::2] *= gb_gain   # Gb
        bayer_nda[1::2, ::2] *= gr_gain   # Gr
        bayer_nda[1::2, 1::2] *= r_gain   # R
    elif (bayer_pattern == 'RGGB'):
        bayer_nda[::2, ::2] *= r_gain     # R
        bayer_nda[::2, 1::2] *= gr_gain   # Gr
        bayer_nda[1::2, ::2] *= gb_gain   # Gb
        bayer_nda[1::2, 1::2] *= b_gain   # B
    else:
        print('bayer_pattern: ', bayer_pattern ,' is not supported!')

    # go back uint16
    bayer_nda = np.array(bayer_nda, dtype=np.uint16)
    return bayer_nda

def gen_bgr_from_bayer(bayer_nda, config_json):
    bayer_nda = sub_black(bayer_nda, config_json['black_level'])

    # set par cfa gain and overall gain
    bayer_nda = set_bayer_gain(bayer_nda, config_json['bayer_pattern'],
                                config_json['b_dgain'], config_json['gb_dgain'], config_json['gr_dgain'], config_json['r_dgain'])
    bayer_nda = set_dgain(bayer_nda, config_json['post_dgain'])

    # demosaic to BGR image
    img_bgr = cv2.cvtColor(bayer_nda, cv2.COLOR_BAYER_BGGR2BGR)
    return img_bgr

def gray_to_bgr_mosaic(img_gray, bayer_pattern):
    bgr_msc = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    if (bayer_pattern == 'BGGR'):
        # BGR [4096, 4096, 4096] -> B only [4096, 0, 0]
        bgr_msc[::2, ::2, (1, 2)] = 0      # B
        bgr_msc[::2, 1::2, (0, 2)] = 0     # Gb
        bgr_msc[1::2, ::2, (0, 2)] = 0     # Gr
        bgr_msc[1::2, 1::2, (0, 1)] = 0    # R
    elif (bayer_pattern == 'RGGB'):
        # BGR [4096, 4096, 4096] -> R only [0, 0, 4096]
        bgr_msc[::2, ::2, (0, 1)] = 0      # R
        bgr_msc[::2, 1::2, (0, 2)] = 0     # Gr
        bgr_msc[1::2, ::2, (0, 2)] = 0     # Gb
        bgr_msc[1::2, 1::2, (1, 2)] = 0    # B
    else:
        print('bayer_pattern: ', bayer_pattern ,' is not supported!')
    return bgr_msc

def set_bgr_mosaic_gain(bgr_msc, bayer_pattern, b_gain, gb_gain, gr_gain, r_gain):
    bgr_msc = np.array(bgr_msc, dtype=np.float64)
    if (bayer_pattern == 'BGGR'):
        bgr_msc[::2, ::2, 0] *= b_gain      # B
        bgr_msc[::2, 1::2, 1] *= gb_gain    # Gb
        bgr_msc[1::2, ::2, 1] *= gr_gain    # Gr
        bgr_msc[1::2, 1::2, 2] *= r_gain    # R
    elif (bayer_pattern == 'RGGB'):
        bgr_msc[::2, ::2, 2] *= r_gain       # R
        bgr_msc[::2, 1::2, 1] *= gr_gain     # Gr
        bgr_msc[1::2, ::2, 1] *= gb_gain     # Gb
        bgr_msc[1::2, 1::2, 0] *= b_gain     # B
    else:
        print('bayer_pattern: ', bayer_pattern ,' is not supported!')
    bgr_msc = np.clip(bgr_msc, None, 65535)
    bgr_msc = np.array(bgr_msc, dtype=np.uint16)
    return bgr_msc

# gray image to bayer style mosaic image
def bayer_to_bgr_mosaic(bayer_nda, config_json):
    bayer_nda = sub_black(bayer_nda, config_json['black_level'])
    img_gray = cv2.cvtColor(bayer_nda, cv2.COLOR_BAYER_BGGR2GRAY)
    bgr_msc = gray_to_bgr_mosaic(img_gray, config_json['bayer_pattern'])
    bgr_msc = set_bgr_mosaic_gain(bgr_msc, config_json['bayer_pattern'],
                            config_json['b_dgain'], config_json['gb_dgain'], config_json['gr_dgain'], config_json['r_dgain'])
    bgr_msc = set_dgain(bgr_msc, config_json['post_dgain'])
    return bgr_msc

def raw_process(image_path, config_json):
    with open(image_path, 'rb') as raw_data:
        # open raw data and reshape to image size
        raw_nda = np.fromfile(raw_data, np.uint16, config_json['width'] * config_json['height'])
        raw_nda = raw_nda.reshape(config_json['height'], config_json['width'])

        img_bgr = gen_bgr_from_bayer(raw_nda, config_json)
        bgr_msc = bayer_to_bgr_mosaic(raw_nda, config_json)

        cv2.namedWindow('demosaic', cv2.WINDOW_NORMAL)
        cv2.namedWindow('mosaic', cv2.WINDOW_NORMAL)
        cv2.imshow('demosaic', img_bgr)
        cv2.imshow('mosaic', bgr_msc)
        cv2.waitKey()
        cv2.destroyAllWindows()

    return

def main():
    image_path, json_path = arg_parse()
    config_json = json_parse(json_path)
    raw_process(image_path, config_json)

    return

if __name__ == '__main__':
    main()
