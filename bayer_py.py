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

def raw_process(image_path, config_json):
    with open(image_path, 'rb') as raw_data:
        raw_nda = np.fromfile(raw_data, np.uint16, config_json['width'] * config_json['height'])
        # set signed 64bit for overflow support
        raw_nda = np.array(raw_nda, dtype=np.int64)
        # sub black and clip to 0
        raw_nda - config_json['black_level']
        raw_nda = np.clip(raw_nda, 0, None)

        # dgain and clip to 65535
        raw_nda *= config_json['dgain']
        raw_nda = np.clip(raw_nda, None, 65535)
        # go back to unsigned 16bit
        raw_nda = np.array(raw_nda, dtype=np.uint16)

        # reshape to image width and height
        raw_nda = raw_nda.reshape(config_json['height'], config_json['width'])

        # demosaic to BGR image
        img_bgr = cv2.cvtColor(raw_nda, cv2.COLOR_BAYER_BGGR2BGR)
        img_gray = cv2.cvtColor(raw_nda, cv2.COLOR_BAYER_BGGR2GRAY)

        # Gray [4096] -> BGR [4096, 4096, 4096]
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        # BGR [4096, 4096, 4096] -> R only [0, 0, 4096]
        img_gray[::2, ::2, (0, 1)] = 0      # R
        img_gray[::2, 1::2, (0, 2)] = 0     # Gr
        img_gray[1::2, ::2, (0, 2)] = 0     # Gb
        img_gray[1::2, 1::2, (1, 2)] = 0    # B

        # resizable window
        cv2.namedWindow('demosaic', cv2.WINDOW_NORMAL)
        cv2.namedWindow('mosaic', cv2.WINDOW_NORMAL)

        cv2.imshow('demosaic', img_bgr)
        cv2.imshow('mosaic', img_gray)
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
