import pdb
import sys
import os
import argparse

from tqdm import tqdm
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', type=str, default=None, help='Dir to convert')
parser.add_argument('-f', '--frm', type=str, default='png', help='What to convert from')
parser.add_argument('-t', '--to', type=str, default='jpg', help='What format to convert to')
parser.add_argument('-n', '--name', type=str, default='c', help='Test')
parser.add_argument('-s', '--size', type=int, default=1, help='Down sample size')
parser.add_argument('-m', '--mx_size', type=int, default=200, help='Down sample files larger than:')
args = parser.parse_args()


def convert(old_file: str, new_file: str):
    img = Image.open(old_file)
    w, h = img.size
    if max(w, h) > args.mx_size:
        w, h = w // args.size, h // args.size
    img = img.resize((w, h))
    img.save(new_file)


def main():
    path = args.dir
    par = '/'.join(path.split('/')[:-1])
    folder = path.split('/')[-1]
    print(par, folder)
    new_name = f'{args.name}_x{args.size}_{args.frm[0]}2{args.to[0]}_{folder}'
    new_path = os.path.join(par, new_name)
    print(new_path, flush=True)
    os.makedirs(exist_ok=True, name=new_path)

    list_of_files = list(os.walk(path))
    for root, _, files in tqdm(list_of_files):
        for file in files:
            if file.endswith(f'.{args.frm}'):
                sub_folders = root[len(path) + 1:]  # +1 to exclude first /
                os.makedirs(os.path.join(new_path, sub_folders), exist_ok=True)
                new_file = os.path.join(new_path, sub_folders, f'{file[:-len(args.frm)]}{args.to}')
                old_file = os.path.join(root, file)
                convert(old_file, new_file)

    print(par, folder)


if __name__ == '__main__':
    main()
