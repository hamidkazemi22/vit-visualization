import pdb
import torch 
from tqdm import tqdm
import xmltodict
import os


def translate_obj(cls: str, obj) -> list:
    if isinstance(obj, list):
        return [[v['bndbox']['xmin'], v['bndbox']['xmax'], v['bndbox']['ymin'], v['bndbox']['ymax']] for v in obj if
                v['name'] == cls]
    if obj['name'] == cls:
        v = obj
        return [[v['bndbox']['xmin'], v['bndbox']['xmax'], v['bndbox']['ymin'], v['bndbox']['ymax']]]
    raise NotImplementedError


def objects(expected_class: str, path: str) -> list:
    with open(path, 'r') as f:
        data = f.read()
    xml = xmltodict.parse(data)
    return [translate_obj(expected_class, v) for k, v in xml['annotation'].items() if k == 'object']


def get_path(xml_path: str) -> str:
    return xml_path[:-4] + '.JPEG'


def translate_folder(xml_folder: str, root: str) -> {}:
    parent = os.path.join(root, xml_folder)
    return {f'{get_path(path)}': objects(xml_folder, os.path.join(parent, path)) for path in os.listdir(parent)}


def translate_dataset(root: str, classes: list):
    return {f'{dr}': translate_folder(dr, root) for dr in tqdm(os.listdir(root)) if
            os.path.isdir(os.path.join(root, dr)) and dr in classes}


def main():
    with open('im1knames.txt', 'r') as f:
        im1k_classes = f.read().split('\n')
    dataset = translate_dataset('/cmlscratch/aminjun/Datasets/ImageNetBoxes/Annotation/', im1k_classes)
    torch.save(dataset, 'boxes.pt')


if __name__ == '__main__':
    main()
