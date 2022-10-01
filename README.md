## What do Vision Transformers Learn? A Visual Exploration

![Visualizations](readme_images/1.png)

### ViT Models:

To visualize the features of the ViT models:

```python
PYTHONPATH=. python experiments/it15/vis35.py -l <layer_number> -f <feature_number> -n  <network_number> -v <tv_coefficient>
```

For example: 

```python
PYTHONPATH=. python experiments/it15/vis35.py -l 4 -f 20 -n  35 -v 0.1
```

### Clip Models:   
To visualize the features of the CLIP models:

```python
PYTHONPATH=. python experiments/it15/vis98.py -l <layer_number> -f <feature_number> -n  <network_number> -v <tv_coefficient>
```

For example: 

```python
PYTHONPATH=. python experiments/it15/vis98.py -l 4 -f 20 -n  98 -v 0.1
```




For the ViT models the -n option should be in [34, 35, 36, 37, 38, 39], 
and for the CLIP models the -n option should be in [94, 95, 96, 97, 98, 99]

To list all the available network numbers use:
```python3
python show_models.py
```

Here we list some of them:

- 34:     ViT0_B_16_imagenet1k
- 35:     ViT1_B_32_imagenet1k
- 36:     ViT2_L_16_imagenet1k
- 37:     ViT3_L_32_imagenet1k
- 38:     ViT4_B_16
- 39:     ViT5_B_32
- 94:     CLIP0_RN50
- 95:     CLIP1_RN101
- 96:     CLIP2_RN50x4
- 97:     CLIP3_RN50x16
- 98:     CLIP4_ViT-B/32
- 99:     CLIP5_ViT-B/16

We use the timm library to load the pretrained models.
After running these commands, you can find the visualizations in the `desktop` folder.


Other experiments done in the paper can be found in the `experiments` folder.

For the experiments that we need to load the imagenet dataset like the isolating CLS experiment, the code 
assumes that the dataset is in data/imagenet/train for the training set, and data/imagenet/val for the validation set.

We will update the readme with more instructions on how to run other experiments soon.
