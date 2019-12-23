
# Object
Detect Anime character face using YOLO framework.  
Doesn't support classification yet.

# Usage

## Config
Config is set of parameters to controll program outside.  
Written in json file for hierarchy.

### model
| key name          | description                  | format                   |  
| ----------------  | ---------------------------- | ------------------------ |  
| image_size        | input image dimension        | (width, height, channel) |
| anchros           | anchors widht, height        | [[width, height]]        |
| class_count       | not supported                | number of classification |

### dataset
| key name         | description                        | format |
| ---------------- | ---------------------------------- | ------ |
| index            | path to index file                 | string |
| image_dir        | find image at 'image_dir/index'    | string |
| label_dir        | find label at 'label_dir/index'    | string |
| batch_size       | number of batch                    | int    |
| num_workers      | number of cpu use for data loading | int    |

### postprocessing
| key name          | description                        | format |
| ----------------- | ---------------------------------- | ------ |
| conf_threshold    | path to index file                 | float |
| iou_threshold     | find image at 'image_dir/index'    | float |
| acc_iou_threshold | find label at 'label_dir/index'    | float |

### train
| key name                 | description                         | format     |
| ------------------------ | ----------------------------------- | ---------- |
| set                      | dataset use on training             | dataset    |
| plan                     | learning rate / epoch               |            |
| checkpoint               | make checkpoint just in case        |            |
| log                      | enable/disable logging              |            |
| loss                     | hyper parameter used at loss        |            |
| validation               | define validation. see below        | validation |
| enable_anomaly_detection | use anomaly detection for debugging | bool       |

### validation
| key name | description                                    | format         |  
| -------- | ---------------------------------------------- | -------------- |  
| set      | dataset use on validation                      | dataset        |
| target   | target accuracy to be saved                    |                |
| post     | postprocessing parameters use on validation    | postprocessing |

### test
| key name | description                  | format         | 
| -------- | ---------------------------- | -------------- |
| set      | dataset use on test          | dataset        |
| post     | post processing parameters   | postprocessing |
| font     | font size used at plot/video | integer        |
| vide_dir | video dirrectory             | string         |

## Train
 1. Open [train_notebook.ipynb](train_notebook.ipynb)  
 2. Run

## Test
 1. Open [eval_notebook.ipynb](eval_notebook.ipynb)  
 2. Run

# License
Thanks to our [mainly referenced project](#reference), we are using same license.  
[License Link](LICENSE)

# Performance
Using 1000 PSD file, 2500 training images achieved 80% of accuracy.  
Need to be tested more.

# Model
This is a trained model with config.  
 1. Download 'config.json' and 'model.dat' on below link.  
 2. Put 'config.json' on config/

[Download Link](https://drive.google.com/drive/folders/1h6otAg93CjbDy6yuYx-X9OgklHEiX_Q2?usp=sharing)

# Reference
[Yolo Homepage](https://pjreddie.com/darknet/yolo/)  
[Yolo 3 Document](https://pjreddie.com/media/files/papers/YOLOv3.pdf)  
[Referenced project](https://github.com/westerndigitalcorporation/YOLOv3-in-PyTorch)  