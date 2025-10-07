# CDTFusion

## Dataset
The dataset is organized in the following structure:
```
datasets/
├── fmb/
│   ├── test/
│   │   ├── ir/      # Infrared images
│   │   ├── lbl/     # Segmentation ground truth
│   │   └── vi/      # Visible images
│   └── train/
│       ├── ir/
│       ├── lbl/
│       └── vi/
├── pos/
│   ├── ...
└── whu/
    ├── ...
```
All images are cropped to a size of 512×512 pixels.

## Model Saving Path
Trained models will be saved in the following directory structure:
```
fusion/
├── fmb/
├── pos/
└── whu/
```

## Training
To train the models for each dataset, use the following commands:

For FMB dataset:
```bash
python fmb_train_step1.py
python fmb_train_step2.py
```

For POS dataset:
```bash
python pos_train_step1.py
python pos_train_step2.py
```

For WHU dataset:
```bash
python whu_train_step1.py
python whu_train_step2.py
```

## Testing
To test the trained models, use the following commands:

For FMB dataset:
```bash
python fmb_test.py
```

For POS dataset:
```bash
python pos_test.py
```

For WHU dataset:
```bash
python whu_test.py
```
