# CDTFusion
# README

## Dataset
The dataset is organized in the following hierarchical structure:
```
datasets/
├── fmb/
│   ├── test/
│   │   ├── ir/      # Infrared images
│   │   ├── lbl/     # Label images
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
python fmb/train_step1.py
python fmb/train_step2.py
```

For POS dataset:
```bash
python pos/train_step1.py
python pos/train_step2.py
```

For WHU dataset:
```bash
python whu/train_step1.py
python whu/train_step2.py
```

## Testing
To test the trained models, use the following commands:

For FMB dataset:
```bash
python fmb/test.py
```

For POS dataset:
```bash
python pos/test.py
```

For WHU dataset:
```bash
python whu/test.py
```
