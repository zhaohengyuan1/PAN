# Testing x2, x3, x4
python test.py -opt options/test/test_PANx2.yml
python test.py -opt options/test/test_PANx3.yml
python test.py -opt options/test/test_PANx4.yml


# Training x2, x3, x4
python train.py -opt options/train/train_PANx2.yml
python train.py -opt options/train/train_PANx3.yml
python train.py -opt options/train/train_PANx4.yml


# Training SRResNet_PA or RCAN_PA
python train.py -opt options/train/train_SRResNet.yml
python train.py -opt options/train/train_RCAN.yml
