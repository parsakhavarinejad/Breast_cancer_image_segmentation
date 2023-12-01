## Dataset

The dataset is available on Kaggle:
[https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset/](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset/)

If you want to try the project, download the dataset and place it inside this folder.
with this structure:

Dataset_BUSI_with_GT/
|-- benign/
|-- malignant/
|-- normal/

### Combining Masks

To facilitate preprocessing, you can use `combine_masks.py` to merge the masks associated with each image. Run the following command:

```bash
python combine_masks.py
```

This script will combine the masks for images that have two separate masks and store the combined masks in the data folder.