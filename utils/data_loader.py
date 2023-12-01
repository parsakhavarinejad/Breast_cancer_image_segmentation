class CustomImageMaskDataset(Dataset):
    def __init__(self, dataframe, image_transform=None):
        self.data = dataframe
        self.image_transform = image_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]['image_path']
        mask_path = self.data.iloc[idx]['mask_path']

        image = Image.open(image_path).convert('L') # I used RGB too, doesn't fix the problem
        mask = Image.open(mask_path).convert('L')

        if self.image_transform:
            # Resize to 240*240 then transforming to Tensor
            image = self.image_transform(image) 
            mask = self.image_transform(mask)
        
        return image, mask