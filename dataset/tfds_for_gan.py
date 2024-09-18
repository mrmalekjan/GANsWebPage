import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

class FashionMNIST_For_Gan:
    def __init__(self) -> None:
      self.data_iterator = self.load_dataset().as_numpy_iterator()

    def normalizer(self,data):
        image = data['image']
        return image/255

    def load_dataset(self):
      ds = tfds.load('fashion_mnist', split = 'train')
      ds = ds.map(self.normalizer)
      ds = ds.cache()
      ds = ds.shuffle(60000)
      ds = ds.batch(128)
      ds = ds.prefetch(64)
      return ds

    def visualizer(self, ncols=4):
      fig, ax =plt.subplots(ncols=ncols, figsize=(20,20))
      for idx in range(ncols):
        sample = self.data_iterator.next()
        ax[idx].imshow(np.squeeze(sample['image']))
        ax[idx].title.set_text(sample['label'])