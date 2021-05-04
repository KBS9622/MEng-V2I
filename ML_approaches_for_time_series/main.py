if __name__ == '__main__':
  
  import os
  import multiprocessing
  from utils import load_dataset
  from beautifultable import BeautifulTable as BT

  import torch
  from torch.utils.data import DataLoader
  from torch.utils.data.sampler import SubsetRandomSampler

  comments = True
  cuda = torch.cuda.is_available()
  n_workers = multiprocessing.cpu_count()
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  gpus = True if torch.cuda.device_count() > 1 else False
  mem = False if device == 'cpu' else True
  
  train_set, valid_set, test_set = load_dataset(data_path, dataset, comments=comments)

  train_loader = DataLoader(dataset = train_set.dataset, 
                            sampler=SubsetRandomSampler(train_set.indices),
                            batch_size = batch_size, num_workers=n_workers,
                            pin_memory = mem)

  valid_loader = DataLoader(dataset = valid_set.dataset, 
                            sampler=SubsetRandomSampler(valid_set.indices),
                            batch_size = batch_size, num_workers=n_workers,
                            pin_memory = mem)

  test_loader = DataLoader(dataset = test_set, batch_size = 1,
                           shuffle = False, num_workers=n_workers, pin_memory = mem)