import os
from pyhdf.SD import SD, SDC
import numpy as np

hdf_path = "/home/vajira/SRchange/data-raw/albedo/1981/GLASS02B03.V50.A1981161.2021300.hdf"
try:
    hdf = SD(hdf_path, SDC.READ)
    datasets = hdf.datasets()
    print("Datasets in HDF:")
    for name, info in datasets.items():
        print(f"  {name}: {info}")
    
    # Check the first dataset
    first_ds_name = list(datasets.keys())[0]
    data_obj = hdf.select(first_ds_name)
    data = data_obj.get()
    print(f"\nShape of '{first_ds_name}': {data.shape}")
    print(f"Attributes of '{first_ds_name}':")
    for attr_name, attr_val in data_obj.attributes().items():
        print(f"  {attr_name}: {attr_val}")
    
    hdf.end()
except Exception as e:
    print(f"Error: {e}")
