""" this is the init.py file for the package 
when importing the package, this file will be executed
so we use this file to expose the classes and functions we want to be available
to the user, this mean we can import the classes and functions from the package
without using the module name
"""
from src.maskcompare import (
    normalize_mask,
    clip_mask,
    cosine_distance,
    l2_distance,
    emd,
    
    

)