import pandas as pd
from constants import PRODUCTS_DATASET_CSV, PRODUCTS_CATEGORIES_COL

def get_categories_from_descriptions():
    categories = set()
    
    df = pd.read_csv(PRODUCTS_DATASET_CSV)
    descriptions = df[PRODUCTS_CATEGORIES_COL].tolist()
    
    return set(descriptions)
    