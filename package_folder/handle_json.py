import numpy as np
import ast
def extract_name(gender):
    # The quotes says that they are string we will
    # We use the module ast to have extract the list
    gender_item = ast.literal_eval(gender)
    # We iterate inside the list to extract the name
    
    # if the name is empty we return nan
    if len(gender_item) == 0 :
        list_gender_item = np.nan
        return list_gender_item
    else:
        list_gender_item = [gender_item[elt]['name'] for elt in range(len(gender_item))] 
        # Then return the different gender join with '|'
        return '|'.join(list_gender_item)



