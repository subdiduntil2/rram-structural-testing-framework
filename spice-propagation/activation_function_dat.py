import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os, glob
import re, json
from collections import Counter, defaultdict
import seaborn as sns
import matplotlib.pylab as plt
import itertools
import copy
# from optimization import merge_multiple_groups, group_strings_by_write

def merge_multiple_groups(data, merge_groups):
    # Create a mapping from key to its index for quick lookup.
    key_to_index = {row[0]: i for i, row in enumerate(data)}
    indices_to_remove = set()
    
    # Process each group of keys to merge.
    for group in merge_groups:
        # Ensure that every key in the current group exists.
        if all(key in key_to_index for key in group):
            # Get indices of all keys in the group.
            indices = [key_to_index[key] for key in group]
            # Choose the smallest index as the base row to keep.
            base_index = min(indices)
            # Convert all arrays in the group to boolean before merging.
            bool_arrays = [data[i][1] for i in indices]
            # Merge arrays using element-wise logical OR.
            merged_array = np.array(np.logical_or.reduce(bool_arrays))
            # print("bool are => ",bool_arrays,np.shape(bool_arrays),type(bool_arrays))
            # Update the base row with the merged array.
            data[base_index][1] = merged_array.astype(int)
            # Mark all other indices in this group for removal.
            for i in indices:
                if i != base_index:
                    indices_to_remove.add(i)
    
    # Build a new list excluding rows that were merged away.
    new_data = [row for i, row in enumerate(data) if i not in indices_to_remove]
    # stacked_flattened = np.array([row[1].flatten() for row in new_data])
    # print("stacked flatten ",stacked_flattened,np.shape(stacked_flattened),type(stacked_flattened))
    return new_data

def binary_shift_add(bin1: str, bin2: str, shift: int = 1) -> tuple:
    # Convert binary strings to integers
    int1 = int(bin1, 2)
    int2 = int(bin2, 2)
    # Perform left shift
    shifted = int2 << shift
    # Perform addition
    result_int = int1 + shifted
    # Convert result back to binary string
    result_bin = bin(result_int)
    return result_int, result_bin

# if __name__ == "__main__":
#     result_int, result_bin = binary_shift_add('10111100','11111111')
#     print(f"Integer result: {result_int}")
#     print(f"Binary result: {result_bin}")

def group_strings_by_write(strings):
    groups = defaultdict(list)
    for s in strings:
        key_3 = 0
        # Count zeros excluding the last character
        if "rp_20.0_rn_0.01" in s: key_3 = 1
        elif "rp_0.01_rn_20.0" in s: key_3 = 2
        elif "rp_20_rn_20" in s: key_3=3
        if "ninit_20" in s or "ninit_20.0 " in s: key_2 = 0
        elif "ninit_0" in s or "ninit_0.0" in s: key_2 = 1
        # key = s[-4:-1].count('0')
        # key_2 = s[-1].count('0')
        groups[100*key_3+50*key_2].append(s)
    dict_out = dict(groups)
    return dict_out

def move_none_keys_to_front(d, case_sensitive=True, in_place=False):
    if case_sensitive:
        matcher = lambda k: "_0" in str(k)
    else:
        matcher = lambda k: "_0" in str(k).lower()

    none_keys = [k for k in d if matcher(k)]
    other_keys = [k for k in d if not matcher(k)]

    new = {k: d[k] for k in none_keys + other_keys}

    if in_place:
        d.clear()
        d.update(new)
        return d
    return new

def generate_thermometer_codes(n=8): return ['1' * i + '0' * (n - i) for i in range(n + 1)]

thermometer_codes_8bit = generate_thermometer_codes(8)
combinations = [list(combo) for combo in itertools.product(thermometer_codes_8bit, repeat=4)]
print("shape combos is => ",np.shape(combinations))

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Choose activation function
        if activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)  # Output layer (no activation for regression; use softmax/sigmoid if needed)
        return x

# Sigmoid Activation Function
def sigmoid(x, max_adc=255): return 1 / (1 + np.exp(-x/max_adc))

def to6bit(value: int, method: str = "scale") -> int:
    if not isinstance(value, int): raise TypeError("value must be an integer")
    v = 0 if value < 0 else (255 if value > 255 else value)

    if method.lower() == "truncate": return v >> 2  # same as v // 4
    # scale (rounded): using integer math to avoid floating inaccuracies
    return (v * 63 + 127) // 255

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu_derivative(x):
    return (x > 0).astype(float)

# Tanh Activation Function
def tanh(x, max_adc=255): 
    return np.tanh(x/max_adc)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# xxspot
# ReLU Activation Function
def relu(x):
    x = x-128
    # print("x and xafter",x,to6bit(x))
    return np.maximum(0, x)

#LUT size eq.
def quantized_sigmoid_lut(adc_in, min_val=-5, max_val=5, max_adc=255, num_entries=256):
    # Generate evenly spaced input values
    # print("no quant")
    adc_in = adc_in-128
    inputs = np.linspace(min_val, max_val, num_entries)
    # Compute sigmoid outputs
    outputs = 1 / (1 + np.exp(-inputs))
    # Stack into a lookup table of shape (num_entries, 2)
    lut = np.stack((inputs, outputs), axis=1)
    # lut2 = np.stack((inputs, 1000*outputs), axis=1)
    diffs = np.abs(lut[:, 0] - max_val*(adc_in/max_adc))
    idx = np.argmin(diffs)
    # print("lut sigmoid and size are => ",lut,np.shape(lut))
    # lut[:,1]=np.rint(lut[:, 1] * 1000).astype(int)
    # n_unique = np.unique(lut[:, 1]).size
    # print("lut is => ",lut)
    # print(n_unique)     
    # print("luttt is => ",max_val*(adc_in/max_adc),idx,lut[idx,1],np.shape(lut))
    # if(adc_in==255):print(lut)
    # print("mapping is => ",adc_in, lut[idx,0], 1000*lut[idx,1])
    # print("sigmoid output is => ",adc_in, 1000*lut[idx,1])
    return lut[idx,1]

# def helper_function(value_to_propagate_array,components_string):
#     if components[3]=="0": bulk_string="0"
#     else:
#         if value_to_propagate_array[0]=="0" and value_to_propagate_array[1]=="0": bulk_string="0"
#         elif value_to_propagate[0]=="1" and value_to_propagate[1]=="1": bulk_string="0"
#         else:
#             if value_to_propagate[0]=="1" and value_to_propagate[1]=="0" and components[3]!="0": bulk_string="P"
#             elif value_to_propagate[0]=="0" and value_to_propagate[1]=="1" and components[3]!="0": bulk_string="N"

def column_mapper(value_to_propagate,key_sort):
    # first2 = value_to_propagate[:2]
    # middle = value_to_propagate[2:-2]
    # last2 = value_to_propagate[-2:]
    # value_to_propagate = last2 + middle + first2
    # match = re.search(r'_(\w{4})$', key_sort).group(1)
    match = re.search(r"neighs_(.*?)_ninit", string=key_sort).group(1)
    components = [char for char in match]
    # print("value to prop is => ",value_to_propagate, components)
    # print("what i use => ",components,value_to_propagate)
    #start mapping to the next layer
    if components[3]=="0": bulk_string="0"
    else:
        if value_to_propagate[0]=="0" and value_to_propagate[1]=="0": bulk_string="0"
        elif value_to_propagate[0]=="1" and value_to_propagate[1]=="1": bulk_string="0"
        else:
            if value_to_propagate[0]=="1" and value_to_propagate[1]=="0" and components[3]!="0": bulk_string="P"
            elif value_to_propagate[0]=="0" and value_to_propagate[1]=="1" and components[3]!="0": bulk_string="N"
            else: print("wtf3")
    #search first neigh. value
    if components[0]=="0": first_string="0"
    else:
        if value_to_propagate[2]=="0" and value_to_propagate[3]=="0": first_string="0"
        elif value_to_propagate[2]=="1" and value_to_propagate[3]=="1": first_string="0"
        else:
            if value_to_propagate[2]=="1" and value_to_propagate[3]=="0" and components[0]!="0": first_string="P"
            elif value_to_propagate[2]=="0" and value_to_propagate[3]=="1" and components[0]!="0": first_string="N"
            else: print("wtf0")
    #search second neigh. value
    if components[1]=="0": second_string="0"
    else:
        if value_to_propagate[4]=="0" and value_to_propagate[5]=="0": second_string="0"
        elif value_to_propagate[4]=="1" and value_to_propagate[5]=="1": second_string="0"
        else:
            if value_to_propagate[4]=="1" and value_to_propagate[5]=="0" and components[1]!="0": second_string="P"
            elif value_to_propagate[4]=="0" and value_to_propagate[5]=="1" and components[1]!="0": second_string="N"
            else: print("wtf1")
    #search third neigh. value
    if components[2]=="0": 
        third_string="0"
        # print("mpika components elka",components,value_to_propagate)
    else:
        if value_to_propagate[6]=="0" and value_to_propagate[7]=="0": third_string="0"
        elif value_to_propagate[6]=="1" and value_to_propagate[7]=="1": third_string="0"
        else:
            if value_to_propagate[6]=="1" and value_to_propagate[7]=="0" and components[2]!="0": third_string="P"
            elif value_to_propagate[6]=="0" and value_to_propagate[7]=="1" and components[2]!="0": third_string="N"
            else: print("wtf2")
    #search for initial weight+input
    param_match = re.search(r'rp_\d+\.\d+_rn_\d+\.\d+_inp_\d+\.\d+_inn_\d+\.\d+', key_sort)
    if param_match is None: param_match = re.search(r'rp_\d+_rn_\d+\.\d+_inp_\d+\.\d+_inn_\d+\.\d+', key_sort)
    if param_match is None: param_match = re.search(r'rp_\d+\.\d+_rn_\d+_inp_\d+\.\d+_inn_\d+\.\d+', key_sort)
    if param_match is None: param_match = re.search(r'rp_\d+_rn_\d+_inp_\d+\.\d+_inn_\d+\.\d+', key_sort)
    param_match = param_match.group(0)
    #return all_string 
    all_string = param_match+"_neighs_"+first_string+second_string+third_string+bulk_string+"_ninit_20"  
    return all_string

def value_to_code(v: int, num_bins: int = 32, start: int = 0, end: int = 255) -> int:
    """
    Map a value v in [start…end] to a code in [0…num_bins-1].
    Each bin has equal width: (end - start + 1) / num_bins.
    """
    if not (start <= v <= end):
        raise ValueError(f"value {v} out of range [{start}…{end}]")
    bin_width = (end - start + 1) // num_bins
    return (v - start) // bin_width

# if __name__ == "__main__":
#     # Precompute a lookup table if you like:
#     lookup = [value_to_code(v) for v in range(256)]
#     lookup = np.unique(lookup)
#     lookup_bin = [format(v, '05b') for v in lookup]
#     print("lookup bin ",lookup_bin)
#     new_lookup_bin = []
#     mapping = {'0':'00','1':'10'}
#     for bits in lookup_bin:
#         encoded = ''.join(mapping[c] for c in bits)
#         new_lookup_bin.append(encoded) 
#     print("encoded is => ",new_lookup_bin)
#     # You can also see the full distribution:
#     # count how many values fall in each code
#     counts = [0]*32
#     for v in range(256): counts[value_to_code(v)] += 1
#     # print("\nCounts per code (should all be 8):",lookup_bin)
#     # for code, cnt in enumerate(counts): print(f"  code {code:2d}: {cnt} values")
    
def value_to_therm(value_to_propagate_init): #try out shift-and-add
    value_to_propagate_init=int(value_to_propagate_init)
    output="00000000" #initialize it to all zeros  
    #newnew
    if value_to_propagate_init >= 0   and value_to_propagate_init <= 7:   output = "01010101"
    if value_to_propagate_init >= 8   and value_to_propagate_init <= 15:  output = "01010100"
    if value_to_propagate_init >= 16  and value_to_propagate_init <= 23:  output = "01010001"
    if value_to_propagate_init >= 24  and value_to_propagate_init <= 31:  output = "01010000"
    if value_to_propagate_init >= 32  and value_to_propagate_init <= 39:  output = "01000101"
    if value_to_propagate_init >= 40  and value_to_propagate_init <= 47:  output = "01000100"
    if value_to_propagate_init >= 48  and value_to_propagate_init <= 55:  output = "01000001"
    if value_to_propagate_init >= 56  and value_to_propagate_init <= 63:  output = "01000000"
    if value_to_propagate_init >= 64  and value_to_propagate_init <= 71:  output = "00010101"
    if value_to_propagate_init >= 72  and value_to_propagate_init <= 79:  output = "00010100"
    if value_to_propagate_init >= 80  and value_to_propagate_init <= 87:  output = "00010001"
    if value_to_propagate_init >= 88  and value_to_propagate_init <= 95:  output = "00010000"
    if value_to_propagate_init >= 96  and value_to_propagate_init <= 103: output = "00000101"
    if value_to_propagate_init >= 104 and value_to_propagate_init <= 111: output = "00000100"
    if value_to_propagate_init >= 112 and value_to_propagate_init <= 119: output = "00000001"
    if value_to_propagate_init >= 120 and value_to_propagate_init <= 127: output = "00000000"
    #newnew
    #newnew2
    if value_to_propagate_init >= 128 and value_to_propagate_init <= 135: output = "00000000"
    if value_to_propagate_init >= 136 and value_to_propagate_init <= 143: output = "00000010"
    if value_to_propagate_init >= 144 and value_to_propagate_init <= 151: output = "00001000"
    if value_to_propagate_init >= 152 and value_to_propagate_init <= 159: output = "00001010"
    if value_to_propagate_init >= 160 and value_to_propagate_init <= 167: output = "00100000"
    if value_to_propagate_init >= 168 and value_to_propagate_init <= 175: output = "00100010"
    if value_to_propagate_init >= 176 and value_to_propagate_init <= 183: output = "00101000"
    if value_to_propagate_init >= 184 and value_to_propagate_init <= 191: output = "00101010"
    if value_to_propagate_init >= 192 and value_to_propagate_init <= 199: output = "10000000"
    if value_to_propagate_init >= 200 and value_to_propagate_init <= 207: output = "10000010"
    if value_to_propagate_init >= 208 and value_to_propagate_init <= 215: output = "10001000"
    if value_to_propagate_init >= 216 and value_to_propagate_init <= 223: output = "10001010"
    if value_to_propagate_init >= 224 and value_to_propagate_init <= 231: output = "10100000"
    if value_to_propagate_init >= 232 and value_to_propagate_init <= 239: output = "10100010"
    if value_to_propagate_init >= 240 and value_to_propagate_init <= 247: output = "10101000"
    if value_to_propagate_init >= 248 and value_to_propagate_init <= 255: output = "10101010"
    return output

# def value_to_therm(value_to_propagate_init):
#     value_to_propagate_init=int(value_to_propagate_init)
#     output="00000000" #initialize it to all zeros  
#     # newnew
#     if value_to_propagate_init >= 0   and value_to_propagate_init <= 15:   output = "01010101010101"
#     if value_to_propagate_init >= 16   and value_to_propagate_init <= 31:  output = "00010101010101"
#     if value_to_propagate_init >= 32  and value_to_propagate_init <= 47:   output = "00000101010101"
#     if value_to_propagate_init >= 48  and value_to_propagate_init <= 63:   output = "00000001010101"
#     if value_to_propagate_init >= 64  and value_to_propagate_init <= 79:   output = "00000000010101"
#     if value_to_propagate_init >= 80  and value_to_propagate_init <= 95:   output = "00000000000101"
#     if value_to_propagate_init >= 96  and value_to_propagate_init <= 111:  output = "00000000000001"
#     if value_to_propagate_init >= 112 and value_to_propagate_init <= 127:  output = "00000000000000"
#     #newnew
#     #newnew2
#     if value_to_propagate_init >= 128 and value_to_propagate_init <= 143: output = "00000000000000"
#     if value_to_propagate_init >= 144 and value_to_propagate_init <= 159: output = "00000000000010"
#     if value_to_propagate_init >= 160 and value_to_propagate_init <= 175: output = "00000000001010"
#     if value_to_propagate_init >= 176 and value_to_propagate_init <= 191: output = "00000000101010"
#     if value_to_propagate_init >= 192 and value_to_propagate_init <= 207: output = "00000010101010"
#     if value_to_propagate_init >= 208 and value_to_propagate_init <= 223: output = "00001010101010"
#     if value_to_propagate_init >= 224 and value_to_propagate_init <= 239: output = "00101010101010"
#     if value_to_propagate_init >= 240 and value_to_propagate_init <= 255: output = "10101010101010"
#     return output

def expand_binary(binary_str: str) -> str:
    # binary_str = bin(int(binary_str))
    if "b" in binary_str: binary_str=format(int(binary_str, 0), '08b')
    # if len(binary_str) != 8 or any(ch not in "01" for ch in binary_str):
    #     raise ValueError("Input must be an 8-bit binary string (only '0' and '1').")
    if(int(binary_str,2)>=0 and int(binary_str,2)<128): return "".join("00" if ch == "1" else "01" for ch in binary_str)
    if(int(binary_str,2)>=128 and int(binary_str,2)<=255): return "".join("10" if ch == "1" else "00" for ch in binary_str)
    # return "".join("10" if ch == "1" else "00" for ch in binary_str)

# Example usage:
# print(expand_binary("10101100"))  # Output: "10001010001000"

def propagate_once(dict_in,non_defective=False,sigmoid=True):
    dict_out = copy.deepcopy(dict_in) #initialize output dictionary
    for key,value in dict_in.items(): # for each unique defect location
        if "_0" not in key: #skip non-defective dictionary (doesn't have to change)
            #first neigh
            # column_2 = "rp_20.0_rn_0.01_inp_0.85_inn_0.25_neighs_NNNN"
            # column_2_value = dict_in[key][column_2]
            # column_3 = "rp_20.0_rn_0.01_inp_0.85_inn_0.25_neighs_NNPN"
            # column_3_value = dict_in[key][column_3]
            # #second neigh
            # column_4 = "rp_20.0_rn_0.01_inp_0.85_inn_0.25_neighs_NNN0"
            # column_4_value = dict_in[key][column_4]
            # column_5 = "rp_20.0_rn_0.01_inp_0.85_inn_0.25_neighs_NNP0"
            # column_5_value = dict_in[key][column_5]
            # #third neigh
            # column_6 = "rp_20.0_rn_0.01_inp_0.85_inn_0.25_neighs_NNNP"
            # column_6_value = dict_in[key][column_6]
            # column_7 = "rp_20.0_rn_0.01_inp_0.85_inn_0.25_neighs_NNNP"
            # column_7_value = dict_in[key][column_7]
            #second part of victim cell
            column_8 = "rp_20_rn_20_inp_0.55_inn_0.55_neighs_0000_ninit_20"
            column_8_value = dict_in[key][column_8]
            # print("init print is -> ",key,dict_in["ADC_MAC_1n1m_st_schematic_21"]["rp_20.0_rn_0.01_inp_0.85_inn_0.25_neighs_0000_ninit_20"])
            #bulk
            # column_9 = "rp_20_rn_20_inp_0.55_inn_0.55_neighs_0000"
            # column_9_value = dict_in[key][column_9]
            # column_10 = "rp_20_rn_20_inp_0.55_inn_0.55_neighs_0000"
            # column_10_value = dict_in[key][column_10]
            for key_sort,value_sort in dict_in[key].items(): #for each input (total:2*729)
                #no need to correct shape (done before)
                # print("aaa",key,len(dict_in[key].keys()))
                # print("value sort ",value_sort)
                if non_defective: def_range_temp=0 
                else: def_range_temp=1
                for i in range(def_range_temp+1):
                    # i=i+1
                    #check shape
                    if(sigmoid==False):
                        # value_to_propagate_init=value_sort[i][5]
                        # value_to_propagate_init=value_sort[i][2]
                        # value_to_propagate_init=bin(int(value_sort[i][5]))[2:].zfill(8)
                        # print(value_sort[i][5],value_sort[i][5][:-8])
                        # print("value sort is => ",np.shape(value_sort))
                        # print("value sort is => ",value_sort[i])
                        if "b" not in value_sort[i][5]: 
                            # value_sort[i][5] = value_sort[i][5][:8]
                            value_to_propagate_init=expand_binary(value_sort[i][5][:8])
                        else: 
                            value_to_propagate_init=expand_binary(value_sort[i][5])
                    elif(sigmoid==True): 
                        value_to_propagate_init=expand_binary(bin(int(float(value_sort[i][3])))[:8])
                        # print("im on sigmoid")
                    # value_to_propagate_init=bin(int(float(value_sort[i][3])))[2:].zfill(8)
                    # print("value to prop sigmoid ",value_to_propagate_init)
                    # print("values are => ",value_sort[i][5],value_to_propagate_init)
                    # value_to_propagate = value_to_therm(value_to_propagate_init)
                    column_8_value[0][-1]='01010101'
                    layer_input_values = [column_8_value[0][-1], column_8_value[0][-1], column_8_value[0][-1], column_8_value[0][-1],column_8_value[0][-1], column_8_value[0][-1],
                    value_to_propagate_init, column_8_value[0][-1], column_8_value[0][-1], column_8_value[0][-1]]
                    # if(key_sort=="rp_20.0_rn_0.01_inp_0.85_inn_0.25_neighs_0000_ninit_20" and (key=="ADC_MAC_1n1m_st_schematic_21" or key=="ADC_MAC_1n1m_st_schematic_0")): 
                    #     # print("shift and add reg is => ",key,key_sort,i,value_sort[i][5],value_to_propagate_init,value_sort[i],dict_in[key][key_sort])
                    #     print("init print v2 is -> ",key,dict_in["ADC_MAC_1n1m_st_schematic_21"]["rp_20.0_rn_0.01_inp_0.85_inn_0.25_neighs_0000_ninit_20"])
                    #     print("")
                    shift_and_add_reg = 0
                    for j in range (8): # or 8
                        slice=''.join(s[8-(j+1)] for s in layer_input_values) # change it with a for loop (for 8-bits)
                        # print("slice init is => ",slice)
                        slice="0000"+slice[6]+slice[7]+"0000"
                        key_sort_neutral = "rp_20.0_rn_0.01_inp_0.85_inn_0.25_neighs_PPPP_ninit_20"
                        # print("slice isss => ",slice, key_sort_neutral)
                        all_string = column_mapper(slice,key_sort_neutral)
                        # print("key init and second are => ",all_string,key)
                        # print("all string is => ",dict_in[key][all_string][0])
                        layer_2_value_int = int(dict_in[key][all_string][0][-3],2)
                        # if(key_sort=="rp_20.0_rn_0.01_inp_0.85_inn_0.25_neighs_0000_ninit_20" and (key=="ADC_MAC_1n1m_st_schematic_21" or key=="ADC_MAC_1n1m_st_schematic_0")):
                        #     print("init print v3 is -> ",key,dict_in["ADC_MAC_1n1m_st_schematic_21"]["rp_20.0_rn_0.01_inp_0.85_inn_0.25_neighs_0000_ninit_20"])
                        #     print("")
                        if j == 0: shift_and_add_reg = layer_2_value_int
                        else: shift_and_add_reg += (layer_2_value_int << 1)
                        # else: shift_and_add_reg += (layer_2_value_int)
                    # add previous weights as well (assume previous and next layer same weights)
                    # all_string = column_mapper(value_to_propagate,key_sort)
                    # start mapping to the next layer
                    # give new value
                    # layer_2_value = dict_in[key][all_string]
                    if(i<np.shape(dict_out[key][key_sort])[0]): 
                        if(i<np.shape(dict_in[key][all_string])[0]):
                            # print("layer 2 value => ",layer_2_value[0])
                            # dict_out[key][key_sort][i][2] = layer_2_value[0][2]
                            # dict_out[key][key_sort][i][4] = layer_2_value[0][4]
                            # dict_out[key][key_sort][i][5] = bin(int(layer_2_value[0][2]))[:8]
                            # print("shift and add reg is => ",shift_and_add_reg,dict_out[key][key_sort])
                            dict_out[key][key_sort][i][2] = shift_and_add_reg
                            dict_out[key][key_sort][i][4] = shift_and_add_reg
                            dict_out[key][key_sort][i][5] = bin(int(shift_and_add_reg))[:8]
                            # if(key_sort=="rp_20.0_rn_0.01_inp_0.85_inn_0.25_neighs_0000_ninit_20" and (key=="ADC_MAC_1n1m_st_schematic_21" or key=="ADC_MAC_1n1m_st_schematic_0")): 
                            #     print("init print v4 is -> ",key,shift_and_add_reg,dict_in["ADC_MAC_1n1m_st_schematic_21"]["rp_20.0_rn_0.01_inp_0.85_inn_0.25_neighs_0000_ninit_20"])
                            #     print("")
    return dict_out

model = SimpleNet(input_size=3, hidden_size=5, output_size=1, activation='tanh')
sample_input = torch.randn((2, 3))  # Batch of 2 samples, each with 3 features
output = model(sample_input)
# print("Model output:\n", output)

dir_init = 'For_ETS26_v1' 
df = np.array(pd.read_csv(dir_init+"\mapper.csv"))
print("df is",df)
# df = df[1::2] #=> IF NAN ERROR
df_adc = pd.read_csv("adc_df.csv").to_numpy()
names = list(df[:,0])
ins_adc = list(df[:,1])
values = list(int(df_adc[np.argmin(np.abs(df_adc[:, 0] - elemant)),1]) for elemant in ins_adc)
print("max values are -> ",max(values),min(values))
# print("values are => ",np.shape(values),np.shape(ins_adc))
# values = list(df[:,2])
rp = list(df[:,3])
rn = list(df[:,4])
dir_path = dir_init+'\defective_cells'
num_def_cells = len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))])

# map defective cells to resistive defects
fault_dict_list = []
fault_dict = {}
for filename in glob.glob(os.path.join(dir_init+"/defective_cells",'*.scs')):
    defect_detection_flag = False
    adc_mac_match = re.search(r'ADC_MAC_1n1m_st_schematic_\d+', filename)
    # if adc_mac_match is None: adc_mac_match = re.search(r'CVCO5_V1_\d+', filename)
    # if adc_mac_match is None: adc_mac_match = re.search(r'CVCO5_V3_\d+', filename)
    # if adc_mac_match is None: adc_mac_match = re.search(r'ADC_MAC_1n1m_st_rev_schematic_\d+', filename)
    # if adc_mac_match is None: adc_mac_match = re.search(r'CVCO5_V3_\d+', filename)
    if adc_mac_match is None: continue
    else: 
        adc_mac_match = adc_mac_match.group(0)
        with open(filename, 'r') as f:
            for l_no, line in enumerate(f):
                    if 'r_def' in line:
                        nets_for_defect = list(set(re.findall(r'\((.*?)\)', line)[0].split()))
                        value = line.split("resistor r=",1)[1]
                        # print("adc_match before adding to dict is => ",adc_mac_match)
                        fault_dict[adc_mac_match] = ["r_def",nets_for_defect[0],nets_for_defect[1],int(value)]
                        defect_detection_flag = True
                        break
            if (defect_detection_flag==False): fault_dict[adc_mac_match] = ["r_def","_0","_0",0]
        fault_dict_list.append(fault_dict) 
file = dir_init+'/fault_dict_all.json' 
with open(file, 'w') as f: json.dump(fault_dict, f)

# encoding results from mapper
path = dir_init+"\defective_cells"
dict_faults = dict()
dict_all = dict()
for i,name in enumerate(names):
    adc_mac_match = re.search(r'ADC_MAC_1n1m_st_schematic_\d+', name)
    # if adc_mac_match is None: adc_mac_match = re.search(r'CVCO5_V1_\d+', name)
    # if adc_mac_match is None: adc_mac_match = re.search(r'ADC_MAC_1n1m_st_rev_schematic_\d+', name)
    # if adc_mac_match is None: adc_mac_match = re.search(r'CVCO5_V3_\d+', name)
    # if adc_mac_match is None: continue
    # else:
    adc_mac_match = adc_mac_match.group(0)
    param_match = re.search(r'rp_\d+\.\d+_rn_\d+\.\d+_inp_\d+\.\d+_inn_\d+\.\d+', name)
    res_match = re.findall(r'(?<=_rp_)(\d+\.\d+)|(?<=_rn_)(\d+\.\d+)', name)
    if param_match is None: param_match = re.search(r'rp_\d+_rn_\d+\.\d+_inp_\d+\.\d+_inn_\d+\.\d+', name)
    if param_match is None: param_match = re.search(r'rp_\d+\.\d+_rn_\d+_inp_\d+\.\d+_inn_\d+\.\d+', name)
    if param_match is None: param_match = re.search(r'rp_\d+_rn_\d+_inp_\d+\.\d+_inn_\d+\.\d+', name)
    if res_match == []: res_match = re.findall(r'(?<=_rp_)(\d+)|(?<=_rn_)(\d+)', name)
    if(rp[i]>=24500 and float(res_match[0][0])<=15): rp_flag = "okay"
    elif(rp[i]<24500 and float(res_match[0][0])>15): rp_flag = "okay"
    else: rp_flag = "flip"
    if(rn[i]>=24500 and float(res_match[1][1])<=15): rn_flag = "okay"
    elif(rn[i]<24500 and float(res_match[1][1])>15): rn_flag = "okay"
    else: rn_flag = "flip"    
    param_match = param_match.group(0)
    neighs_match = re.search(r'neighs_\w+', name).group(0)
    ins_temp = ins_adc[i]
    value_temp = values[i]
    operation_finale = param_match+"_"+neighs_match
    # print("fault dict is => ",fault_dict)
    key_finale = adc_mac_match+"_"+fault_dict[adc_mac_match][0]+"_"+fault_dict[adc_mac_match][1]+"_"+fault_dict[adc_mac_match][2]
    key_finale = re.sub(r"_\d+_", "_", key_finale)
    # xxspot
    sigmoid_multiplier=1000 #dec resolution eq.
    value_finale = [operation_finale,fault_dict[adc_mac_match][3],ins_temp,value_temp,sigmoid_multiplier*quantized_sigmoid_lut(value_temp),relu(value_temp),bin(value_temp)[2:].zfill(8),rp_flag,rn_flag]
    # if key_finale in dict_faults: dict_faults[key_finale].append([rp_flag,rn_flag])
    # else: dict_faults[key_finale] = list([[rp_flag,rn_flag]])4
    key_finale = adc_mac_match
    # print("dict key finale is => ",key_finale)
    if key_finale in dict_all: dict_all[key_finale].append(value_finale)
    else: dict_all[key_finale] = list([list(value_finale)])
    # print("value finale is => ",value_finale)
print("dict all are => ",dict_all.keys(),len(dict_all.keys()),np.shape(dict_all))

#grouping dictionaries
dict_final = dict()
for key, value in dict_all.items():
    grouped_data = defaultdict(list)
    for sublist in value:
        key_2 = sublist[0]
        if key_2 in grouped_data: grouped_data[key_2].append(sublist[1:])
        else: grouped_data[key_2] = list([list(sublist[1:])])
    result = [[key] + values for key, values in grouped_data.items()]
    grouped_data = dict(grouped_data)
    dict_final[key]=grouped_data
    if key == "ADC_MAC_1n1m_st_schematic_0": dict_final["ADC_MAC_1n1m_st_rev_schematic_0"] = dict_final[key]

#sorting of results based on string or neighbor scoring system

sorting_alternative = False
dict_final = move_none_keys_to_front(dict_final)
del dict_final["ADC_MAC_1n1m_st_rev_schematic_0"]
print("dict final keys => ",dict_final.keys(),len(dict_final))
for key,value in dict_final.items(): 
    if "schematic_0" in key:
        print("logged in golden",key) 
        key_0 = key
        value_0 = value
    else:
        for subkey_0,subvalue_0 in value_0.items(): #get input from non-defective + stuck-here
            for subkey, subvalue in value.items(): #where to append
                # if "WL_internal" in key and "neighs_0000" in subkey and "neighs_0000" in subkey_0: 
                #     print("golden keys are => ",len(value_0.keys()), len(value.keys()))
                if(subkey_0 == subkey): 
                    # if "WL_internal" in key and "neighs_0000" in subkey and "neighs_0000" in subkey_0: print("matched")
                    subvalue_0 = list(np.squeeze(subvalue_0))
                    dict_final[key][subkey].insert(int(0),subvalue_0)
                    dict_final[key][subkey] = sorted(dict_final[key][subkey], key=lambda x: int(x[0]))
                    # print("dict finale is ",key,subkey,len(dict_final[key]),len(dict_final[key][subkey]))
                    # print("\n")
        if sorting_alternative is True: 
            dict_final_saved = dict_final[key]
            dict_final[key] = dict(sorted(dict_final[key].items()))
        #start global sorting via scoring
        if sorting_alternative is False:
            dict_inputs_sorted = dict()
            for key_sort,value_sort in dict_final[key].items():
                score_temp = 0
                numbers = re.findall(r'\d+\.\d+|\d+', key_sort)
                numbers = [float(num) if '.' in num else int(num) for num in numbers]
                match = re.search(r"neighs_(.*?)_ninit", string=key_sort).group(1)
                components = [char for char in match]
                if numbers[0]>1 and numbers[1]<1 and numbers[2]>0.55 and numbers[3]<0.55:score_temp+=1
                if numbers[0]<1 and numbers[1]>1 and numbers[2]<0.55 and numbers[3]>0.55:score_temp+=1
                if numbers[0]>1 and numbers[1]<1 and numbers[2]<0.55 and numbers[3]>0.55:score_temp-=1
                if numbers[0]<1 and numbers[1]>1 and numbers[2]>0.55 and numbers[3]<0.55:score_temp-=1
                for j in range(3):
                    if(components[j] == "N"):score_temp-=1
                    if(components[j] == "P"):score_temp+=1
                if (components[-1] == "N"): score_temp -= 4
                if (components[-1] == "P"): score_temp += 4
                dict_inputs_sorted[key_sort]=score_temp
            dict_inputs_sorted = dict(sorted(dict_inputs_sorted.items(), key=lambda item: item[1]))
            dict_final[key] = {key_inside: dict_final[key][key_inside] for key_inside in dict_inputs_sorted.keys()} 
        #end global sorting via scoring
# print("dict_final_1 is => ",dict_final["ADC_MAC_1n1m_st_schematic_1"]["rp_20.0_rn_0.01_inp_0.85_inn_0.25_neighs_NNNN_ninit_20"])
        
# Split the keys into chunks of 81
chunk_num = 81
dict_all = []
for key0, value0 in dict_final.items():
    sub_dicts = []
    keys = list(value0.keys())
    for i in range(0, len(keys), chunk_num):
        sub_dict = {key: value[key] for key in keys[i:i + chunk_num]}
        sub_dicts.append(sub_dict)
    # if "_0" in key0: print(sub_dicts[8],np.shape(sub_dicts[8]))
    dict_all.append(sub_dicts)

# fix sizing issue
for key_init, value_init in dict_final.items(): #for each unique defect
    if "_0" not in key_init:
        for key, value in value_init.items(): #for each input
            gap=2-np.shape(value)[0]
            if(gap>0):
                for i in range(gap): value.append(value[-1])
                dict_final[key_init][key]=value
            
#create layer-2-layer mapping
#xxspot
def_range = 1
layer_2_layer = True
num_extra_layers = 1
print("dict final is => ",dict_final.keys())
# if(layer_2_layer): dict_final = propagate_once(propagate_once(propagate_once(propagate_once(dict_final))))
if(layer_2_layer): dict_final = propagate_once(propagate_once(dict_final))
# if(layer_2_layer): dict_final = propagate_once(dict_final)
    # dict_between_2 = propagate_once(dict_between_1)
    # dict_final=dict_between_1
    # print("dict final is => ",dict_final)
    # print("dict final is => ", len(dict_final))
    # print("wrote that MF")
# print("dict_fault_fault is => ",dict_final)

# for plotting sensing map
dict_inputs = dict()
dict_optimization = dict()
dict_chad = dict()
gkaou_sim_counter_v1 = 0
gkaou_sim_counter_v2 = 0
rows_all = 2*729
column_df_names = []
row_df_names = []
#xxspot
flag_to_append = False
output_mode_classify = False
for key_init, value_init in dict_final.items(): #for each unique defect
        # for key, value in value_init.items(): #for each input
        #     print("len keys is => ",value_init.keys(),len(value_init.keys()))
        #     row_df_names.append(key)
            # if (flag_to_append == False and np.shape(row_df_names)[0]<=rows_all): row_df_names.append(key)
    if "_0" not in key_init:
        fault_array = np.empty([len(value_init),def_range])
        row_df_names = list(value_init.keys()) 
        chad_array = np.empty([len(value_init),def_range+1])
        optimization_array = np.empty([len(value_init),def_range])
        dict_to_append = dict()
        rows_counter = 0
        column_df_names.append(key_init+"_10")
        # column_df_names.append(key_init+"_100")
        # column_df_names.append(key_init+"_1000")
        # column_df_names.append(key_init+"_5000")
        # column_df_names.append(key_init+"_10000")
        # column_df_names.append(key_init+"_50000")
        # column_df_names.append(key_init+"_100000")
        # column_df_names.append(key_init+"_500000")
        # column_df_names.append(key_init+"_1000000")
        # column_df_names.append(key_init+"_5000000")
        for key, value in value_init.items(): #for each input
            # if(np.shape(value)[0]>1): 
            #     if value[1][3] == "flip" and value[1][4] == "flip": dict_faults_temp = {key: 2}
            #     elif value[1][3] == "okay" and value[1][4] == "flip": dict_faults_temp = {key: 1}
            #     elif value[1][3] == "flip" and value[1][4] == "okay": dict_faults_temp = {key: 1}
            #     else: dict_faults_temp = {key: 0}
            #     if key_init in dict_faults: dict_faults[key_init].append(dict_faults_temp) 
            #     else: dict_faults[key_init] = list(dict_faults_temp)
            full_sens_flag = True
            for i in range (def_range+1): #for each def strength
                chad_array[rows_counter][i] = value[i][1]
                if(value[i][2]==-1): gkaou_sim_counter_v1+=1
                if(np.shape(value)[0] <= 6): gkaou_sim_counter_v2+=1
                if(i<def_range):
                    # do it as a binary classification over a threshold (0.75 for sigmoid and 0.5 for tanh)
                    if(output_mode_classify==False):
                        to_determine_it_all = abs(float(value[0][2])-float(value[i+1][2]))
                        if value[1][-2] == "flip" and value[1][-1] == "flip": fault_array[rows_counter][0] = 2
                        elif value[1][-2] == "okay" and value[1][-1] == "flip": fault_array[rows_counter][0] = -1
                        elif value[1][-2] == "flip" and value[1][-1] == "okay": fault_array[rows_counter][0] = 1
                        else: fault_array[rows_counter][0] = 0
                        thres=1.0
                        thres_v2=5.0
                        # thres_v3=6.0
                        # sigmoid -> index3 && ReLU -> index4
                        # normal decimal -> index2 && normal binary -> index5
                        # xxspot
                        # print("values again =>",value[i+1])
                        # print("keys are => ",key_init,key)
                        # print("values are => ",len(value)) #need to add ADC model
                        # print("value bef is => ",value)
                        # print("value is => ",value[0])
                        to_determine_it_all = 1*abs(float(value[0][4])-float(value[1][4])) #shape of 6
                        # print("values are =>",value)
                        # if(to_determine_it_all>40):print("gkaou result",key_init,key,value[0],value[1])
                        # to_determine_it_all = 100*abs(float(value[0][3])-float(value[i+1][3]))
                        # to_determine_it_all = abs(value[0][1]-value[i+1][2])
                        # to_determine_it_all = abs(value[0][1]-value[i+1][1])/0.003 #based on analog in only
                        # optimization_array[rows_counter][i] = to_determine_it_all
                        # print("to determine it all is => ",to_determine_it_all)
                        if (to_determine_it_all>=0.0 and to_determine_it_all<thres):
                            full_sens_flag = False
                            optimization_array[rows_counter][i] = to_determine_it_all
                        elif(to_determine_it_all>=thres): 
                            optimization_array[rows_counter][i] = to_determine_it_all
                        # elif(to_determine_it_all>=thres_v2): 
                        #     optimization_array[rows_counter][i] = 2
                        # elif(to_determine_it_all>=thres_v3): 
                        #     optimization_array[rows_counter][i] = 3
                            # optimization_array[rows_counter][i] = to_determine_it_all
                    # do it as a binary classification over a threshold (0.75 for sigmoid and 0.5 for tanh)
                    if(output_mode_classify==True):
                        thres_binary=0.75*sigmoid_multiplier
                        # to_determine_it_all = 1*abs(float(value[0][2])-float(value[1][2])) #shape of 6
                        if(float(value[0][3])>=thres_binary and float(value[1][3])>=thres_binary):
                            full_sens_flag = False
                            optimization_array[rows_counter][i] = 0
                        if(float(value[0][3])<thres_binary and float(value[1][3])<thres_binary):
                            full_sens_flag = False
                            optimization_array[rows_counter][i] = 0
                        if(float(value[0][3])>=thres_binary and float(value[1][3])<thres_binary): 
                            optimization_array[rows_counter][i] = 1
                            # optimization_array[rows_counter][i] = to_determine_it_all
                        if(float(value[0][3])<thres_binary and float(value[1][3])>=thres_binary): 
                            optimization_array[rows_counter][i] = 1
                            # optimization_array[rows_counter][i] = to_determine_it_all
            if(full_sens_flag): dict_to_append[key] = value
            rows_counter+=1
        # print("\n")
        if(flag_to_append == False and len(value_init.keys())==rows_all): flag_to_append = True
        if key_init in dict_to_append:
            dict_inputs[key_init].append(dict_to_append) 
        else:
            dict_inputs[key_init] = list(dict_to_append)
        optimization_array[optimization_array>1000000]=1000000
        dict_optimization[key_init] = optimization_array
        dict_faults[key_init] = fault_array
        dict_chad[key_init] = chad_array
    else: continue    
    
# print("dict_faults is => ",dict_faults)
for key, value in reversed(dict_faults.items()):
    print("mpika!!!!")
    plt.figure()
    sns.heatmap(value)
    plt.title(key)
    plt.savefig(key+"_flip_v3"+".png",format="png")

# print("df opt is => ",dict_optimization,len(dict_optimization))
df_optimization = pd.DataFrame(np.concatenate(list(dict_optimization.values()), axis=1))
df_optimization.columns = column_df_names
# row_df_names = np.unique(row_df_names)
df_optimization.index = row_df_names
#xxspot
result_file = dir_init+"/result_itc_nov_test.xlsx" 
df_optimization.to_excel(result_file)
    
sorting_alternative = True
row_df_names = np.array(row_df_names)
print("df opt is -> ",df_optimization.keys(),len(df_optimization))
print("dict opt is -> ",dict_optimization.keys(),len(dict_optimization))
print("row df names => ",row_df_names,np.shape(row_df_names),type(row_df_names))
dict_wins = dict()
dict_points = dict()
for key, value in reversed(dict_optimization.items()):
    if sorting_alternative is False:
        print("values and keys are => ",key,np.shape(value),type(value),type(row_df_names))
        mask = (value.astype(float) > 0.5).squeeze()
        print("mask is => ",mask)
        print(np.shape(row_df_names[mask]))
        sns.heatmap(value)
        # plt.savefig(key+".png")
        plt.title(key)
        plt.show()
    # for multi-tile plotting
    elif sorting_alternative is True:
        print("mpika sorting alt")
        num_groups = 6
        list_merge = []
        dict_tempp  = dict(zip(row_df_names,value))
        dict_keys_grouped = group_strings_by_write(list(dict_tempp.keys()))
        print("dict temp keys are => ",len(dict_keys_grouped))
        dict_keys_grouped = dict(sorted(dict_keys_grouped.items()))
        fig2, axes2 = plt.subplots(3, 2, figsize=(6, 18), constrained_layout=True)  # Adjust size as needed
        dict_bests = dict()
        counter = 0
        for key_grouped, value_grouped in dict_keys_grouped.items(): #24 groups here -> save/plot after
            dict_tempp_2 = {key_inside:dict_tempp[key_inside] for key_inside in value_grouped} #to uncomment 
            dict_inputs_sorted = dict() #to uncomment
            for key_sort,value_sort in dict_tempp_2.items():
                score_temp = 0
                numbers = re.findall(r'\d+\.\d+|\d+', key_sort)
                numbers = [float(num) if '.' in num else int(num) for num in numbers]
                # print("key sort is => ",key_sort)
                match = re.search(r"neighs_(.*?)_ninit", string=key_sort).group(1)
                components = [char for char in match]
                if numbers[0]>1 and numbers[1]<1 and numbers[2]>0.55 and numbers[3]<0.55:score_temp+=1
                if numbers[0]>1 and numbers[1]<1 and numbers[2]<0.55 and numbers[3]>0.55:score_temp-=1
                for j in range(3):
                    if(components[j] == "N"):score_temp-=1
                    if(components[j] == "P"):score_temp+=1
                if (components[-1] == "N"): score_temp -= 4
                if (components[-1] == "P"): score_temp += 4
                dict_inputs_sorted[key_sort]=score_temp
            dict_inputs_sorted = dict(sorted(dict_inputs_sorted.items(), key=lambda item: item[1]))
            dict_tempp_3 = {key_inside:dict_tempp_2[key_inside] for key_inside in dict_inputs_sorted.keys()} #to uncomment
            dict_ones =  {key: sum(1 for x in arr if x >= 0.5) for key, arr in dict_tempp_3.items()}
            max_count = max(dict_ones.values(), default=0)
            dict_top_3 = {key: count for key, count in dict_ones.items() if count == max_count}
            dict_bests[str(key)+"_"+str(key_grouped)]=max_count
            to_plot = list(dict_tempp_3.values())
            sns.heatmap(to_plot,cmap="grey",ax=axes2[counter//2][counter%2])
            axes2[counter//2][counter%2].set_title(str(key_grouped))
            counter+=1
        # plt.show()
        # for key,value in dict_bests.items(): dict_points[key[-3:]]=dict_points.get(key[-3:],0)+value
        # dict_points =  dict(sorted(dict_points.items(), key=lambda item: item[1], reverse=True))
        # print("dict_bests is => ",dict_bests)
        # print("dict points is => ",dict_points)
        max_count_groups = max(dict_bests.values(),default=0)
        dict_bests_bests =  {key: count for key, count in dict_bests.items() if count == max_count_groups}
        if (max_count_groups>0): 
            for key,value in dict_bests_bests.items(): dict_wins[key[-3:]]=dict_wins.get(key[-3:],0)+1
            dict_wins =  dict(sorted(dict_wins.items(), key=lambda item: item[1], reverse=True))
            # print("dict_bests_bests is => ",dict_bests_bests)
            print("dict_bests is => ",dict_bests_bests)
            print("dict_wins is => ",dict_wins)
        else: 
            print("no winner")
            print("dict_bests is => ",dict_bests_bests)
            print("dict_wins is => ",dict_wins)
        print()
        string_to_Save = key+"_ets_write_merged_binary_v3"+".png"
        #xxspot
        plt.savefig(string_to_Save)
    # plt.savefig(key+"_itc_write_merged_binary_v3"+".png")

# # Tanh Activation Function
# def quantized_tanh_lut(adc_in, min_val=-3, max_val=3, max_adc=255, num_entries=1024):
#     # Generate evenly spaced input values
#     inputs = np.linspace(min_val, max_val, num_entries)
#     # Compute tanh outputs
#     outputs = np.tanh(inputs)
#     # Stack into a lookup table of shape (num_entries, 2)
#     lut = np.stack((inputs, outputs), axis=1)
#     diffs = np.abs(lut[:, 0] - max_val*(adc_in/max_adc))
#     idx = np.argmin(diffs)
#     return lut[idx,1]

# Tanh Activation Function
def quantized_tanh_lut(adc_in, min_val=-2.7, max_val=2.7, max_adc=255, num_entries=33):
    # Generate evenly spaced input values
    inputs = np.linspace(min_val, max_val, num_entries)
    # Compute tanh outputs
    outputs = np.tanh(inputs)
    # Stack into a lookup table of shape (num_entries, 2)
    lut = np.stack((inputs, outputs), axis=1)
    diffs = np.abs(lut[:, 0] - max_val*(adc_in/max_adc))
    idx = np.argmin(diffs)
    return lut[idx,1]

# dict_final_layer = dict_final
# for key,value in dict_final.items(): # for each defect
#     if "_0" not in key: #skip non-defective dictionary
#         for key_sort,value_sort in dict_final[key].items(): #for each input
#             #no need to correct shape (done before)
#             for i in range(def_range):
#                 i=i+1
#                 #check shape
#                 if(np.shape(value_sort)[0]==1): print("keys for single array are: ",key,key_sort)
#                 #start mapping to the next layer
#                 if value_sort[i][5][0]=="0" and value_sort[i][5][1]=="0": bulk_string="0"
#                 elif value_sort[i][5][0]=="1" and value_sort[i][5][1]=="1": bulk_string="0"
#                 elif value_sort[i][5][0]=="1" and value_sort[i][5][1]=="0": bulk_string="P"
#                 elif value_sort[i][5][0]=="0" and value_sort[i][5][1]=="1": bulk_string="N"
#                 #search first neigh. value
#                 if value_sort[i][5][2]=="0" and value_sort[i][5][3]=="0": first_string="0"
#                 elif value_sort[i][5][2]=="1" and value_sort[i][5][3]=="1": first_string="0"
#                 elif value_sort[i][5][2]=="1" and value_sort[i][5][3]=="0": first_string="P"
#                 elif value_sort[i][5][2]=="0" and value_sort[i][5][3]=="1": first_string="N"
#                 #search second neigh. value
#                 if value_sort[i][5][4]=="0" and value_sort[i][5][5]=="0": second_string="0"
#                 elif value_sort[i][5][4]=="1" and value_sort[i][5][5]=="1": second_string="0"
#                 elif value_sort[i][5][4]=="1" and value_sort[i][5][5]=="0": second_string="P"
#                 elif value_sort[i][5][4]=="0" and value_sort[i][5][5]=="1": second_string="N"
#                 #search third neigh. value
#                 if value_sort[i][5][6]=="0" and value_sort[i][5][7]=="0": third_string="0"
#                 elif value_sort[i][5][6]=="1" and value_sort[i][5][7]=="1": third_string="0"
#                 elif value_sort[i][5][6]=="1" and value_sort[i][5][7]=="0": third_string="P"
#                 elif value_sort[i][5][6]=="0" and value_sort[i][5][7]=="1": third_string="N"
#                 #search for initial weight+input
#                 param_match = re.search(r'rp_\d+\.\d+_rn_\d+\.\d+_inp_\d+\.\d+_inn_\d+\.\d+', key_sort)
#                 if param_match is None: param_match = re.search(r'rp_\d+_rn_\d+\.\d+_inp_\d+\.\d+_inn_\d+\.\d+', key_sort)
#                 if param_match is None: param_match = re.search(r'rp_\d+\.\d+_rn_\d+_inp_\d+\.\d+_inn_\d+\.\d+', key_sort)
#                 if param_match is None: param_match = re.search(r'rp_\d+_rn_\d+_inp_\d+\.\d+_inn_\d+\.\d+', key_sort)
#                 param_match = param_match.group(0)
#                 #return all_string 
#                 all_string = param_match+"_neighs_"+first_string+second_string+third_string+bulk_string
#                 #give new value 
#                 layer_2_value = dict_final[key][all_string] 
#                 # print("defect is => ",key)
#                 # print("layer 1 defective is => ",key_sort,value_sort[i][0],value_sort[i][2])
#                 # print("layer 1 non-defective is (just keys) => ",key_sort,value_sort[0][0],value_sort[0][2])
#                 # print("layer 2 observation is => ",all_string,layer_2_value) #map to the non-defective list
#                 if(i<np.shape(dict_final_layer[key][key_sort])[0]): 
#                     if(i<np.shape(layer_2_value)[0]): dict_final_layer[key][key_sort][i]=layer_2_value[i]
# dict_final=dict_final_layer