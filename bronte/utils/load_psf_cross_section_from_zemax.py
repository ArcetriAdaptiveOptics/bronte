import numpy as np

def load_position_value_array(filename):
    '''
    Loads the txt file generated via HuygensPSF 2D profile on Zemax
    returns a 2D array containing the position in um and the normalized intensity
    values
    '''
    with open(filename, 'r', encoding='utf-16') as f:
        lines = f.readlines()

    # Find the start of the data section
    for idx, line in enumerate(lines):
        if "Data" in line and "Position" in line and "Value" in line:
            start_idx = idx + 1
            break

    # Parse data lines
    data = []
    for line in lines[start_idx:]:
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                pos = float(parts[1].replace(',', '.'))
                val = float(parts[2].replace(',', '.'))
                data.append([pos, val])
            except ValueError:
                continue  # skip non-numeric lines

    return np.array(data)