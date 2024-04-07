import pandas as pd
from faiss import IndexFlatL2


# Replace 'your_file.csv' with the path to your actual CSV file
df = pd.read_csv('BCR_Asig - Sheet1.csv')
data = df.to_numpy()

import faiss

dimension = data.shape[1]  # Assuming data is your 2D numpy array from the CSV
index = IndexFlatL2(dimension)

# Assuming data is a np.float32 array
index.add(data)

faiss.write_index(index, "your_index.index")