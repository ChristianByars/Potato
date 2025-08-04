import splitfolders

input_folder = r"C:\Users\chris\OneDrive\Desktop\SDSU\Pavement Distress\Potato\Potato Disease Dataset(raw)"
output_folder = r"C:\Users\chris\OneDrive\Desktop\SDSU\Pavement Distress\Potato\FinalDataset"

splitfolders.ratio(input_folder, output_folder, seed = 1337, ratio = (.8, .1, .1))