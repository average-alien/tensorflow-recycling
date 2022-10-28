import zipfile

with zipfile.ZipFile("filtered-symbols.zip", 'r') as zip_ref:
    zip_ref.extractall("data/")