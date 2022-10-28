import zipfile

with zipfile.ZipFile("recycling_symbols.zip", 'r') as zip_ref:
    zip_ref.extractall("data/")