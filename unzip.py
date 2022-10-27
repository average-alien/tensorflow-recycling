import zipfile

with zipfile.ZipFile("data/recycling_symbols-20221026T203810Z-001.zip", 'r') as zip_ref:
    zip_ref.extractall("data/")