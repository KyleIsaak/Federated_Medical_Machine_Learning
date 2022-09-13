import pathlib

from pydicom import Dataset

from utils import read_img

DATA_DIR = pathlib.Path(__file__).parent / "data"
OUTPUT_DIR = pathlib.Path(__file__).parent / "dicom"

if __name__ == '__main__':
    images = DATA_DIR.rglob("*.jpg")

    #
    for i, path in enumerate(images):
        img = read_img(str(path))

        ds: Dataset = Dataset()
        ds.Rows = img.shape[0]
        ds.Columns = img.shape[1]
        ds.PhotometricInterpretation = "MONOCHROME1"
        ds.BitsStored = 8
        ds.SamplesPerPixel = 1
        ds.BitsAllocated = 8
        ds.HighBit = ds.BitsStored - 1
        ds.PixelRepresentation = 0
        ds.PixelData = img.tobytes()
        ds.DataSetDescription = path.parent.stem
        ds.is_implicit_VR = True
        ds.is_little_endian = True
        ds.save_as(OUTPUT_DIR / f"{i}.dcm")
