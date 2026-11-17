"""把比對圖切成一列一張（方便檢視）。"""
import sys
from PIL import Image
im = Image.open(sys.argv[1]); h = 662 + 4
for r in range((im.height + 4) // h):
    im.crop((0, r * h, im.width, min(im.height, r * h + 662))).save(sys.argv[1].replace(".png", f"_row{r}.png"))
