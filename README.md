# SuperReolution_Face-Recognition
Recognizing faces from low resolution images.
## Super Resolution

### ESRGAN (SOTA)
1. Add your low resolution images to ESRGAN/LR folder
2. Go to ESRGAN/
3. python test.py
4. Output will be in ESRGAN/output folder

### Idealo repository fork. (pre-SOTA)
1. pip install ISR
2.
```import numpy as np
from PIL import Image
img = Image.open('Image_path')
lr_img = np.array(img)```

```python
from ISR.models import RDN

rdn = RDN(weights='model_to_use')
sr_img = rdn.predict(lr_img)
Image.fromarray(sr_img)

```

Currently 4 models are available:

RDN: psnr-large, psnr-small, noise-cancel
RRDN: gans

### Face Detection and Recognition

1. Follow the facenet_pytorch/examples/infer notebook for complete pipeline.


