from torcheval.metrics import FrechetInceptionDistance
from torchmetrics.functional.multimodal.clip_score import CLIPScore
from torchmetrics.image.kid import KernelInceptionDistance
#from torchmetrics.image.inception import InceptionScore

from reproducibility_project.utils import load_data, load_gen_data
from torch import Tensor

# TODO: Think about moving stuff to gpu

def load_tensors():
    real_data = load_data()
    gen_data = load_gen_data()

    # want two tensors, lined up
    REAL_IMAGE_KEY = "image"
    real_tensors = [
        d[REAL_IMAGE_KEY]
        for d in real_data
    ]
    
    FAKE_IMAGE_KEY = "img"
    fake_tensors = [
        gen_data[i][FAKE_IMAGE_KEY]
        for i in range(len(real_data))
    ]

    CAPTION_KEY = "text"
    captions = [
        d[CAPTION_KEY]
        for d in real_data
    ]

    # TODO: COnvert to torch tensors

    return real_tensors, fake_tensors, captions

real_tensors, fake_tensors, captions = load_tensors()
N = len(real_tensors)

def calc_frechet():
    BATCH_SIZE = 8
    frech = FrechetInceptionDistance()
    for k in range(N // BATCH_SIZE + 1):
        frech.update(real_tensors[k*BATCH_SIZE:(k+1)*BATCH_SIZE], is_real=True)
        frech.update(fake_tensors[k*BATCH_SIZE:(k+1)*BATCH_SIZE], is_real=False)
    
    return frech.compute()

def calc_clip():
    MODEL = "openai/clip-vit-base-patch16"
    clip = CLIPScore(MODEL)
    BATCH_SIZE = 8
    for k in range(N // BATCH_SIZE + 1):
        start, end = k*BATCH_SIZE, (k+1)*BATCH_SIZE
        clip.update(captions[start:end], fake_tensors[start:end])
    
    return clip.compute()

def calc_kid():
    BATCH_SIZE = 8
    kid = KernelInceptionDistance()
    for k in range(N // BATCH_SIZE + 1):
        kid.update(real_tensors[k*BATCH_SIZE:(k+1)*BATCH_SIZE], real=True)
        kid.update(fake_tensors[k*BATCH_SIZE:(k+1)*BATCH_SIZE], real=False)
    
    return kid.compute()

if __name__ == "__main__":
    print("FRECHET SCORE: ", calc_frechet())
    print("CLIP SCORE: ", calc_clip())
    print("KID SCORE: ", calc_kid())

