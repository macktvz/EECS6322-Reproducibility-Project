from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore

from reproducibility_project.utils import load_dataset, load_gen_data
from torch import Tensor
import torch
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms import Resize
# TODO: Think about moving stuff to gpu


EVAL_N = 1000
DESIRED_RES = 2048
def load_tensors():
    real_data = load_dataset()
    gen_data = load_gen_data()

    # want two tensors, lined up
    real_tensors = [
        t
        for t in real_data["tensor"]
    ]
    
    FAKE_IMAGE_KEY = "img"
    fake_tensors = [
        pil_to_tensor([d["img"] for d in gen_data if int(d["idx"]) == i and int(d["res"]) == DESIRED_RES][0])
        for i in range(len(real_data["url"]))
    ]

    CAPTION_KEY = "caption"
    captions = [
        c
        for c in real_data[CAPTION_KEY]
    ]
    # make sure these mfs r the right size
    good_inds = [i for i in range(len(real_tensors)) if real_tensors[i].shape[0] == 3]

    real_tensors = [real_tensors[i] for i in good_inds]
    fake_tensors = [fake_tensors[i] for i in good_inds if i < len(fake_tensors)]
    captions = [captions[i] for i in good_inds]

    # TODO: COnvert to torch tensors
    sz = min(EVAL_N, len(fake_tensors))
    return real_tensors[:sz], fake_tensors[:sz], captions[:sz]

real_tensors, fake_tensors, captions = load_tensors()
N = len(real_tensors)

def calc_frechet():
    rsz = Resize((299,299))
    
    rt = torch.stack([rsz.forward(t) for t in real_tensors])
    ft = torch.stack([rsz.forward(t) for t in fake_tensors])
    BATCH_SIZE = 32
    frech = FrechetInceptionDistance()
    for k in range(N // BATCH_SIZE + 1):
        frech.update(rt[k*BATCH_SIZE:(k+1)*BATCH_SIZE], real=True)
        frech.update(ft[k*BATCH_SIZE:(k+1)*BATCH_SIZE], real=False)
        print(f"processed {(k+1)*32} imgs")
    
    return frech.compute()

def calc_is():
    rsz = Resize((299,299))
    
    rt = torch.stack([rsz.forward(t) for t in real_tensors]).to(torch.uint8)
    ft = torch.stack([rsz.forward(t) for t in fake_tensors]).to(torch.uint8)
    BATCH_SIZE = 32
    frech = InceptionScore()
    for k in range(N // BATCH_SIZE + 1):
        frech.update(ft[k*BATCH_SIZE:(k+1)*BATCH_SIZE])
    
    return frech.compute()

def calc_clip():
    MODEL = "openai/clip-vit-base-patch16"
    clip = CLIPScore(MODEL)
    BATCH_SIZE = 32
    for k in range(N // BATCH_SIZE + 1):
        start, end = k*BATCH_SIZE, (k+1)*BATCH_SIZE
        clip.update(captions[start:end], fake_tensors[start:end])
    
    return clip.compute()

def calc_kid():
    rsz = Resize((299,299))

    rt = torch.stack([rsz.forward(t) for t in real_tensors]).to(torch.uint8)
    ft = torch.stack([rsz.forward(t) for t in fake_tensors]).to(torch.uint8)
    BATCH_SIZE = 32
    kid = KernelInceptionDistance(subset_size=50)
    for k in range(N // BATCH_SIZE + 1):
        kid.update(rt[k*BATCH_SIZE:(k+1)*BATCH_SIZE], real=True)
        kid.update(ft[k*BATCH_SIZE:(k+1)*BATCH_SIZE], real=False)
    
    return kid.compute()

from pathlib import Path
import pickle


SAVE_PATH = Path(__file__).parent / "eval"
SAVE_PATH.mkdir(exist_ok=True)
def save_eval(f, c, k, i, n=EVAL_N):
    f_name = SAVE_PATH / f"{n}.pkl"
    with f_name.open("wb") as fl:
        pickle.dump({"frechet":f, "clip":c, "kernel":k, "n":n, "i": i}, fl)

if __name__ == "__main__":
    print(f"EVALUATING {len(fake_tensors)} TENSORS")
    print("FRECHET SCORE: ", f := calc_frechet())
    print("CLIP SCORE: ", c := calc_clip())
    print("KID SCORE: ", k := calc_kid())
    print("I SCORE: ", i := calc_is())
    save_eval(f, c, k, i)

