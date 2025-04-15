from pipeline import Pixelsmith, AutoencoderKL, DDIMScheduler
from reproducibility_project.utils import load_dataset
from reproducibility_project.config import DATA_PATH, GEN_IMGS_PATH
import torch
import pickle


seed = 1
generator = torch.manual_seed(seed)
# list of dicts with keys ["image", "caption", "width", "height"]
data = load_dataset()
N = len(data["caption"])
resolutions = [
    2048,
    4096
]

batch_size = 1

already_generated = [p.name for p in GEN_IMGS_PATH.glob("*")]
DEVICE = "cuda"
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to(DEVICE)
pixelsmith = Pixelsmith.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",torch_dtype=torch.float16,vae=vae).to(DEVICE)
pixelsmith.scheduler = DDIMScheduler.from_config(pixelsmith.scheduler.config, set_alpha_to_one=True)

def gen_images(prompts, res):
    return pixelsmith(prompt=prompts,
                negative_prompt=None,
                generator=generator,
                image = None,
                slider = 30,
                guidance_scale=7.5,
                pag_scale=0,
                height = res,
                width = res,
                ).images

def make_gen_img_data(img, idx, res):
    return {"img": img, "idx": idx, "res": res}

def gen_file_name(idx, res):
    return f"{idx}_{res}.pkl"

def save_data(img, idx, res):
    with (GEN_IMGS_PATH / gen_file_name(idx, res)).open('wb') as f:
        pickle.dump(make_gen_img_data(img, idx, res), f)
# what format to save in?

for r in resolutions:
    for i in range(N // batch_size + 1):
        idxs = range(i*batch_size, (i+1)*batch_size)
        
        to_gen_idx = [idx for idx in idxs if gen_file_name(idx, r) not in already_generated]
        if not to_gen_idx: continue
        prompts = [data["caption"][i] for i in to_gen_idx]
        batch_images = gen_images(prompts, r)
        # format and save data
        for img, idx in zip(batch_images, to_gen_idx):
            save_data(img, idx, r)
