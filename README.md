# Install
`conda create -n pixelsmith python=3.11`
`conda activate pixelsmith`
`conda install pytorch=2.2.1 torchvision=0.17.1 pytorch-cuda=12.1 huggingface_hub=0.20.2 -c pytorch -c nvidia -c conda-forge`
`pip install -r requirements.txt`

# Copy to Instance
`rsync -av --info=progress2 <FILES> <USERNAME>@<SERVER-IP>:<REMOTE-PATH>`
`rsync -av --info=progress2 ./ ubuntu@159.54.186.49:/`

# USE
After creating the virtual env, download data: `python get_data.py`, then (assuming that you are running in an environment with a GPU available), you can `python run_experiments.py` to generate the images associated with the captions in the downloaded data. All data is saved in the data folder. Finally, `python evaluate.py` will read the data and evaluate it on KID, FID, IS, and CLIP.