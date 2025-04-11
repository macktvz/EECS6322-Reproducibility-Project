conda create -n pixelsmith python=3.11
conda activate pixelsmith
conda install pytorch=2.2.1 torchvision=0.17.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt