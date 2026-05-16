conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install uv
uv pip install -r requirements.txt
uv pip install wandb
# uv pip install wandb==0.19.11
uv pip install mkl==2024.0.0
uv pip install "numpy<2"
uv pip install openpyxl
uv pip install fvcore
uv pip install msgpack