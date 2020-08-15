# UnifiedQA
https://github.com/allenai/unifiedqa

# Environment
```bash
conda create -name t5 python==3.6.9
conda install cudatoolkit==10.1.243 # set to the version that is compatible to your GPU
conda install cudnn==7.6.5 # set to the version that is compatible to your GPU
pip install tensorflow==2.2.0 # must lower than 2.2.0 (T5 require) and installed by pip 
pip install tessorflow_text==2.2.0 # must same with tensorflow
pip install mesh-tensorflow==0.1.13 # msut set to 0.1.13 (T5 require)
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch  # set to the version that is compatible to your GPU
pip install t5[gcp]
```

# Files
## Models

Download following four files from [UnifiedQA 11B](https://console.cloud.google.com/storage/browser/unifiedqa/models/11B) and put into models/unifiedqa_trained/
```bash
model.ckpt-1100500.meta
model.ckpt-1100500.index
model.ckpt-1100500.data-00001-of-00002
model.ckpt-1100500.data-00002-of-00002
```
Note: In the experiments reported in UnifedQA paper, they always used the checkpoint closest to 100k steps (it usually corresponds to checkpoint 1100500)

## Data
SocialiQa data is included in this repo. It is downloaded from [UnifiedQA SocialIQa](https://console.cloud.google.com/storage/browser/unifiedqa/data/social_iqa)

# Fine-tune
```bash
conda activate t5
python fine-tune.py \
    --model_path models/unifiedqa_trained/models_11B_model.ckpt-1100500.index #change model path accordingly if you use other model
```
Note: if you encounter following error, make sure you import pytorch first.
```bash
ImportError: dlopen: cannot load any more object with static TLS 
```
If your still have this error, run this command first.
```bash
export LD_PRELOAD=path/to/miniconda3/lib/libgomp.so
```
