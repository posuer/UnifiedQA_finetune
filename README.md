# UnifiedQA
https://github.com/allenai/unifiedqa

# Environment
```bash
conda create --name t5 python=3.6.9
conda install cudatoolkit==10.1.243 # set to the version that is compatible to your GPU
conda install cudnn==7.6.5 # set to the version that is compatible to your GPU
pip install tensorflow==2.2.0 # must lower than 2.2.0 (T5 require) and installed by pip 
pip install tensorflow_text==2.2.0 # must same with tensorflow
pip install mesh-tensorflow==0.1.13 # msut set to 0.1.13 (T5 require)
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch  # set to the version that is compatible to your GPU
pip install transformers
pip install google
```

# Files
## Models
Create Google Cloud [Credentials](https://cloud.google.com/docs/authentication/getting-started#creating_a_service_account) and save the json file at current directory.

```bash
export GOOGLE_APPLICATION_CREDENTIALS="[path/to/yours.json]"
conda activate t5
python download_files.py \
    --model_size 11b
    --model_step 1100500
```
[UnifiedQA 11B step 1100500](https://console.cloud.google.com/storage/browser/unifiedqa/models/11B) will be downloaded. 

Note: In the experiments reported in UnifedQA paper, they always used the checkpoint closest to 100k steps (it usually corresponds to checkpoint 1100500)

## Data
SocialiQa data is included in this repo. It is downloaded from [UnifiedQA SocialIQa](https://console.cloud.google.com/storage/browser/unifiedqa/data/social_iqa)

# Fine-tune
```bash
./run.sh
```
Note: if you encounter following error, make sure you import pytorch first.
```bash
ImportError: dlopen: cannot load any more object with static TLS 
```
If your still have this error, run this command first.
```bash
export LD_PRELOAD=path/to/miniconda3/lib/libgomp.so
```