# Dynamically Mitigating the Data Discrepancy with Our Proposed Methods for Replay Attack Detection

This repo contains the implementation of our work towards building a more robust replay attack detection system. We propose an informative and  complementary feature representation and leverage a more effective training objective. Experimental results in terms of min-tDCF and EER, as well as more detailed analysis will be reported in this repo.

We are continuously adding comments and refining the repository. If you have some questions, feel free to open an issue:)

## Contents
- [x] source code of proposed methods 
- [x] attack samples for analysis
- [x] model scores of seperate groups
- [x] High-resolution images (in the near future)

## Environment
+ apex   0.1
+ PyTorch  1.1.0
+ sacred 0.7.5
+ Python 3.6+

## Train the model
    python main.py with 'epoch=50' 'lr=0.001'  'load_model=False' 'load_file=results/Model-epoch-25.pth' 'test_first=False' 'num_workers=1' 'eval_mode=False'

## Test the model
    python main.py with 'epoch=50' 'lr=0.001'  'load_model=False' 'load_file=results/models/best-eer-ep36-0.786008.pt' 'test_first=False' 'num_workers=1' 'eval_mode=True' 'server=0' 'train_batch=32' 'GRL_LAMBDA=0.001' 'evalProtocolFile=/data/dyq/anti-spoofing/ASVspoof2019/ASVspoof2019_PA_real/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.real.cm.eval.trl.txt' 'eval_dir=/fast/dyq/ASVspoof2019_PA_real/GDgram_magnitude_1024_400_240'




