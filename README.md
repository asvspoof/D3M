# Dynamically Mitigating Data Discrepancy with Balanced Focal Loss for Replay Attack Detection

This repo contains the implementation of our work towards building a more robust replay attack detection system. We propose an informative and complementary feature representation and leverage a more effective training objective. Experimental results in terms of min-tDCF and EER, as well as more detailed analysis will be reported in this repo. 

Source code and other details for replay attack detection, tested on ASVspoof2019 PA and Real-PA dataset.

We are continuously adding comments and refining the repository. If you have some questions, feel free to open an issue:)

## Contents
- [x] source code of proposed methods 
- [x] attack samples for analysis
- [x] model scores of seperate groups
- [x] High-resolution images (in the near future)

## Environment
+ apex   0.1 (for mixed precision training)
+ PyTorch  1.1.0 (DL framework)
+ sacred 0.7.5 (record experimental details)
+ Python 3.6+ 

To install most dependencies automatically:

    pip install -r requirements.txt

## Train the model
    python main.py with 'epoch=50' 'lr=0.001'  'load_model=False' 'load_file=results/Model-epoch-25.pth' 'test_first=False' 'num_workers=1' 'eval_mode=False'

## Test the model
    python main.py with 'epoch=50' 'lr=0.001'  'load_model=False' 'load_file=results/models/best-eer-ep36-0.786008.pt' 'test_first=False' 'num_workers=1' 'eval_mode=True' 'server=0' 'train_batch=32' 'GRL_LAMBDA=0.001' 'evalProtocolFile=/data/to/anti-spoofing/ASVspoof2019/ASVspoof2019_PA_real/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.real.cm.eval.trl.txt' 'eval_dir=/data/to/ASVspoof2019_PA_real/GDgram_magnitude_1024_400_240'

## Use scared for experiment management
We use [scared](https://github.com/IDSIA/Sacred) to manage our experiments, and you can create a file named `myexp.py` with your own configurations.
For instance, 

    from sacred import Experiment
    from sacred.observers import MongoObserver
    from sacred.utils import apply_backspaces_and_linefeeds

    ex = Experiment("ASVSPOOF2019")
    ex.observers.append(MongoObserver.create(
        url='mongodb://exp:user@yourip:port/sacred?authMechanism=SCRAM-SHA-1',
        db_name='sacred'))
    ex.captured_out_filter = apply_backspaces_and_linefeeds


## Citation
If you find this work helpful, please cite it in your publications.

    @misc{dou2020dynamically,
    title={Dynamically Mitigating Data Discrepancy with Balanced Focal Loss for Replay Attack Detection},
    author={Yongqiang Dou and Haocheng Yang and Maolin Yang and Yanyan Xu and Dengfeng Ke},
    year={2020},
    eprint={2006.14563},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
    }
