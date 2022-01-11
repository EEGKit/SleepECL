

## Expert Knowledge Inspired Contrastive Learning for Sleep Staging

#### Environmental requirement:

```
wget==3.2
pyedflib==0.1.22
torch==1.6.0+cu101
scikit-learn==0.24.1
scipy==1.5.4
numpy==1.19.5
```

#### Run the code:

Download the third-party public data set sleep-EDF:

```shell
python download_sleepedf.py
```

Prepare the single-channel EEG data:

```shell
python prepare_sleepedf.py
```

Extract prior features for self-supervised pretraining:

```
python prepare_feature.py
```

Run SleepECL:

```sh
python main.py --window_size 9 --sub_window_size 3 --cnn_level dcc prior --sa_level dcc prior --data_dir ./data/sleepedf/sleep-cassette/eeg_fpz_cz --feature_path ./data/sleepedf/sleep-cassette/feature/eeg_fpz_cz_powerband.pkl
```
