# Transformer Based Global+Local Fingerprint Matching
Fingerprint feature extraction is a task that is solved using either a global or a local representation. State-of-the-art global approaches use heavy deep learning models to process the full fingerprint image at once, which makes the corresponding approach memory intensive. On the other hand, local approaches involve minutiae based patch extraction, multiple feature extraction steps and an expensive matching stage, which make the corresponding approach time intensive. However, both these approaches provide useful and sometimes exclusive insights for solving the problem. Using both approaches together for extracting fingerprint representations is semantically useful but quite inefficient. Our convolutional transformer based approach with an in-built minutiae extractor provides a time and memory efficient solution to extract a global as well as a local representation of the fingerprint. The use of these representations along with a smart matching process gives us state-of-the-art performance across multiple databases.

## Architecture
The following is the architecture used by us:

<img src="./figs/arch.png" alt="architecture" width="400"/>


<!-- ## Setup
* The `requirements.txt` file can be used to setup a virtual environment.
```
pip install -r requirements.txt
```
* The imagenet model for CvT is used to initialize our training. Download the model from [here](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/saraansh_tandon_research_iiit_ac_in/EriTwfzu6e5AmS_-VPSLt48BW1HX0IilyWbUm5KBZWmcSw?e=akMI5c) and place it in the `pretrained/` folder.
* The `MSU-LatentAFIS` folder is built upon the [MSU-LatentAFIS](https://github.com/prip-lab/MSU-LatentAFIS/tree/1d6e837651a1b5dac3bd48d672397f620bf9a0a5) repository. Hence to use it the setup described in the original repo will have to be performed separately.

## Data

To train/validate our approach we need:

* Segmented images of single fingerprints. For this we can use the `tools/augmenter.py` file.
```
python tools/augmenter.py --input_dir <> --output_dir <> --segment
```
* Corresponding Global representation obtained from a teacher. The dataloader expects a `.npy` file corresponding to each image containing a 192 dimensional numpy array. We have used DeepPrint in our paper, but it can't be shared due to its propreitary nature.
* Corresponding minutiae points obtained from a minutiae extractor and a corresponding set of local representations from a teacher. The dataloader expects a '.npy' file corresponding to each image containing a dictionary with all the required local information. This can be obtained using the `MSU-LatentAFIS` folder. In our work we have used Verifinger minutiae points but we can't share that as it is a proprietary software. The following command can be used to obtain local embeddings using LatentAFIS' in-built minutiae detector.
```
python extraction/pipeline.py --gpu 0 \
--mode gt --patch_types 1 --data_format all_in_one \
--input_dir <> --output_dir <> \
--minu_from afis;
```

Make three separate folders for each of the above and maintain the same folder structure in each of them.

## Train
To train our models we use the `submit.sh` file. This takes a configuration file as input and also allows in-line parameter assignment. Images dir correspond to the first point in the Data section, Global embs dir corresponds to the second point in the Data section, and Local embs dir corresponds to the third point in the Data section.
```
bash run.sh -g <# gpus> -t train --cfg <configuration file> \
OUTPUT_DIR <global model save dir> \
DATASET.TRAIN_IMGS <train images dir> \
DATASET.TRAIN_GLOBAL_EMBS <train global embs dir> \
DATASET.TRAIN_TOKEN_EMBS <train local embs dir> \
DATASET.VAL_IMGS <validation images dir> \
DATASET.VAL_GLOBAL_EMBS <validation global embs dir> \
DATASET.VAL_TOKEN_EMBS <validation local embs dir> \
```
To perform different types of training just change the configuration file or the in-line parameters.


##### Global
Use configuration file `experiments/global.yaml`. This will train the model to learn only global embedding extraction.

##### Local
Use configuration file `experiments/local.yaml`. This will train the model to learn minutiae extraction and corresponding local embedding extraction.


## Embedding Extraction
```
bash run.sh -g 1 -t inference --cfg <config file> \
TEST.MODEL_FILE <model .pth file> \
OUTPUT_DIR <embs save dir> \
DATASET.VAL_DATASET inference \
DATASET.VAL_IMGS <inference images dir> \
DATASET.VAL_GLOBAL_EMBS <inference global embs dir> \
DATASET.VAL_TOKEN_EMBS <inference local embs dir>
```
Change the configuration file similar to the Train section to extract embeddings from different types of models.
**Note:** For Global+Local models,  a single embedding extraction step would generate both global and local embeddings.

## Matching
##### Global
This requires simple cosine distance computation for each fingerprint pair as the global embeddings are fixed-length vectors. The embeddings are fetched from the `global_embs.npy` file created by the Embedding Extraction step.
```
python metrics/global_metrics.py --embs_path <embs save dir>/global_embs.npy \
--people 100 --accq 8 --score_path <global scores save dir>;
```
##### Local
For local matching we use the minutiae matcher provided by LatentAFIS. For this purpose we use the `MSU-LatentAFIS/inference.sh` file. This will convert the pickle files created by the Embedding Extraction step into template format required for local matching and then perform the matching itself.
```
bash inference.sh <embs save dir> <local scores save dir> <# subjects> <# impressions/subject> pkl;
```

##### Global + Local
Perform the global and local matching processes individually to obtain the global and local scores for all fingerprint pairs. 
**Note:** We are calculating the local scores for all pairs only for experimental purposes. Our inference algorithm would not require the use of the local scores for all pairs while calculating the Global+Local scores.
```
python metrics/norm_merge_scores.py --global_dir <global scores save dir> \
--afis_dir <local scores save dir> \
--save_dir <final scores save dir> \
--norm bisigm --ts_thresh 0.75 --fs_thresh 0.15;
```
The values for `norm, ts_thresh, fs_thresh` are set to the ones set in the paper. These can be changed according to the use case.

## Models
The models trained for the paper can be found [here](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/saraansh_tandon_research_iiit_ac_in/Eo2XSZm0gOxKhm11EH8_SygBI33Vc1jtYjlFbwUDgnNSKg?e=Sm1UK8). Place these in the `models` folder.
 -->
## Citation
