# Synthetic Data Experiments

### Datasets
Download the PASCAL VOC dataset and SUN 2012 dataset. We use objects from PASCAL VOC as foreground and images from SUN 2012 for background textures.

### Preprocessing
```
# Download PASCAL VOC annotations
mkdir cachedir; cd cachedir;
wget https://people.eecs.berkeley.edu/~shubhtuls/cachedir/vpsKps/segkps.zip
unzip segkps.zip
cd ..

# Extract images for foreground objects
cd lsi/data/syntheticPlanes
# Modify the paths in 'preprocess.m'
# run from matlab:
>> preprocess
```

### Training
We provide below sample scripts to train the 2 layer prediction model and the 1 layer baseline. You might need to change the paths to datasets, desired snapshot directory, and precomputed PASCAL VOC foreground objects in the flags.
```
# 2 layer experiment
python ldi_enc_dec.py --dataset=synthetic --pascal_objects_dir=/code/lsi/cachedir/sbd/objects --sun_imgs_dir=/datasets/SUN2012pascalformat/JPEGImages --batch_size=4 --n_layers=2 --use_unet=true --num_iter=500000 --disp_smoothness_wt=0.1 --exp_name=synth_ldi_nl2 --n_layerwise_steps=3 --trg_splat_downsampling=0.5 --compose_splat_wt=1.0 --indep_splat_wt=1.0 --self_cons_wt=10 --splat_bdry_ignore=0.05 --zbuf_scale=50 --log_freq=500 --checkpoint_dir=/code/lsi/cachedir/snapshots/

# 1 layer experiment
python ldi_enc_dec.py --dataset=synthetic --pascal_objects_dir=/code/lsi/cachedir/sbd/objects --sun_imgs_dir=/datasets/SUN2012pascalformat/JPEGImages --batch_size=4 --n_layers=1 --use_unet=true --num_iter=500000 --disp_smoothness_wt=0.1 --exp_name=synth_ldi_nl1 --n_layerwise_steps=3 --trg_splat_downsampling=0.5 --compose_splat_wt=1.0 --indep_splat_wt=1.0 --self_cons_wt=10 --splat_bdry_ignore=0.05 --zbuf_scale=50 --log_freq=500 --checkpoint_dir=/code/lsi/cachedir/snapshots/
```

### Evaluation
To evaluate the trained models, run:
```
# 2 layer experiment
python ldi_pred_eval.py  --exp_name=synth_ldi_nl2 --train_iter=500000  --dataset=synthetic --pascal_objects_dir=/code/lsi/cachedir/sbd/objects --sun_imgs_dir=/datasets/SUN2012pascalformat/JPEGImages --batch_size=4 --n_layers=2 --n_layerwise_steps=3 --use_unet --synth_ds_factor=2  --checkpoint_dir=/code/lsi/cachedir/snapshots --results_vis_dir=/code/lsi/cachedir/visualization/ --results_eval_dir=/code/lsi/cachedir/evaluation/  --trg_splat_downsampling=0.5 --data_split=test --num_eval_iter=250 --zbuf_scale=50 --visuals_freq=5

# 1 layer experiment
python ldi_pred_eval.py  --exp_name=synth_ldi_nl1 --train_iter=500000  --dataset=synthetic --pascal_objects_dir=/code/lsi/cachedir/sbd/objects --sun_imgs_dir=/datasets/SUN2012pascalformat/JPEGImages --batch_size=4 --n_layers=1 --n_layerwise_steps=3 --use_unet --synth_ds_factor=2  --checkpoint_dir=/code/lsi/cachedir/snapshots --results_vis_dir=/code/lsi/cachedir/visualization/ --results_eval_dir=/code/lsi/cachedir/evaluation/  --trg_splat_downsampling=0.5 --data_split=test --num_eval_iter=250 --zbuf_scale=50 --visuals_freq=5
```

### Ablations
We report some architecture ablations in the paper. To train and evaluate these noetworks, please change (one at a time) the flags 'n_layerwise_steps', 'disp_smoothness_wt', and 'indep_splat_wt' to 0 in the training and evaluation commands for the 2 layer experiment.
