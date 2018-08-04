# Kitti Data Experiments

### Datasets
Download the KITTI raw dataset. We will use the sequences from city environments for training and evaluation.

### Preprocessing
To run the evaluations, we need to compute 'dis-occluded' pixels to analyze view synthesis error over. We do so by using a (slightly modified) off the shelf stereo matching algorithm. The instructions for this preproessing are below. Note that this step is only required for evaluation.
```
# Download stereo matching toolbox
cd external;
wget http://ttic.uchicago.edu/~dmcallester/SPS/spsstereo.zip
unzip spsstereo.zip

# Use our patch to modify and compile
cd spsstereo;
git init; git add ./*; git commit -m “init”;
git apply ../stereo.patch
cmake .
make

# Run stereo matching
cd CODE_ROOT/lsi/data/kitti
python preprocess.py --kitti_data_root=/data0/shubhtuls/datasets/kitti --spss_exec=/data0/shubhtuls/code/lsi/external/spsstereo_git_patch/spsstereo
```

### Training
We provide below sample scripts to train the 2 layer prediction model and the 1 layer baseline. You might need to change the paths to datasets and desired snapshot directory in the flags.
```
# 2 layer experiment
python ldi_enc_dec.py --dataset=kitti --kitti_data_root=/data0/shubhtuls/datasets/kitti --kitti_dataset_variant=raw_city --batch_size=4 --n_layers=2 --use_unet=true --num_iter=500000 --disp_smoothness_wt=0.1 --exp_name=kitti_rcity_ldi_nl2 --n_layerwise_steps=3 --trg_splat_downsampling=0.5 --compose_splat_wt=1.0 --indep_splat_wt=1.0 --self_cons_wt=10 --splat_bdry_ignore=0.05 --img_width=768 --zbuf_scale=50 --log_freq=500 --checkpoint_dir=/data0/shubhtuls/code/lsi/cachedir/snapshots/

# 1 layer experiment
python ldi_enc_dec.py --dataset=kitti --kitti_data_root=/data0/shubhtuls/datasets/kitti --kitti_dataset_variant=raw_city --batch_size=4 --n_layers=1 --use_unet=true --num_iter=500000 --disp_smoothness_wt=0.1 --exp_name=kitti_rcity_ldi_nl1 --n_layerwise_steps=3 --trg_splat_downsampling=0.5 --compose_splat_wt=1.0 --indep_splat_wt=1.0 --self_cons_wt=10 --splat_bdry_ignore=0.05 --img_width=768 --zbuf_scale=50 --log_freq=500 --checkpoint_dir=/data0/shubhtuls/code/lsi/cachedir/snapshots/
```

### Evaluation
To evaluate the trained models, run:
```
# 2 layer experiment
python ldi_pred_eval.py  --exp_name=kitti_rcity_ldi_nl2 --train_iter=400000 --dataset=kitti --kitti_data_root=/data0/shubhtuls/datasets/kitti --kitti_dataset_variant=raw_city --batch_size=4 --n_layers=2 --img_width=768 --n_layerwise_steps=3 --use_unet --synth_ds_factor=2  --checkpoint_dir=/data0/shubhtuls/code/lsi/cachedir/snapshots  --trg_splat_downsampling=0.5 --data_split=val --num_eval_iter=250 --zbuf_scale=50 --splat_bdry_ignore=0.05

# 1 layer experiment
python ldi_pred_eval.py  --exp_name=kitti_rcity_ldi_nl1 --train_iter=400000 --dataset=kitti --kitti_data_root=/data0/shubhtuls/datasets/kitti --kitti_dataset_variant=raw_city --batch_size=4 --n_layers=1 --img_width=768 --n_layerwise_steps=3 --use_unet --synth_ds_factor=2  --checkpoint_dir=/data0/shubhtuls/code/lsi/cachedir/snapshots  --trg_splat_downsampling=0.5 --data_split=val --num_eval_iter=250 --zbuf_scale=50 --splat_bdry_ignore=0.05
```