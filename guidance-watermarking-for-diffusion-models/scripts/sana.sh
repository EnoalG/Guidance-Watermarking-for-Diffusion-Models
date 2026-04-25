RES_DIR="/path/to/results/guidance_wm/datasets"

python -m guidance-watermarking-for-diffusion-models.guidance_benchmark --purge_previous --no-use_dataset_im_size --res_dir $RES_DIR --confs sana.yaml --wm_conf prob-guidance.yaml --detector_conf videoseal_w.yaml --transforms_conf standard_ssig.yaml --test_params_conf sana.yaml --batch_size 1 --nsamples 20

python -m guidance-watermarking-for-diffusion-models.detector_benchmark --purge_previous --no-use_dataset_im_size --res_dir $RES_DIR --confs sana.yaml --wm_conf prob-guidance.yaml --detector_conf videoseal_w.yaml --transforms_conf standard_ssig_real.yaml --batch_size 1 --nsamples 20 --opt_param 5_attacks/wm_scale_600.0
