#!/bin/bash
# Please update the path to your own environment in config.yaml and following arguments befrore running the script

cd ./scripts

exp_dir="../pretrained"

####'bedrooms'
config="../config/rearrange/diffusion_bedrooms_instancond_lat32_v_rearrange.yaml"
exp_name="bedrooms_rearrange"
weight_file=$exp_dir/$exp_name/model_17000
threed_future='/cluster/balrog/jtang/3d_front_processed/bedrooms/threed_future_model_bedroom.pkl'

python  completion_rearrange.py $config  $exp_dir/$exp_name/gen_top2down_notexture_nofloor $threed_future  --weight_file $weight_file \
    --without_screen  --n_sequences 1000 --render_top2down --no_texture --without_floor  --save_mesh --clip_denoised --retrive_objfeats --arrange_objects  --compute_intersec


####'livingrooms'
config="../config/rearrange/diffusion_livingrooms_instancond_lat32_v_rearrange.yaml"
exp_name="livingrooms_rearrange"
weight_file=$exp_dir/$exp_name/model_81000
threed_future='/cluster/balrog/jtang/3d_front_processed/livingrooms/threed_future_model_livingroom.pkl'

python  completion_rearrange.py $config  $exp_dir/$exp_name/gen_top2down_notexture_nofloor $threed_future  --weight_file $weight_file \
    --without_screen  --n_sequences 1000 --render_top2down --no_texture --without_floor  --save_mesh --clip_denoised --retrive_objfeats --arrange_objects  --compute_intersec
