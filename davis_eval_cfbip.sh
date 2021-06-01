config="configs.resnet101_cfbip"
datasets="davis2017"
ckpt_path="./pretrain_models/resnet101_cfbip_davis.pth"
python tools/eval_net.py --config ${config} --dataset ${datasets} --ckpt_path ${ckpt_path} --global_atrous_rate 2