config="configs.resnet101_cfbi"
datasets="davis2017"
ckpt_path="./pretrain_models/resnet101_cfbi_davis.pth"
python tools/eval_net.py --config ${config} --dataset ${datasets} --ckpt_path ${ckpt_path}