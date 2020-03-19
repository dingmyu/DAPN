now=`date +%Y-%m-%d,%H:%m:%s`
srun -p Test  --gres=gpu:8 -n1 --job-name=cdan python -u eval.py --net ResNet50 --dset imagenet --t_dset_path list/test_transfer_list.txt --output_dir test  2>&1|tee logs/eval-${now}.log
