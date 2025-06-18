#!/bin/bash

method=FastAA
project=azencot-group/TSAA-r
kfold=3
ratio=1.0

for model in FEDformer Autoformer
do

for preLen in 96 192 336 720
do

# weather 
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/weather/ \
 --data_path weather.csv \
 --task_id weather \
 --model $model \
 --data custom_aa \
 --features M \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 21 \
 --dec_in 21 \
 --c_out 21 \
 --des 'Exp' \
 --itr 3\
 --method $method\
 --project $project\
 --kfold $kfold\
 --n_aug_ratio $ratio\
 --seasonality 144 

done
done


"""

for model in FEDformer Autoformer
do

# ili
for preLen in 24 36 48 60
do

python -u run.py \
 --is_training 1 \
 --root_path ./dataset/illness/ \
 --data_path national_illness.csv \
 --task_id ili \
 --model $model \
 --data custom_aa \
 --features M \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 7 \
 --dec_in 7 \
 --c_out 7 \
 --des 'Exp' \
 --itr 3 \
 --method $method\
 --project $project\
 --kfold $kfold\
 --n_aug_ratio $ratio\
 --seasonality 52

done

for preLen in 96 192 336 720
do

 

# ETTm2
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --task_id ETTm2 \
  --model $model \
  --data ETTm2_aa \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --itr 3\
  --method $method\
  --project $project\
  --kfold $kfold\
  --n_aug_ratio $ratio\
  --seasonality 96 

# traffic  
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/traffic/ \
 --data_path traffic.csv \
 --task_id traffic \
 --model $model \
 --data custom_aa \
 --features M \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 862 \
 --dec_in 862 \
 --c_out 862 \
 --des 'Exp' \
 --itr 3 \
 --method $method\
 --train_epochs 3\
 --project $project\
 --kfold $kfold\
 --n_aug_ratio $ratio\
 --seasonality 24  

done 
done
"""
"""

for model in FEDformer Autoformer
do



for preLen in 96 192 336 720
do

# weather 
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/weather/ \
 --data_path weather.csv \
 --task_id weather \
 --model $model \
 --data custom_aa \
 --features M \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 21 \
 --dec_in 21 \
 --c_out 21 \
 --des 'Exp' \
 --itr 3\
 --method $method\
 --project $project\
 --kfold $kfold\
 --n_aug_ratio $ratio\
 --seasonality 144 


done
done
"""
"""
############
### fedformer w
###########
# ili
for preLen in 24 36 48 60
do

python -u run.py \
 --is_training 1 \
 --root_path ./dataset/illness/ \
 --data_path national_illness.csv \
 --task_id ili \
 --model FEDformer \
 --data custom_aa \
 --features M \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 7 \
 --dec_in 7 \
 --c_out 7 \
 --des 'Exp' \
 --itr 3 \
 --method $method\
 --project $project\
 --kfold $kfold\
 --n_aug_ratio $ratio\
 --version Wavelets \
 --seasonality 52

done
for preLen in 96 192 336 720
do
# ETTm2
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/ETT-small/ \
 --data_path ETTm2.csv \
 --task_id ETTm2 \
 --model FEDformer \
 --data ETTm2_aa \
 --features M \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 7 \
 --dec_in 7 \
 --c_out 7 \
 --des 'Exp' \
 --d_model 512 \
 --itr 3\
 --method $method\
 --project $project\
 --kfold $kfold\
 --n_aug_ratio $ratio\
 --version Wavelets \
 --seasonality 96 

# traffic  
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/traffic/ \
 --data_path traffic.csv \
 --task_id traffic \
 --model FEDformer \
 --data custom_aa \
 --features M \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 862 \
 --dec_in 862 \
 --c_out 862 \
 --des 'Exp' \
 --itr 3 \
 --method $method\
 --project $project\
 --kfold $kfold\
 --n_aug_ratio $ratio\
 --train_epochs 3\
 --version Wavelets \
 --seasonality 24  
 
done 
"""