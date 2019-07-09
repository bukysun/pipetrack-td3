# dense observation with 2-element extracted features [cos(theta), d] 
# 2019/04/22
nohup python train-ppo-tf-cnn.py --seed 123 --taskname dense_obs > nohup.ppo.dense_obs.out &

# 2019/04/22 10:00
# dense observation with 3-element extracted features [cos(theta), sin(theta), d] 
nohup python train-ppo-tf-cnn.py --seed 123 --taskname dense_obs_3 > nohup.ppo.dense_obs_3.out &

# 2019/04/22 14:00
# dense observation with 3-element extracted features [cos(theta), sin(theta), d], add process for None reward and feat. 
nohup python train-ppo-tf-cnn.py --seed 123 --taskname dense_obs_3.none_process > nohup.ppo.dense_obs_3.none_process.out &

# 2019/04/22 20:00
# add initial angle encoding into the state
nohup python train-ppo-tf-cnn.py --seed 123 --taskname init_angle_enc --policy_type coord_cnn > nohup.ppo.init_angle_enc.out &

# 2019/04/24 15:18
# add initial angle encoding into the state
nohup python train-ppo-tf-cnn.py --seed 123 --taskname real_rew --policy_type coord_cnn > nohup.ppo.real_rew.out &
