import argparse


def get_ppo_rnd_args_train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--taskname", default="debug", type=str)
    parser.add_argument("--env_id", default="PipelineTrack-v1", type=str)
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='/home/uwsim/workspace/results/pipeline_track/checkpoint',type=str)
    parser.add_argument('--log_dir', help='the directory to save plotting data', default='/home/uwsim/workspace/results/pipeline_track/log_dir', type=str) 
    parser.add_argument('--policy_type', default="coord_cnn", type=str)
    parser.add_argument("--timesteps_per_actorbatch", default=256, type=int)
    parser.add_argument('--max_timesteps', default=1e5, type=int)
    parser.add_argument("--save_per_iter", default=100, type=int)
    parser.add_argument("--optim_batchsize", default=64, type=int)
    parser.add_argument("--optim_epochs", default=4, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--lam", default=0.95, type=float)
    parser.add_argument("--clip_param", default=0.2, type=float)
    parser.add_argument("--entcoeff", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-5, type=float)
    parser.add_argument("--optim_stepsize", default=1e-3, type=float)
    parser.add_argument("--hidden_size", default=64, type=int)
    parser.add_argument("--num_hid_layers", default=1, type=int)
    parser.add_argument("--kind", default="large", type=str)
    parser.add_argument("--convfeat", default=32, type=int)
    parser.add_argument("--rep_size", default=512, type=int)
    parser.add_argument("--proportion_of_exp_used_for_predictor_update", default=1.0, type=float)
    parser.add_argument("--enlargement", default=2, type=int)
    parser.add_argument("--int_coeff", default=1.0, type=float)
    parser.add_argument("--ext_coeff", default=10.0, type=float)

    
    return parser.parse_args()



