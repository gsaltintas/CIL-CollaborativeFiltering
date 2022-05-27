import pandas as pd
import numpy as np
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from data import SampleGenerator
from utils import extract_users_items_predictions

NUM_USERS = 10000
NUM_ITEMS = 1000
NUM_NEGATIVE = 0 # Don't change this! It gets rid of the weird negative sampling (supposedly)

gmf_config = {'alias': 'gmf_factor8_noneg-explicit',
              'num_epoch': 100,
              'batch_size': 256,
              # 'optimizer': 'sgd',
              # 'sgd_lr': 1e-3,
              # 'sgd_momentum': 0.9,
              # 'optimizer': 'rmsprop',
              # 'rmsprop_lr': 1e-3,
              # 'rmsprop_alpha': 0.99,
              # 'rmsprop_momentum': 0,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': NUM_USERS,
              'num_items': NUM_ITEMS,
              'latent_dim': 8, # also called "num_factors" in the original authors' implementation
              'num_negative': NUM_NEGATIVE,
              'l2_regularization': 0, # 0.01
              'use_cuda': False,
              'device_id': 0,
              'use_checkpoint': False, # Resume training from some checkpoint, e.g. if doing pretraining
              'checkpoint_loc': 'checkpoints/{}'.format('gmf_factor8_noneg-explicit_Epoch0_HR0.0721_NDCG0.0307.model'),
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

mlp_config = {'alias': 'mlp_factor8_noneg_mean_bz256_166432168_pretrain_reg_0.0000001',
              'num_epoch': 100,
              'batch_size': 256,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': NUM_USERS,
              'num_items': NUM_ITEMS,
              'latent_dim': 32,
              'num_negative': NUM_NEGATIVE,
              'layers': [16,64,32,16,8], #  The 0-th layer (embedding layer) is created when MLP object is initialised, so these are the other layers
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'use_cuda': False,
              'device_id': 0,
              'pretrain': False, # pretrain=True uses embedding weights from GMF as initialisation
              'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4-implict_Epoch23_HR0.1384_NDCG0.0620.model'),
              'use_checkpoint': False, # Resume training from some checkpoint, e.g. if doing pretraining
              'checkpoint_loc': 'checkpoints/{}'.format('mlp_factor8_noneg_mean_bz256_166432168_pretrain_reg_0.0000001_Epoch199_HR0.0879_NDCG0.0373.model'),
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

neumf_config = {'alias': 'pretrain_neumf_factor8_noneg',
                'mlp_config': mlp_config,
                'gmf_config': gmf_config,
                'num_epoch': 100,
                'batch_size': 256,
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users': NUM_USERS,
                'num_items': NUM_ITEMS,
                'latent_dim_mf': gmf_config['latent_dim'],
                'latent_dim_mlp': mlp_config['latent_dim'],
                'num_negative': NUM_NEGATIVE,
                'layers': mlp_config['layers'], # Change only if you don't want to use MLP with the NeuMF layer. Otherwise they need to match up
                'l2_regularization': 0.01,
                'alpha': 0.5,
                'use_cuda': False,
                'device_id': 0,
                'pretrain': True, # True if want to use saved GMF + MLP models
                'pretrain_mf': 'checkpoints/{}'.format('gmf.model'),
                'pretrain_mlp': 'checkpoints/{}'.format('mlp.model'),
                'use_checkpoint': False,
                'checkpoint_loc': 'checkpoints/{}'.format('pretrain_neumf_factor8_noneg_Epoch23_HR0.0000_NDCG0.0000.model'),
                'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
                }


if __name__ == '__main__':
    data_pd = pd.read_csv('./data/cil/data_train.csv')

    users, movies, predictions = extract_users_items_predictions(data_pd)

    cil_rating = pd.DataFrame.from_dict({'userId': users, 'itemId': movies, 'rating': predictions})

    # DataLoader for training
    sample_generator = SampleGenerator(ratings=cil_rating)
    evaluate_data = sample_generator.evaluate_data
    
    # Specify the exact model from {'GMF', 'MLP', 'NEUMF'}
    TRAIN_MODEL = 'GMF'

    if TRAIN_MODEL == 'GMF':
      config = gmf_config
      engine = GMFEngine(config)

    if TRAIN_MODEL == 'MLP':
      config = mlp_config
      engine = MLPEngine(config)

    if TRAIN_MODEL == 'NEUMF':
      # Generally you need a GMF and an MLP model trained & saved before doing NeuMF, if
      # going by the paper's final architecture. So make sure those are done!
      config = neumf_config
      engine = NeuMFEngine(config)

    for epoch in range(config['num_epoch']):
        print('Epoch {} starts !'.format(epoch))
        print('-' * 80)
        train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
        engine.train_an_epoch(train_loader, epoch_id=epoch)
        hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
        engine.save(config['alias'], epoch, hit_ratio, ndcg)