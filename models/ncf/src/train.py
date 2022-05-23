import pandas as pd
import numpy as np
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from data import SampleGenerator
from utils import extract_users_items_predictions


gmf_config = {'alias': 'gmf_factor8_noneg-explicit',
              'num_epoch': 50,
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
              'num_users': 10000,
              'num_items': 1000,
              'latent_dim': 8,
              'num_negative': 0,
              'l2_regularization': 0, # 0.01
              'use_cuda': False,
              'device_id': 0,
              'pretrain_gmf': 'checkpoints/{}'.format('gmf_factor8_noneg-explicit_Epoch10_HR0.0962_NDCG0.0404.model'),
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

mlp_config = {'alias': 'mlp_factor8_noneg_mean_bz256_166432168_pretrain_reg_0.0000001',
              'num_epoch': 200,
              'batch_size': 256,  # 1024,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 10000,
              'num_items': 1000,
              'latent_dim': 8,
              'num_negative': 0,
              'layers': [16,64,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'use_cuda': False,
              'device_id': 0,
              'pretrain': False, # pretrain=True would use embedding weights from GMF as initialisation
            #   'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4-implict_Epoch23_HR0.1384_NDCG0.0620.model'),
              'pretrain_mlp': 'checkpoints/{}'.format('mlp_factor8_noneg_mean_bz256_166432168_pretrain_reg_0.0000001_Epoch199_HR0.0879_NDCG0.0373.model'),
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

neumf_config = {'alias': 'pretrain_neumf_factor8_noneg',
                'mlp_config': mlp_config,
                'num_epoch': 24,
                'batch_size': 256,
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users': 10000,
                'num_items': 1000,
                'latent_dim_mf': 8,
                'latent_dim_mlp': 8,
                'num_negative': 0,
                'layers': [16,64,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
                'l2_regularization': 0.01,
                'alpha': 0.5,
                'use_cuda': False,
                'device_id': 0,
                'pretrain': True,
                'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8_noneg-explicit_Epoch20_HR0.1005_NDCG0.0437.model'),
                'pretrain_mlp': 'checkpoints/{}'.format('mlp_factor8_noneg_mean_bz256_166432168_pretrain_reg_0.0000001_Epoch199_HR0.0879_NDCG0.0373.model'),
                'pretrain_neumf': 'checkpoints/{}'.format('pretrain_neumf_factor8_noneg_Epoch23_HR0.0000_NDCG0.0000.model'),
                'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
                }


if __name__ == '__main__':
    data_pd = pd.read_csv('./data/cil/data_train.csv')

    users, movies, predictions = extract_users_items_predictions(data_pd)

    cil_rating = pd.DataFrame.from_dict({'userId': users, 'itemId': movies, 'rating': predictions})

    print('Range of userId is [{}, {}]'.format(cil_rating.userId.min(), cil_rating.userId.max()))
    print('Range of itemId is [{}, {}]'.format(cil_rating.itemId.min(), cil_rating.itemId.max()))
    # DataLoader for training
    sample_generator = SampleGenerator(ratings=cil_rating)
    evaluate_data = sample_generator.evaluate_data
    # Specify the exact model
    # config = gmf_config
    # engine = GMFEngine(config)
    # config = mlp_config
    # engine = MLPEngine(config)
    config = neumf_config
    engine = NeuMFEngine(config)
    for epoch in range(config['num_epoch']):
        print('Epoch {} starts !'.format(epoch))
        print('-' * 80)
        train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
        engine.train_an_epoch(train_loader, epoch_id=epoch)
        hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
        engine.save(config['alias'], epoch, hit_ratio, ndcg)