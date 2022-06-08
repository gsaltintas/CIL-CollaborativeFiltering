import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from utils import save_checkpoint, use_optimizer


class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
        self._writer.add_text('config', str(config), 0)
        self.opt = use_optimizer(self.model, config)
        self.crit = torch.nn.MSELoss()

    def train_single_batch(self, users, items, ratings):
        assert hasattr(self, 'model'), 'Please specify the exact model!'
        if self.config['use_cuda'] is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
        self.opt.zero_grad()
        ratings_pred = self.model(users, items)
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss.backward()
        self.opt.step()
        loss = (torch.sqrt(loss)).item() # Train with MSE but report RMSE
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model!'
        self.model.train()
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
            loss = self.train_single_batch(user, item, rating)
            if batch_id % int(len(train_loader) * 0.2) == 0: print('[Training Epoch {}] Batch {}, RMSE loss {}'.format(epoch_id, batch_id, loss))
            total_loss += loss
        self._writer.add_scalar('model/loss', total_loss, epoch_id)

    def evaluate(self, evaluate_data, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model!'
        self.model.eval()
        with torch.no_grad():
            test_users, test_items, test_ratings = evaluate_data[0], evaluate_data[1], evaluate_data[2]
            ratings_pred = self.model(test_users, test_items)
            loss = torch.sqrt(self.crit(ratings_pred.view(-1), test_ratings))

        self._writer.add_scalar('performance/val_loss', loss, epoch_id)
        print('[Evaluating epoch {}] RMSE loss = {:.4f}'.format(epoch_id, loss))

        return loss.item()

    def save(self, alias, epoch_id, validation_loss):
        assert hasattr(self, 'model'), 'Please specify the exact model!'
        model_dir = self.config['model_dir'].format(alias, epoch_id, validation_loss)
        save_checkpoint(self.model, model_dir)