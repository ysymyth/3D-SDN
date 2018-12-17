import itertools
import torch

from bulb.saver import Saver


class Net(object):
    def __init__(self,
                 model=None,
                 writer=None,
                 data_loader=None,
                 weights_only=True,
                 **kwargs):

        self.model = model
        self.writer = writer
        self.data_loader = data_loader
        self.weights_only = weights_only

        self.num_epoch = 0

        self._init(**kwargs)

    def _register_vars(self, vars_dict):
        for (var_name, var) in vars_dict.items():
            setattr(self, var_name, var)

    def _prepare(self):
        for var_name in self._var_names:
            var = getattr(self, var_name)
            var = torch.tensor(var).cuda()
            setattr(self, var_name, var)

    def _process(self):
        for var_name in itertools.chain(self._loss_names, self._metric_names):
            var = getattr(self, var_name)
            setattr(self, var_name, var.item())

    def _log(self):
        strings = [
            '[{:s}]'.format(self.name),
            'epoch: {:d}'.format(self.num_epoch + 1),
            'batch: {:d}/{:d}'.format(self.num_batch + 1, len(self.data_loader)),
            'loss: {:.4f}'.format(self.loss),
        ]

        print('\t'.join(strings))

    def _summarize(self):
        for var_name in itertools.chain(self._loss_names, self._metric_names):
            var = getattr(self, var_name)

            if self.writer is not None:
                self.writer.add_scalar(
                    '{:s}/{:s}'.format(self.name, var_name),
                    var,
                    self.num_step,
                )

    def save(self):
        assert self.name == 'train'

        if self.saver is None:
            return

        if self.weights_only:
            model = self.model.state_dict()
            optimizer = self.optimizer.state_dict()
        else:
            model = self.model
            optimizer = self.optimizer

        obj = {
            'model': model,
            'optimizer': optimizer,
        }

        self.saver.save_model(obj, num_step=self.num_step)

    def load(self, ckpt_dir):
        obj = Saver.load_model(ckpt_dir=ckpt_dir)

        if self.weights_only:
            self.model.load_state_dict(obj['model'])

            if self.name == 'train':
                self.optimizer.load_state_dict(obj['optimizer'])

            elif (self.name == 'test') or (self.name == 'online'):
                pass
        else:
            self.model = obj['model']

            if self.name == 'train':
                self.optimizer = obj['optimizer']

            elif (self.name == 'test') or (self.name == 'online'):
                pass

    def step_epoch(self, num_step=None):
        self.pre_epoch()

        for (self.num_batch, vars_dict) in enumerate(self.data_loader):
            self.num_step = num_step or self.num_epoch * len(self.data_loader) + self.num_batch

            self._var_names = list(vars_dict.keys())
            self._register_vars(vars_dict)

            self.pre_batch()

            result = self.step_batch()

            metrics_dict = {}
            if 'loss' in result:
                losses_dict = result['loss']
                if 'metrics' in result:
                    metrics_dict = result['metrics']
            else:
                losses_dict = result

            losses_dict['loss'] = torch.tensor(0.0).cuda() + sum(losses_dict.values())

            self._loss_names = list(losses_dict.keys())
            self._register_vars(losses_dict)

            self._metric_names = list(metrics_dict.keys())
            self._register_vars(metrics_dict)

            self.post_batch()

        self.post_epoch()

        self.num_epoch += 1


class TrainMixin(object):
    name = 'train'

    def _init(self,
              optimizer=None,
              lr=None,
              lr_decay_epochs=None,
              lr_decay_rate=None,
              log_steps=1,
              summarize_steps=1,
              save_steps=None,
              saver=None):

        if optimizer is not None:
            self.optimizer = optimizer
        elif lr is not None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        if lr_decay_epochs is None:
            self.scheduler = None
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_decay_epochs, gamma=lr_decay_rate)

        self.log_steps = log_steps
        self.summarize_steps = summarize_steps
        self.save_steps = save_steps
        self.saver = saver

    def _optimize(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def pre_batch(self):
        self._prepare()

    def post_batch(self):
        self._optimize()
        self._process()

        if (self.num_step % self.log_steps == 0):
            self._log()

        if (self.num_step % self.summarize_steps == 0):
            self._summarize()

        if (self.save_steps is not None) and (self.num_step % self.save_steps == 0):
            self.save()

    def pre_epoch(self):
        self.model.train()

    def post_epoch(self):
        if self.scheduler is not None:
            self.scheduler.step(epoch=self.num_epoch)

        if self.writer is not None:
            self.writer.add_scalar(
                '{:s}/lr'.format(self.name),
                self.optimizer.param_groups[0]['lr'],
                self.num_step,
            )


class TestMixin(object):
    name = 'test'

    def _init(self):
        pass

    def pre_batch(self):
        self._prepare()

    def post_batch(self):
        self._process()

        for var_name in itertools.chain(self._loss_names, self._metric_names):
            if var_name in self._loss_metrics:
                _var = self._loss_metrics[var_name]
            else:
                _var = 0

            var = getattr(self, var_name)
            self._loss_metrics[var_name] = _var + (var - _var) / (self.num_batch + 1)

    def pre_epoch(self):
        self.model.eval()
        self._loss_metrics = {}

    def post_epoch(self):
        for var_name in self._loss_metrics.keys():
            setattr(self, var_name, self._loss_metrics[var_name])

        self._log()
        self._summarize()
