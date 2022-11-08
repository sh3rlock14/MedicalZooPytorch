from pytorch_lightning import LightningModule
from lib.medzoo.Unet3Dpl import UNet3Dpl
from lib.medzoo.types_ import *
from lib.utils.general import prepare_input
from types import SimpleNamespace

from torch.optim import SGD, Adam, RMSprop
from torch import cat

tmp = SimpleNamespace()

class ThesisExperiment(LightningModule):

    def __init__(self,
                model: UNet3Dpl,
                criterion,
                optimizer: str,
                params: dict) -> None:

                super(ThesisExperiment, self).__init__()

                # FOR MANUAL OPTIMIZATION set to False
                self.automatic_optimization = True
                
                self.model = model
                self.criterion = criterion
                self.optimizer = optimizer.lower()
                self.params = SimpleNamespace(**params) #params
                self.curr_device = None
                self.hold_graph = False

                try:
                    self.hold_graph = self.params['retain_first_backpass']
                except:
                    pass
    
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)
    
    def training_step(self, batch, batch_idx):

        input_tensor, target = self.prepare_data_modalities(batch)

        # Check if `input_tensor` has `requires_grad` = True

        output = self.forward(input_tensor)

        loss_dice, per_ch_score = self.criterion(output, target)


        # Log metrics
        # when logging, check if Trainer params are accessible
    

    def validation_step(self, batch, batch_idx):

        input_tensor, target = self.prepare_data_modalities(batch)

        output = self.forward(input_tensor)

        loss, per_ch_score = self.criterion(output, target)

        # Log metrics
        # when logging, check if Trainer params are accessible:
        # set `onStep` = True to use Trainer's `log_every_n_steps` param


    def prepare_data_modalities(self, input_tuple, inModalities=-1, inChannels=-1) -> None:
        if self.params is not None:
            modalities = self.params.inModalities
            channels = self.params.inChannels
            #in_cuda = self.params.cuda
        else:
            modalities = inModalities
            channels = inChannels
            #in_cuda = cuda
    
        if modalities == 4:
            if channels == 4:
                img_1, img_2, img_3, img_4, target = input_tuple
                input_tensor = cat((img_1, img_2, img_3, img_4), dim=1)
            elif channels == 3:
                # t1 post constast is ommited
                img_1, _, img_3, img_4, target = input_tuple
                input_tensor = cat((img_1, img_3, img_4), dim=1)
            elif channels == 2:
                # t1 and t2 only
                img_1, _, img_3, _, target = input_tuple
                input_tensor = cat((img_1, img_3), dim=1)
            elif channels == 1:
                # t1 only
                input_tensor, _, _, target = input_tuple
        elif modalities == 3:
            if channels == 3:
                img_1, img_2, img_3, target = input_tuple
                input_tensor = cat((img_1, img_2, img_3), dim=1)
            elif channels == 2:
                img_1, img_2, _, target = input_tuple
                input_tensor = cat((img_1, img_2), dim=1)
            elif channels == 1:
                input_tensor, _, _, target = input_tuple
        elif modalities == 2:
            if channels == 2:
                img_t1, img_t2, target = input_tuple

                input_tensor = cat((img_t1, img_t2), dim=1)

            elif channels == 1:
                input_tensor, _, target = input_tuple
        elif modalities == 1:
            input_tensor, target = input_tuple

        #if in_cuda:
        #    input_tensor, target = input_tensor.cuda(), target.cuda()

        return input_tensor, target


    def configure_optimizers(self):

        if self.optimizer == 'sgd':
            optimizer = SGD(self.model.parameters(),
                            lr=self.params.lr,
                            momentum=self.params.momentum,
                            weight_decay=self.params.weight_decay)
        elif self.optimizer == 'adam':
            optimizer = Adam(self.model.parameters(),
                            lr=self.params.lr,
                            weight_decay=self.params.weight_decay)
        elif self.optimizer == 'rmsprop':
            optimizer = RMSprop(self.model.parameters(),
                            lr=self.params.lr,
                            momentum=self.params.momentum,
                            alpha=self.params.alpha,
                            weight_decay=self.params.weight_decay
                            )
        
        return optimizer
            







