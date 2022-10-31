from pytorch_lightning import LightningModule
from lib.medzoo.Unet3Dpl import UNet3Dpl
from lib.medzoo.types_ import *
from lib.utils.general import prepare_input
from types import SimpleNamespace


tmp = SimpleNamespace()

class ThesisExperiment(LightningModule):

    def __init__(self,
                model: UNet3Dpl,
                params: dict) -> None:

                super(ThesisExperiment, self).__init__()

                # FOR MANUAL OPTIMIZATION set to False
                self.automatic_optimization = True
                
                self.model = model
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

        input_tensor, target = prepare_input(batch, self.params)


