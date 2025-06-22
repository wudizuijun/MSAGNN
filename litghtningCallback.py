from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
import torch

class MyCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")
        
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the train epoch ends."""
        self._log_loss_per_epoch(trainer, pl_module, 'training')
        pl_module._epoch_rec += 1

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the val epoch ends."""
        self._log_loss_per_epoch(trainer, pl_module, 'validate')
        
    def _log_loss_per_epoch(self, trainer: Trainer, pl_module: LightningModule, phase):
        if phase == 'training':
            epoch_mean = torch.stack(pl_module.training_step_outputs).mean()
        elif phase == 'validate':
            epoch_mean = torch.stack(pl_module.val_step_outputs).mean()
        pl_module._loss[phase].append(epoch_mean.item())
        pl_module.log("{}_loss".format(phase), epoch_mean, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        pl_module.logger.experiment.add_scalars('loss', {phase: epoch_mean}, pl_module._epoch_rec)
        # free up the memory
        pl_module._free_step_outputs(phase)

# trainer = Trainer(callbacks=[MyPrintingCallback()])