from typing import Any, Dict, List, Optional, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.sampler import FederatedSampler
# from models import CNN, MLP
import fsvae_models.fsvae as fsvae
from fsvae_models.snn_layers import LIFSpike
from utils import arg_parser, average_weights
from datasets import load_dataset_snn
import os
import global_v as glv
from network_parser import parse
from utils import aboutCudaDevices
from utils import AverageMeter
from utils import CountMulAddSNN
import torchvision

def add_hook(net):
    count_mul_add = CountMulAddSNN()
    hook_handles = []
    for m in net.modules():
        if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Linear) or isinstance(m,
                                                                                          torch.nn.ConvTranspose3d) or isinstance(
            m, LIFSpike):
            handle = m.register_forward_hook(count_mul_add)
            hook_handles.append(handle)
    return count_mul_add, hook_handles

class FedAvg:
    """Implementation of FedAvg
    http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf
    """

    def __init__(self, args: Dict[str, Any]):
        self.args = args
        self.device = torch.device(
            f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
        )

        self.train_loader, self.test_loader = self._get_data(
            root=self.args.data_root,
            n_clients=self.args.n_clients,
            n_shards=self.args.n_shards,
            non_iid=self.args.non_iid,
        )

        # if self.args.model_name == "mlp":
        #     self.root_model = MLP(input_size=784, hidden_size=128, n_classes=10).to(
        #         self.device
        #     )
        #     self.target_acc = 0.97
        # elif self.args.model_name == "cnn":
        #     self.root_model = CNN(n_channels=1, n_classes=10).to(self.device)
        #     self.target_acc = 0.99
        # else:
        #     raise ValueError(f"Invalid model name, {self.args.model_name}")
        self.root_model = fsvae.FSVAELarge()

        self.reached_target_at = None  # type: int

    def _get_data(
        self, root: str, n_clients: int, n_shards: int, non_iid: int
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Args:
            root (str): path to the dataset.
            n_clients (int): number of clients.
            n_shards (int): number of shards.
            non_iid (int): 0: IID, 1: Non-IID

        Returns:
            Tuple[DataLoader, DataLoader]: train_loader, test_loader
        """
        # train_set = MNISTDataset(root=root, train=True)
        # test_set = MNISTDataset(root=root, train=False)

        # train_loader = DataLoader(train_set, batch_size=128, sampler=sampler)
        # test_loader = DataLoader(test_set, batch_size=128)
        data_path = os.path.expanduser("./data")
        train_loader, test_loader = load_dataset_snn.load_mvtec(data_path, non_iid=non_iid, n_clients=n_clients, n_shards=n_shards)

        return train_loader, test_loader

    def _train_client(
        self, root_model: nn.Module, train_loader: DataLoader, client_idx: int
    ) -> Tuple[nn.Module, float]:
        """Train a client model.

        Args:
            root_model (nn.Module): server model.
            train_loader (DataLoader): client data loader.
            client_idx (int): client index.

        Returns:
            Tuple[nn.Module, float]: client model, average client loss.
        """
        n_steps = glv.network_config['n_steps']
        max_epoch = glv.network_config['epochs']

        loss_meter = AverageMeter()
        recons_meter = AverageMeter()
        dist_meter = AverageMeter()

        mean_q_z = 0
        mean_p_z = 0
        mean_sampled_z = 0
        network = copy.deepcopy(root_model)
        network.train()
        network = network.to(self.device)
        # optimizer = torch.optim.SGD(
        #     model.parameters(), lr=self.args.lr, momentum=self.args.momentum
        # )
        opti = torch.optim.AdamW(network.parameters(),
                                      lr=glv.network_config['lr'],
                                      betas=(0.9, 0.999),
                                      weight_decay=0.001)

        for epoch in range(self.args.n_client_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                real_img = data.to(self.device)
                opti.zero_grad()
                spike_input = real_img.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps)  # (N,C,H,W,T)
                x_recon, q_z, p_z, sampled_z = network(spike_input,
                                                       scheduled=network_config['scheduled'])  # sampled_z(B,C,1,1,T)

                if network_config['loss_func'] == 'mmd':
                    losses = network.loss_function_mmd(real_img, x_recon, q_z, p_z)
                elif network_config['loss_func'] == 'kld':
                    losses = network.loss_function_kld(real_img, x_recon, q_z, p_z)
                else:
                    raise ValueError('unrecognized loss function')

                losses['loss'].backward()

                opti.step()

                loss_meter.update(losses['loss'].detach().cpu().item())
                recons_meter.update(losses['Reconstruction_Loss'].detach().cpu().item())
                dist_meter.update(losses['Distance_Loss'].detach().cpu().item())

                mean_q_z = (q_z.mean(0).detach().cpu() + batch_idx * mean_q_z) / (batch_idx + 1)  # (C,k,T)
                mean_p_z = (p_z.mean(0).detach().cpu() + batch_idx * mean_p_z) / (batch_idx + 1)  # (C,k,T)
                mean_sampled_z = (sampled_z.mean(0).detach().cpu() + batch_idx * mean_sampled_z) / (
                            batch_idx + 1)  # (C,T)

                print(
                    f'Train[[{batch_idx}/{len(train_loader)}] Loss: {loss_meter.avg}, RECONS: {recons_meter.avg}, DISTANCE: {dist_meter.avg}')

                # logits = model(data)
                # loss = F.nll_loss(logits, target)
                # loss.backward()
                # optimizer.step()

                epoch_loss += losses['Reconstruction_Loss'].detach().cpu().item()
                # epoch_correct += (logits.argmax(dim=1) == target).sum().item()
                # epoch_samples += data.size(0)

            # Calculate average accuracy and loss
            epoch_loss /= (batch_idx+1)
            # epoch_acc = epoch_correct / epoch_samples

            print(
                f"Client #{client_idx} | Epoch: {epoch}/{self.args.n_client_epochs} | Loss: {epoch_loss}"
            )

        return network, epoch_loss #/ self.args.n_client_epochs

    def train(self) -> None:
        """Train a server model."""
        train_losses = []

        for epoch in range(self.args.n_epochs):
            clients_models = []
            clients_losses = []

            # Randomly select clients
            m = max(int(self.args.frac * self.args.n_clients), 1)
            idx_clients = np.random.choice(range(self.args.n_clients), m, replace=False)

            # Train clients
            self.root_model.train()

            for client_idx in idx_clients:
                # Set client in the sampler
                self.train_loader.sampler.set_client(client_idx)

                # Train client
                client_model, client_loss = self._train_client(
                    root_model=self.root_model,
                    train_loader=self.train_loader,
                    client_idx=client_idx,
                )
                clients_models.append(client_model.state_dict())
                clients_losses.append(client_loss)

            # Update server model based on clients models
            updated_weights = average_weights(clients_models)
            self.root_model.load_state_dict(updated_weights)

            # Update average loss of this round
            avg_loss = sum(clients_losses) / len(clients_losses)
            train_losses.append(avg_loss)

            if (epoch + 1) % self.args.log_every == 0:
                # Test server model
                total_loss = self.test(epoch)
                avg_train_loss = sum(train_losses) / len(train_losses)

                # Log results
                logs = {
                    "train/loss": avg_train_loss,
                    "test/loss": total_loss,
                    # "test/acc": total_acc,
                    "round": epoch,
                }
                # if total_acc >= self.target_acc and self.reached_target_at is None:
                #     self.reached_target_at = epoch
                #     logs["reached_target_at"] = self.reached_target_at
                #     print(
                #         f"\n -----> Target accuracy {self.target_acc} reached at round {epoch}! <----- \n"
                #     )

                # self.logger.log(logs)

                # Print results to CLI
                print(f"\n\nResults after {epoch + 1} rounds of training:")
                print(f"---> Avg Training Loss: {avg_train_loss}")
                # print(
                #     f"---> Avg Test Loss: {total_loss} | Avg Test Accuracy: {total_acc}\n"
                # )

                # Early stopping
                if self.args.early_stopping and self.reached_target_at is not None:
                    print(f"\nEarly stopping at round #{epoch}...")
                    break
                torch.save(self.root_model.state_dict(), f'checkpoint/{args.name}/checkpoint.pth')

    def test(self, epoch) -> Tuple[float, float]:
        """Test the server model.

        Returns:
            Tuple[float, float]: average loss, average accuracy.
        """
        n_steps = glv.network_config['n_steps']
        max_epoch = glv.network_config['epochs']

        loss_meter = AverageMeter()
        recons_meter = AverageMeter()
        dist_meter = AverageMeter()

        mean_q_z = 0
        mean_p_z = 0
        mean_sampled_z = 0

        count_mul_add, hook_handles = add_hook(self.root_model)

        self.root_model.eval()
        self.root_model = self.root_model.to(self.device)
        with torch.no_grad():
            for batch_idx, (real_img, labels) in enumerate(self.test_loader):
                real_img = real_img.to(self.device, non_blocking=True)
                # labels = labels.to(init_device, non_blocking=True)
                # direct spike input
                spike_input = real_img.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps)  # (N,C,H,W,T)

                x_recon, q_z, p_z, sampled_z = self.root_model(spike_input, scheduled=network_config['scheduled'])

                if network_config['loss_func'] == 'mmd':
                    losses = self.root_model.loss_function_mmd(real_img, x_recon, q_z, p_z)
                elif network_config['loss_func'] == 'kld':
                    losses = self.root_model.loss_function_kld(real_img, x_recon, q_z, p_z)
                else:
                    raise ValueError('unrecognized loss function')

                mean_q_z = (q_z.mean(0).detach().cpu() + batch_idx * mean_q_z) / (batch_idx + 1)  # (C,k,T)
                mean_p_z = (p_z.mean(0).detach().cpu() + batch_idx * mean_p_z) / (batch_idx + 1)  # (C,k,T)
                mean_sampled_z = (sampled_z.mean(0).detach().cpu() + batch_idx * mean_sampled_z) / (
                            batch_idx + 1)  # (C,T)

                loss_meter.update(losses['loss'].detach().cpu().item())
                recons_meter.update(losses['Reconstruction_Loss'].detach().cpu().item())
                dist_meter.update(losses['Distance_Loss'].detach().cpu().item())

                print(
                    f'Test[{epoch}/{max_epoch}] [{batch_idx}/{len(self.test_loader)}] Loss: {loss_meter.avg}, RECONS: {recons_meter.avg}, DISTANCE: {dist_meter.avg}')

                if batch_idx == len(self.test_loader) - 1:
                    os.makedirs(f'checkpoint/{args.name}/imgs/test/', exist_ok=True)
                    torchvision.utils.save_image((real_img + 1) / 2,
                                                 f'checkpoint/{args.name}/imgs/test/epoch{epoch}_input.png')
                    torchvision.utils.save_image((x_recon + 1) / 2,
                                                 f'checkpoint/{args.name}/imgs/test/epoch{epoch}_recons.png')

        for handle in hook_handles:
            handle.remove()

        # writer.add_image('Test/mean_sampled_z', mean_sampled_z.unsqueeze(0), epoch)
        mean_q_z = mean_q_z.permute(1, 0, 2)  # # (k,C,T)
        mean_p_z = mean_p_z.permute(1, 0, 2)  # # (k,C,T)
        # writer.add_image(f'Test/mean_q_z', mean_q_z.mean(0).unsqueeze(0))
        # writer.add_image(f'Test/mean_p_z', mean_p_z.mean(0).unsqueeze(0))

        return loss_meter.avg


if __name__ == "__main__":
    args = arg_parser()
    os.makedirs(f'checkpoint/{args.name}', exist_ok=True)
    params = parse(args.config)
    network_config = params['Network']
    glv.init(network_config, [args.device])
    fed_avg = FedAvg(args)
    fed_avg.train()
