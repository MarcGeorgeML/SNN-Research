import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

import mlflow
import datetime
import random
import numpy as np

from spikingjelly.activation_based import functional

from dataset.build_dataloader import build_dataloaders
from Model.SpikEmo_Model import SpikEmo
from Model.spikformer import Spikformer

from Loss.MultiDSCLoss import MultiDSCLoss
from Loss.SoftHGRLoss import SoftHGRLoss


# =========================
# SEED
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# CONFIG
# =========================
class Config:

    def __init__(self):
        self.seed = 42

        self.batch_size = 32
        self.epochs = 50
        self.lr = 1e-4
        self.weight_decay = 1e-5

        self.model_dim = 256
        self.num_heads = 4
        self.num_layers = 6
        self.hidden_dim = 1024

        self.T = 8
        self.spike_tau = 10.0
        self.spike_thr = 1.0

        self.loss_HGR = 0.3
        self.loss_DSC = 0.4
        self.loss_CE = 0.3

        self.num_classes = 6
        self.device = "cuda"

        self.grad_clip = 1.0         
        self.early_stop_patience = 10

    def to_dict(self):
        return self.__dict__


# =========================
# RUN NAME
# =========================
def generate_run_name(config):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"SpikEmo"
        f"_T{config.T}"
        f"_dim{config.model_dim}"
        f"_h{config.num_heads}"
        f"_lr{config.lr}"
        f"_bs{config.batch_size}"
        f"_{timestamp}"
    )


# =========================
# TRAINER
# =========================
class Trainer:

    def __init__(self, config):

        self.config = config
        self.device = torch.device(config.device)

        self.best_f1 = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0

        self.train_loader, self.val_loader = build_dataloaders(
            "features",
            batch_size=config.batch_size
        )

        self.model = self.build_model()

        self.MultiDSC_loss = MultiDSCLoss()
        self.HGR_loss = SoftHGRLoss()
        self.CE_loss = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.95,
            patience=5
        )

    def build_model(self):

        spikformer_model = Spikformer(
            depths=2,
            T=self.config.T,
            tau=self.config.spike_tau,
            common_thr=self.config.spike_thr,
            dim=self.config.model_dim,
            heads=8
        )

        model = SpikEmo(
            dataset="custom",
            multi_attn_flag=True,
            roberta_dim=768,
            hidden_dim=self.config.hidden_dim,
            dropout=0,
            num_layers=self.config.num_layers,
            model_dim=self.config.model_dim,
            num_heads=self.config.num_heads,
            D_m_audio=512,
            D_m_visual=1000,
            n_classes=self.config.num_classes,
            spikformer_model=spikformer_model
        )

        return model.to(self.device)

    def setup_mlflow(self):
        mlflow.set_tracking_uri("sqlite:///snn.db")
        mlflow.set_experiment("SpikEmo_SNN")

    def start_run(self):
        self.run_name = generate_run_name(self.config)
        mlflow.start_run(run_name=self.run_name)
        mlflow.log_params(self.config.to_dict())

    def log_metrics(self, epoch, train_metrics, val_metrics):

        t_loss, t_HGR, t_DSC, t_CE, t_f1, t_acc = train_metrics
        v_loss, v_HGR, v_DSC, v_CE, v_f1, v_acc = val_metrics

        mlflow.log_metrics({
            "train_loss": float(t_loss),
            "train_f1": float(t_f1),
            "train_acc": float(t_acc),
            "train_HGR": float(t_HGR),
            "train_DSC": float(t_DSC),
            "train_CE": float(t_CE),
            "val_loss": float(v_loss),
            "val_f1": float(v_f1),
            "val_acc": float(v_acc),
            "val_HGR": float(v_HGR),
            "val_DSC": float(v_DSC),
            "val_CE": float(v_CE),
        }, step=epoch)

    def log_best(self):
        mlflow.log_metrics({
            "best_val_f1": float(self.best_f1),
            "best_epoch": self.best_epoch
        })

    def end_run(self):
        mlflow.end_run()

    def save_checkpoint(self, epoch, val_f1):                             
        filename = f"checkpoints/checkpoint_epoch{epoch}_f1{val_f1:.4f}.pt"          
        torch.save(self.model.state_dict(), filename)                     
        print(f"  Checkpoint saved → {filename}")                         

    def run_epoch(self, loader, train=True):

        self.model.train() if train else self.model.eval()

        total_loss = 0
        total_HGR, total_DSC, total_CE = 0, 0, 0

        preds, labels_all = [], []

        context = torch.no_grad() if not train else torch.enable_grad()
        with context:
            for batch in tqdm(loader):

                texts, audios, visuals, _, _, labels = batch
                texts = texts.to(self.device, non_blocking=True)
                audios = audios.to(self.device, non_blocking=True)
                visuals = visuals.to(self.device, non_blocking=True)
                labels = labels.squeeze(1).to(self.device, non_blocking=True)

                if train:
                    self.optimizer.zero_grad()

                f_t, f_a, f_v, _, logits = self.model(texts, audios, visuals)

                loss_HGR = self.HGR_loss(f_t, f_a, f_v)
                loss_DSC = self.MultiDSC_loss(logits, labels)
                loss_CE = self.CE_loss(logits, labels)

                loss = (
                    self.config.loss_HGR * loss_HGR +
                    self.config.loss_DSC * loss_DSC +
                    self.config.loss_CE * loss_CE
                )

                if torch.isnan(loss):
                    print("NaN detected — stopping")
                    return None

                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(                
                        self.model.parameters(),                   
                        self.config.grad_clip                      
                    )                                              
                    self.optimizer.step()

                functional.reset_net(self.model)

                total_loss += loss.item()
                total_HGR += loss_HGR.item()
                total_DSC += loss_DSC.item()
                total_CE += loss_CE.item()

                preds.append(torch.argmax(logits, dim=1).cpu())
                labels_all.append(labels.cpu())

        preds = torch.cat(preds)
        labels_all = torch.cat(labels_all)

        f1 = f1_score(labels_all, preds, average="weighted")
        acc = accuracy_score(labels_all, preds)

        return (
            total_loss / len(loader),
            total_HGR / len(loader),
            total_DSC / len(loader),
            total_CE / len(loader),
            f1,
            acc
        )

    def train(self):

        self.setup_mlflow()
        self.start_run()

        try:
            for epoch in range(self.config.epochs):

                train_metrics = self.run_epoch(self.train_loader, True)
                val_metrics = self.run_epoch(self.val_loader, False)

                if train_metrics is None:
                    break

                self.log_metrics(epoch, train_metrics, val_metrics)

                if train_metrics is None or val_metrics is None:
                    print("NaN detected in metrics — stopping")
                    break

                train_loss, _, _, _, train_f1, train_acc = train_metrics
                val_loss, _, _, _, val_f1, val_acc = val_metrics

                self.scheduler.step(val_loss)

                if val_f1 > self.best_f1:
                    self.best_f1 = val_f1
                    self.best_epoch = epoch + 1
                    self.epochs_no_improve = 0                 # ✦ reset counter
                    self.save_checkpoint(epoch + 1, val_f1)   # ✦ named checkpoint
                    self.log_best()
                else:
                    self.epochs_no_improve += 1                # ✦ increment counter

                print("\n" + "="*50)
                print(f"Run: {self.run_name}")
                print(f"Epoch {epoch+1}/{self.config.epochs}")
                print(f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")
                print(f"Best F1:  {self.best_f1:.4f} (epoch {self.best_epoch})")
                print(f"No improvement: {self.epochs_no_improve}/{self.config.early_stop_patience}")  # ✦
                print("="*50)

                # ✦ Early stopping check
                if self.epochs_no_improve >= self.config.early_stop_patience:    # ✦
                    print(f"\nEarly stopping triggered after {epoch+1} epochs.")  # ✦
                    break                                                          # ✦

        finally:
            self.end_run()


if __name__ == "__main__":

    config = Config()
    set_seed(config.seed)   # ✦
    trainer = Trainer(config)
    trainer.train()
    
