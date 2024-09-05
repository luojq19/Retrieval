import sys
sys.path.append('.')
import torch
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import os, json, argparse, time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from utils import commons
from models.mlp import MLP
from datasets.sequence_dataset import SequenceDataset


def infer(model, data_loader, device):
    model.eval()
    all_output, all_pred = [], []
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            output = model(data)
            all_output.append(commons.toCPU(output))
            all_pred.append(commons.toCPU(output).argmax(dim=1))
        all_output = torch.cat(all_output, dim=0)
        all_pred = torch.cat(all_pred)
        
    return all_output, all_pred

def evaluate(model, val_loader, criterion, device):
    model.eval()
    all_loss = []
    all_output = []
    with torch.no_grad():
        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)
            all_loss.append(commons.toCPU(loss).item())
            all_output.append(commons.toCPU(output))
        all_loss = torch.tensor(all_loss)
    model.train()
    
    return all_loss.mean().item()

def train(model, train_loader, val_loader, criterion, optimizer, lr_scheduler, device, logger, config, writer=None):
    model.train()
    n_bad = 0
    all_loss = []
    all_val_loss = []
    best_val_loss = 1.e10
    epsilon = 1e-4
    for epoch in range(config.num_epochs):
        # input()
        start_epoch = time.time()
        val_loss = evaluate(model, val_loader, criterion, device)
        if val_loss > best_val_loss - epsilon:
            n_bad += 1
            if n_bad > config.patience:
                logger.info(f'No performance improvement for {config.patience} epochs. Early stop training!')
                break
        else:
            logger.info(f'New best performance found! val_loss={val_loss:.4f}')
            n_bad = 0
            best_val_loss = val_loss
            state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(config.ckpt_dir, 'best_checkpoints.pt'))
        all_val_loss.append(val_loss)
        losses = []
        for data, label in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}', dynamic_ncols=True):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)
            losses.append(commons.toCPU(loss).item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        mean_loss = torch.tensor(losses).mean().item()
        all_loss.append(mean_loss)
        lr_scheduler.step(mean_loss)
        state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(config.ckpt_dir, 'last_checkpoints.pt'))
        all_checkpoints = {'epoch': epoch, 'model': state_dict, 'optimizer': optimizer.state_dict(), 'criterion': criterion.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(), 'n_bad': n_bad, 'best_val_loss': best_val_loss}
        torch.save(all_checkpoints, os.path.join(config.ckpt_dir, 'all_checkpoints.pt'))
        end_epoch = time.time()
        logger.info(f'Epoch [{epoch + 1}/{config.num_epochs}]: loss: {mean_loss:.4f}; val_loss: {val_loss:.4f}; train time: {commons.sec2min_sec(end_epoch - start_epoch)}')
        writer.add_scalar('Train/loss', mean_loss, epoch)
        writer.add_scalar('Val/loss', val_loss, epoch)
        
    return all_loss, all_val_loss

class CustomSubset(Subset):
    def __init__(self, dataset, indices):
        super(CustomSubset, self).__init__(dataset, indices)
        self.copy_attributes(dataset)

    def copy_attributes(self, dataset):
        for attr in dir(dataset):
            # Make sure we're only copying relevant attributes
            # You might want to exclude methods or system attributes starting with '__'
            if not attr.startswith('__') and not callable(getattr(dataset, attr)):
                setattr(self, attr, getattr(dataset, attr))
                
def merge_config_args(config, args):
    config.train.seed = args.seed if args.seed is not None else config.train.seed
    config.train.lr = args.lr if args.lr is not None or not hasattr(config.train, 'lr') else config.train.lr
    config.train.weight_decay = args.weight_decay if args.weight_decay is not None or not hasattr(config.train, 'weight_decay') else config.train.weight_decay
    config.train.batch_size = args.batch_size if args.batch_size is not None or not hasattr(config.train, 'batch_size') else config.train.batch_size

def get_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', type=str, default='configs/train.yml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--logdir', type=str, default='logs_ec')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--no_timestamp', action='store_true')
    parser.add_argument('--resume_model_dir', type=str, default=None)
    parser.add_argument('--random_split_train_val', action='store_true')
    
    
    return parser.parse_args()


def main():
    start_overall = time.time()
    args = get_args()
    
    # Load configs
    if args.resume_model_dir is not None:
        print(f'Resuming training from {args.resume_model_dir}')
        config = commons.load_config(os.path.join(args.resume_model_dir, 'config.yml'))
    else:
        config = commons.load_config(args.config)
        config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    merge_config_args(config, args)
    commons.seed_all(config.train.seed)
    
    # Logging
    if args.resume_model_dir is not None:
        log_dir = args.resume_model_dir
    else:   
        log_dir = commons.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag, timestamp=not args.no_timestamp)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    config.train.ckpt_dir = ckpt_dir
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = commons.get_logger('train_mlp', log_dir)
    writer = SummaryWriter(log_dir)
    if args.resume_model_dir is None:
        logger.info(f'Resume training from {args.resume_model_dir}')
    else:
        logger.info(f'Start training for {log_dir}')
    logger.info(args)
    logger.info(config)
    
    # Dataset
    if args.random_split_train_val or not hasattr(config.data, 'valid_data_file'):
        logger.info('Randomly split train and validation set')
        all_data = globals()[config.data.dataset_type](config.data.train_data_file, config.data.label_file, config.data.label_name,  logger=logger)
        num_train_val = len(all_data)
        indices = commons.get_random_indices(num_train_val, seed=config.train.seed)
        train_indices = indices[:int(num_train_val * 0.875)]
        val_indices = indices[int(num_train_val * 0.875):]
        all_pids = all_data.pids
        train_pids, val_pids = [all_pids[i] for i in train_indices], [all_pids[i] for i in val_indices]
        with open(os.path.join(log_dir, 'train_val_pids.json'), 'w') as f:
            json.dump({'train': train_pids, 'val': val_pids}, f)
        trainset = CustomSubset(all_data, train_indices)
        validset = CustomSubset(all_data, val_indices)
    else:
        trainset = globals()[config.data.dataset_type](config.data.train_data_file, config.data.label_file, config.data.label_name,  logger=logger)
        validset = globals()[config.data.dataset_type](config.data.valid_data_file, config.data.label_file, config.data.label_name,  logger=logger)
    testset = globals()[config.data.dataset_type](config.data.test_data_file, config.data.label_file, config.data.label_name, logger=logger)
    train_loader = DataLoader(trainset, batch_size=config.train.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(validset, batch_size=config.train.batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=config.train.batch_size, shuffle=False)
    config.model.out_dim = trainset.num_labels
    logger.info(f'Trainset size: {len(trainset)}; Validset size: {len(validset)}; Testset size: {len(testset)}')
    logger.info(f'Number of labels: {trainset.num_labels}')
    
    # load all checkpoints if resuming
    if args.resume_model_dir is not None:
        all_checkpoints = torch.load(os.path.join(args.resume_model_dir, 'checkpoints', 'all_checkpoints.pt'), map_location=args.device)
        
    # Model
    model = globals()[config.model.model_type](config.model)
    model.to(args.device)
    if args.resume_model_dir is not None:
        model.load_state_dict(all_checkpoints['model'])
        logger.info(f'Model loaded from {os.path.join(args.resume_model_dir, "checkpoints", "all_checkpoints.pt")}')
        model.to(args.device)
        model.train()
    logger.info(model)
    logger.info(f'Trainable parameters: {commons.count_parameters(model)}')
    
    # Train
    criterion = globals()[config.train.loss]()
    optimizer = globals()[config.train.optimizer](model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=config.train.patience-10, verbose=True)
    
    if args.resume_model_dir is not None:
        optimizer.load_state_dict(all_checkpoints['optimizer'])
        criterion.load_state_dict(all_checkpoints['criterion'])
        lr_scheduler.load_state_dict(all_checkpoints['lr_scheduler'])
        logger.info(f'Optimizer, criterion, lr_scheduler loaded from {os.path.join(args.resume_model_dir, "checkpoints", "all_checkpoints.pt")}')
        
    commons.save_config(config, os.path.join(log_dir, 'config.yml'))
    
    train(model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler, device=args.device, logger=logger, config=config.train, writer=writer)
    
    # Test
    best_ckpt = torch.load(os.path.join(config.train.ckpt_dir, 'best_checkpoints.pt'))
    model.load_state_dict(best_ckpt)
    testset = globals()[config.data.dataset_type](config.data.test_data_file, config.data.label_file, config.data.label_name, logger=logger)
    test_loader = DataLoader(testset, batch_size=config.train.batch_size, shuffle=False)
    logger.info(f'Number of test sequences: {len(testset)}')
    
    logits, preds = infer(model, test_loader, args.device)
    preds = F.one_hot(preds, num_classes=config.model.out_dim)
    ground_truth = testset.labels
    logger.info(f'Evaluation on test set {config.data.test_data_file}')
    accuracy = accuracy_score(ground_truth, preds)
    precision = precision_score(ground_truth, preds, average='micro')
    recall = recall_score(ground_truth, preds, average='micro')
    f1 = f1_score(ground_truth, preds, average='micro')
    logger.info(f'Accuracy: {accuracy:.4f}')
    logger.info(f'Precision: {precision:.4f}')
    logger.info(f'Recall: {recall:.4f}')
    logger.info(f'F1: {f1:.4f}')

    # Test 2
    if hasattr(config.data, 'test_data_file2'):
        testset2 = globals()[config.data.dataset_type](config.data.test_data_file2, config.data.label_file, config.data.label_name, logger=logger)
        test_loader2 = DataLoader(testset2, batch_size=config.train.batch_size, shuffle=False)
        logger.info(f'Number of test sequences: {len(testset2)}')
        
        logits, preds = infer(model, test_loader2, args.device)
        preds = F.one_hot(preds, num_classes=config.model.out_dim)
        ground_truth = testset2.labels
        logger.info(f'Evaluation on test set {config.data.test_data_file2}')
        accuracy = accuracy_score(ground_truth, preds)
        precision = precision_score(ground_truth, preds, average='micro')
        recall = recall_score(ground_truth, preds, average='micro')
        f1 = f1_score(ground_truth, preds, average='micro')
        logger.info(f'Accuracy: {accuracy:.4f}')
        logger.info(f'Precision: {precision:.4f}')
        logger.info(f'Recall: {recall:.4f}')
        logger.info(f'F1: {f1:.4f}')


    end_overall = time.time()
    logger.info(f'Elapse time: {commons.sec2hr_min_sec(end_overall - start_overall)}')

if __name__ == '__main__':
    main()
