import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import sys 
import json

current_dir = '/share/projects/TaskTracker'
sys.path.append(current_dir)

from training.triplet_probe.models.processing_per_layer import ParallelConvProcessingModel
from training.dataset import ActivationsDatasetDynamicReturnText, ActivationsDatasetDynamicPrimaryText
from training.triplet_probe.loss_functions.triplet_loss import TripletLoss ,triplet_mining_unique
from training.helpers.data import load_file_paths, process_val_data
from training.helpers.training import load_checkpoint, save_checkpoint, compute_ROC_AUC
from training.utils.constants import CONSTANTS_ALL_MODELS, MODEL_OUTPUT_DIR, OOD_POISONED_FILE,TRAINING_TEXT_DIR

MODEL = 'mistral'
ACTIVATIONS_DIR, ACTIVATIONS_VAL_DIR, ACTIVATION_FILE_LIST_DIR =\
    CONSTANTS_ALL_MODELS[MODEL]['ACTIVATIONS_DIR'], CONSTANTS_ALL_MODELS[MODEL]['ACTIVATIONS_VAL_DIR'], CONSTANTS_ALL_MODELS[MODEL]['ACTIVATION_FILE_LIST_DIR']

print('== Running latest file ==')
# Configuration settings
config = {
    'model': MODEL,
    'activations': ACTIVATIONS_DIR,
    'activations_ood': ACTIVATIONS_VAL_DIR,
    'ood_poisoned_file': OOD_POISONED_FILE, 
    'exp_name': 'mistral_test',
    'margin': 0.3,
    'epochs': 6,
    'num_layers': (0,5),
    'files_chunk': 10,
    'batch_size': 2500,
    'learning_rate': 0.0005,
    'restart': False,  # Set to True if restarting from a checkpoint
    'feature_dim' : 275,
    'pool_first_layer': 5 if MODEL == 'llama3_70b' else 3,
    'dropout' : 0.5,
    'check_each' : 50,
    'conv': True,
    'layer_norm': False,
    'delay_lr_factor': 0.95,
    'delay_lr_step': 800
}

# Change feature_dim of some models 
if MODEL == 'llama3_70b':
    config['feature_dim'] = 350
elif MODEL == 'phi3':
    config['feature_dim'] = 175 
    
# Ensure output directory exists
config['out_dir'] = os.path.join(MODEL_OUTPUT_DIR, f'{config['exp_name']}')

os.makedirs(config.get('out_dir'), exist_ok=True)
with open(os.path.join(config.get('out_dir'), 'config.json'), 'w') as f:
    json.dump(config, f)

# Load training, test, and validation files
train_files = load_file_paths(os.path.join(ACTIVATION_FILE_LIST_DIR, 'train_files_' + MODEL + '.txt'))

val_files_clean = load_file_paths(os.path.join(ACTIVATION_FILE_LIST_DIR, 'val_clean_files_' + MODEL + '.txt'))
val_files_poisoned = load_file_paths(os.path.join(ACTIVATION_FILE_LIST_DIR, 'val_poisoned_files_' + MODEL + '.txt'))


# Model, Optimizer, and Loss Function Setup
model = ParallelConvProcessingModel(feature_dim=config.get('feature_dim'),num_layers=config.get('num_layers'),conv=config.get('conv'),layer_norm=config.get('layer_norm'), pool_first_layer = config.get('pool_first_layer')).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate'))
scheduler = StepLR(optimizer, step_size=config.get('delay_lr_step'), gamma=config.get('delay_lr_factor'))


# Triplet loss with hard and semi-hard mining
triplet_loss = TripletLoss(config.get('margin'))

# Optionally load from checkpoint
global_counter_for_save = 1
start_epoch = 0
best_roc_auc = 0

if config['restart']:
    checkpoint_file = os.path.join(config.get('out_dir'), f'epoch_model_{start_epoch}_checkpoint.pth')
    if os.path.exists(checkpoint_file):
        start_epoch, best_roc_auc = load_checkpoint(checkpoint_file, model, optimizer)
        print(f"Restarting from epoch {start_epoch} with best roc auc {best_roc_auc}")
        
def one_epoch_train(epoch_num, model, loss_function, optimizer, scheduler, train_files, config):
    global global_counter_for_save
    global best_roc_auc
    model.train()
    step = 0
    total_loss = 0
    total_batches = 0
    batch_size = 1024
    
    for i in range(0, len(train_files), config['files_chunk']):
        chunk_files = train_files[i: i + config['files_chunk']]
        dataset = ActivationsDatasetDynamicReturnText(chunk_files, config['activations'],TRAINING_TEXT_DIR, num_layers=config['num_layers'])
        training_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
        
        for data in tqdm(training_loader):
            optimizer.zero_grad()

            # Triplet mining and loss computation
            step += 1
            with torch.no_grad():
                primary, clean, poisoned, text_batch = data[0].cuda(), data[1].cuda(), data[2].cuda(), data[3]
                print(primary.size())
                print(len(text_batch))
                # For models that are read as float16 
                with torch.torch.autocast(device_type='cuda', dtype=torch.float32):
                    primary_embs = model(primary)
                    clean_embs = model(clean)
                    poisoned_embs = model(poisoned)
                triplet_combinations = triplet_mining_unique(primary_embs, clean_embs, poisoned_embs, config['margin'],text_batch, hard=True if global_counter_for_save > 3000 else False, step=step)
                print(len(triplet_combinations))
            
            for k in range(0, len(triplet_combinations), batch_size):
                # Extract embeddings based on mined indices
                indices_clean = triplet_combinations[k:k+batch_size,0]
                indices_poisoned = triplet_combinations[k:k+batch_size,1]
                
                # Batches of primary, clean, secondary
                anchor_embeddings = primary[indices_clean,:]
                positive_embeddings = clean[indices_clean,:]
                negative_embeddings = poisoned[indices_poisoned,:]
                
                # Forward 
                # For models that are read as float16 
                with torch.torch.autocast(device_type='cuda', dtype=torch.float32):
                    anchor_emb_output = model(anchor_embeddings)
                    positive_emb_output = model(positive_embeddings)
                    negative_emb_output = model(negative_embeddings)
                
                
                # Calculate loss for the selected triplets
                loss = loss_function(anchor_emb_output, positive_emb_output, negative_emb_output, step)
                loss = loss * (anchor_emb_output.size(0)/batch_size)
                loss.backward()
                total_loss += loss.item()
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_batches += 1
                
                global_counter_for_save += 1 
                if global_counter_for_save % config['check_each'] == 0:
                    roc_auc = validation_ood(model, val_files_clean, val_files_poisoned, config) 
                    if roc_auc > best_roc_auc:
                        print('=== New best model ===')
                        best_roc_auc = roc_auc
                        save_checkpoint({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_roc_auc': best_roc_auc
                        }, config['out_dir'], f'best_model_checkpoint.pth')

    avg_loss = (total_loss / total_batches) if total_batches > 0 else 0
    print(f'Epoch {epoch_num+1}: Training Loss: {avg_loss}')


def compute_distances_validation(model, val_files, config):
    distances = []
    for i in range(0, len(val_files), config.get('files_chunk')):
        chunk_files = val_files[i: i + config.get('files_chunk')]
        dataset = ActivationsDatasetDynamicPrimaryText(chunk_files, num_layers=config['num_layers'], root_dir=config.get('activations_ood'))
        val_loader = DataLoader(dataset, batch_size=config.get('batch_size'), shuffle=False)
        
        for data in val_loader:
            primary, primary_with_text = [d.cuda() for d in data]
            # For models that are read as float16 
            with torch.torch.autocast(device_type='cuda', dtype=torch.float32):
                primary_embs = model(primary)
                primary_with_text_embs = model(primary_with_text) 
            distances.extend(torch.norm(primary_embs - primary_with_text_embs, p=2, dim=-1).cpu().numpy())
    return distances 
    
    
def validation_ood(model, val_files_clean, val_files_poisoned, config):
    model.eval()
    
    with torch.no_grad():  # Disable gradient computation
        distances_clean = compute_distances_validation(model, val_files_clean, config)
        distances_poisoned = compute_distances_validation(model, val_files_poisoned, config)
    
    distances_poisoned = process_val_data(config.get('ood_poisoned_file'),distances_poisoned)
    roc_auc = compute_ROC_AUC(distances_clean,distances_poisoned)
        
    print(f'Training step {global_counter_for_save+1}: ROC AUC on OOD data: {roc_auc}')
    model.train()
    return roc_auc

for epoch in range(start_epoch, config.get('epochs')):
    one_epoch_train(epoch, model, triplet_loss, optimizer, scheduler, train_files, config)
        

