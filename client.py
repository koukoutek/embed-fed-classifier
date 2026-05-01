import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import nvflare.client as flare

from nvflare.fuel.utils.log_utils import get_script_logger
from monai.transforms import (Compose, LoadImaged, ScaleIntensityd, RandAdjustContrastd, Orientationd, RandGaussianSmoothd, RandFlipd, RandRotated,
                             RandShiftIntensityd, RandGaussianNoised, ThresholdIntensityd, RandAffined)
from monai.data import DataLoader, CacheDataset, Dataset
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from model import *
from utils import *


EMBED_ROOT = "/mnt/d/Users/kokouk/Projects/Data/EMBED"
Training_ROOT = "/mnt/d/Users/kokouk/Projects/EMBEDFedClassifier"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def error_raise(ex): raise Exception(ex)

def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset_path", type=str, default=EMBED_ROOT, nargs="?")
    parser.add_argument("--test_dataset_path", type=str, default=EMBED_ROOT, nargs="?")
    parser.add_argument("--batch_size", type=int, default=4, nargs="?")
    parser.add_argument("--learning_rate", type=float, default=0.0003, nargs="?")
    parser.add_argument("--local_epochs", type=int, default=5, nargs="?")
    parser.add_argument("--client_model_path", type=str, default=f"{Training_ROOT}/embed_net.pth", nargs="?")
    parser.add_argument("--global_model_path", type=str, default=f"{Training_ROOT}/embed_net_global.pth", nargs="?")
    parser.add_argument("--client_config_path", type=str, default="./client_config.yml", nargs="?")
    parser.add_argument("--client_cases", type=str, required=True, nargs="?")
    parser.add_argument("--workdir", type=str, default=Training_ROOT, nargs="?")

    return parser.parse_args()

def evaluate(model_args, input_weights, val_loader):
    logger.info("Evaluating model...")
    net = get_model(model_args)
    net.load_state_dict(input_weights)
    net.to(DEVICE)
    net.eval()

    val_predicted_total = []
    val_ground_total = []
    all_probs = []  
    all_labels = []

    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs = torch.cat([data['image']], dim=1).to(DEVICE)
            inputs = torch.squeeze(inputs, dim=len(inputs.shape) - 1)  # remove redundant channel dimension if exists
            labels = data['label'].to(DEVICE) 

            # breakpoint()
            # import matplotlib.pyplot as plt
            # stacked_imgs = np.hstack([inputs[j,0,:,:].cpu() for j in range(4)])
            # plt.imshow(stacked_imgs, cmap='gray')
            # plt.show()
            # continue

        try:
            outputs = net(inputs) # [0] needed only if using ViT model to get logits

            # Accuracy calculation
            _, predicted_classes = torch.max(outputs.detach().cpu(), 1)
            val_predicted_total.extend(predicted_classes.numpy().tolist())
            val_ground_total.extend(labels.detach().cpu().numpy().tolist())

            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of positive class
            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

        except Exception as e:
            print(f"Error during training: {e}")
    
    tn, fp, fn, tp = confusion_matrix(val_ground_total, val_predicted_total).ravel()
    balanced_accuray = balanced_accuracy_score(val_ground_total, val_predicted_total)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    auc = roc_auc_score(all_labels, all_probs)

    return balanced_accuray, specificity, sensitivity, auc

def main():
    # # Define local parameters
    args = define_parser()

    train_dataset_path = args.train_dataset_path
    test_dataset_path = args.test_dataset_path
    batch_size = args.batch_size
    lr = args.learning_rate
    local_config_path = args.client_config_path
    client_model_path = args.client_model_path
    global_model_path = args.global_model_path
    workdir = args.workdir

    local_cases = args.client_cases
    local_cases = local_cases.split(',') # list of case IDs
    local_config = load_config(local_config_path)

    train_transforms_config = local_config.get('train_transforms')
    eval_transforms_config = local_config.get('eval_transforms')

    train_transforms = Compose([
                            # # Load images and labels
                            LoadImaged(keys=train_transforms_config['LoadImaged_im']['keys'], ensure_channel_first=train_transforms_config['LoadImaged_im']['ensure_channel_first']),
                            # # Intensity normalization
                            ScaleIntensityd(keys=train_transforms_config['ScaleIntensityd']['keys']),
                            # # Image orientation
                            Orientationd(keys=train_transforms_config['Orientationd']['keys'], axcodes=train_transforms_config['Orientationd']['axcodes'], 
                                         as_closest_canonical=train_transforms_config['Orientationd']['as_closest_canonical']),
                            # # Augmentations
                            RandFlipd(keys=train_transforms_config['RandFlipd_x']['keys'], prob=train_transforms_config['RandFlipd_x']['prob'], spatial_axis=train_transforms_config['RandFlipd_x']['spatial_axis']),
                            RandFlipd(keys=train_transforms_config['RandFlipd_y']['keys'], prob=train_transforms_config['RandFlipd_y']['prob'], spatial_axis=train_transforms_config['RandFlipd_y']['spatial_axis']),
                            RandRotated(keys=train_transforms_config['RandRotated']['keys'], prob=train_transforms_config['RandRotated']['prob'], 
                                        range_x=train_transforms_config['RandRotated']['range_x'], range_y=train_transforms_config['RandRotated']['range_y'], mode=train_transforms_config['RandRotated']['mode']),
                            RandAffined(keys=train_transforms_config['RandAffined']['keys'], prob=train_transforms_config['RandAffined']['prob'], 
                                        rotate_range=train_transforms_config['RandAffined']['rotate_range'], scale_range=train_transforms_config['RandAffined']['scale_range'], mode=train_transforms_config['RandAffined']['mode']),
                            # RandAdjustContrastd(keys=train_transforms_config['RandAdjustContrastd']['keys'], prob=train_transforms_config['RandAdjustContrastd']['prob'], gamma=train_transforms_config['RandAdjustContrastd']['gamma']),
                            # RandGaussianSmoothd(keys=train_transforms_config['RandGaussianSmoothd']['keys'], prob=train_transforms_config['RandGaussianSmoothd']['prob'], sigma_x=train_transforms_config['RandGaussianSmoothd']['sigma_x'], sigma_y=train_transforms_config['RandGaussianSmoothd']['sigma_y']),
                            # RandShiftIntensityd(keys=train_transforms_config['RandShiftIntensityd']['keys'], prob=train_transforms_config['RandShiftIntensityd']['prob'], offsets=train_transforms_config['RandShiftIntensityd']['offsets']),
                            # RandGaussianNoised(keys=train_transforms_config['RandGaussianNoised']['keys'], prob=train_transforms_config['RandGaussianNoised']['prob'], mean=train_transforms_config['RandGaussianNoised']['mean'], std=train_transforms_config['RandGaussianNoised']['std']),
                            # ThresholdIntensityd(keys=train_transforms_config['ThresholdIntensityd_clip_upper']['keys'], threshold=train_transforms_config['ThresholdIntensityd_clip_upper']['threshold'], above=train_transforms_config['ThresholdIntensityd_clip_upper']['above'], cval=train_transforms_config['ThresholdIntensityd_clip_upper']['cval']),
                            # ThresholdIntensityd(keys=train_transforms_config['ThresholdIntensityd_clip_lower']['keys'], threshold=train_transforms_config['ThresholdIntensityd_clip_lower']['threshold'], above=train_transforms_config['ThresholdIntensityd_clip_lower']['above'], cval=train_transforms_config['ThresholdIntensityd_clip_lower']['cval']),

    ])
    val_transforms = Compose([
                            # # Load images and labels
                            LoadImaged(keys=eval_transforms_config['LoadImaged_im']['keys'], ensure_channel_first=eval_transforms_config['LoadImaged_im']['ensure_channel_first']),
                            # # Intensity normalization
                            ScaleIntensityd(keys=eval_transforms_config['ScaleIntensityd']['keys']),
                            # # Image orientation
                            Orientationd(keys=eval_transforms_config['Orientationd']['keys'], axcodes=eval_transforms_config['Orientationd']['axcodes'], 
                                         as_closest_canonical=eval_transforms_config['Orientationd']['as_closest_canonical']),
    ])

    print(f"Loading data from {train_dataset_path}...")
    print(f"Using batch size: {batch_size}")
    print(f"Using learning rate: {lr}")

    # print("Before hitting breakpoint")
    # import pdb;
    # pdb.set_trace()
    # or use breakpoint() 

    # # Define data dictionaries based on local cases
    train_cases = [case for case in Path(train_dataset_path).rglob("*") if case.is_file()]
    train_cases = [case for case in train_cases if case.parents[0].name in local_cases]
    val_cases = []
    test_cases = [case for case in Path(test_dataset_path).rglob("*") if case.is_file()]

    benign_cases = [c for c in train_cases if 'benign' in str(c)]
    malignant_cases = [c for c in train_cases if 'malignant' in str(c)]

    val_num_benign = int(0.1 * len(benign_cases))
    val_num_malignant = int(0.1 * len(malignant_cases))
    val_cases.extend(random.sample(benign_cases, val_num_benign))
    val_cases.extend(random.sample(malignant_cases, val_num_malignant))

    train_cases = [c for c in train_cases if c not in val_cases]
    benign_cases = [c for c in train_cases if 'benign' in str(c)]
    malignant_cases = [c for c in train_cases if 'malignant' in str(c)]

    w_benign = (len(train_cases)) / (2 * len(benign_cases))
    w_malignant = (len(train_cases)) / (2 * len(malignant_cases))
    print(f"Number of training cases: {len(train_cases)} (Benign: {len(benign_cases)}, Malignant: {len(malignant_cases)})")
    print(f"Class weights - Benign: {w_benign:.4f}, Malignant: {w_malignant:.4f}")
    w_class = torch.tensor([w_benign, w_malignant])

    train_dict = [{"image": im , "label": 0 if "benign" in str(im) else 1} for im in train_cases]
    val_dict = [{"image": im , "label": 0 if "benign" in str(im) else 1} for im in val_cases]
    test_dict = [{"image": im , "label": 0 if "benign" in str(im) else 1} for im in test_cases]

    # # Define dataset
    logger.info("Preparing datasets...")
    if local_config['dataset'] == 'Dataset':
        train_dataset = Dataset(data=train_dict, transform=train_transforms)
        val_dataset = Dataset(data=val_dict, transform=val_transforms)
        test_dataset = Dataset(data=test_dict, transform=val_transforms)
    elif local_config['dataset'] == 'CacheDataset':
        train_dataset = CacheDataset(data=train_dict, transform=train_transforms, cache_rate=local_config['cache_rate'])
        val_dataset = CacheDataset(data=val_dict, transform=val_transforms, cache_rate=local_config['cache_rate'])
        test_dataset = CacheDataset(data=test_dict, transform=val_transforms, cache_rate=local_config['cache_rate'])

    # # Initialize NVFlare client
    flare.init() 

    logger.info(f"({flare.get_site_name()}) Number of training samples: {len(train_dataset)}")
    logger.info(f"({flare.get_site_name()}) Number of validation samples: {len(val_dataset)}")
    logger.info(f"({flare.get_site_name()}) Number of test samples: {len(test_dataset)}")

    # # Initialize DataLoader
    train_loader = DataLoader(train_dataset, batch_size=local_config.get('dataloader').get('batch_size', 1),
                              shuffle=local_config.get('dataloader').get('shuffle', False), 
                              num_workers=local_config.get('dataloader').get('num_workers', 1),
                              pin_memory=local_config.get('dataloader').get('pin_memory', True))
    val_loader = DataLoader(val_dataset, batch_size=local_config.get('dataloader').get('batch_size', 1), 
                            shuffle=local_config.get('dataloader').get('shuffle', False),
                            num_workers=local_config.get('dataloader').get('num_workers', 1),
                            pin_memory=local_config.get('dataloader').get('pin_memory', True))
    test_loader = DataLoader(test_dataset, batch_size=local_config.get('dataloader').get('batch_size', 1), 
                             shuffle=local_config.get('dataloader').get('shuffle', False),
                             num_workers=local_config.get('dataloader').get('num_workers', 1),
                             pin_memory=local_config.get('dataloader').get('pin_memory', True))

    # # Get model from config and initialize model object
    model_args = local_config.get('model') if local_config.get('model') else error_raise("model_args must be provided")
    net = get_model(model_args)

    local_epochs = local_config['local_epochs']
    best_metric = local_config['save_model_when']

    while flare.is_running():

        input_model = flare.receive()
        client_id = flare.get_site_name()

        if flare.is_train():
            logger.info(f"({client_id}) current_round={input_model.current_round}, total_rounds={input_model.total_rounds}")
            net.load_state_dict(input_model.params) # Load received global model weights

            if input_model.current_round > 0 and client_id == 'site-1':
                logger.info(f"({client_id}) Performing testing with current global model before local training...")
                test_metrics = evaluate(model_args, net.state_dict(), test_loader)
                logger.info(f"({client_id}) -- Balanced accuracy on test set: {test_metrics[0]:.4f}")
                logger.info(f"({client_id}) -- Specificity on test set: {test_metrics[1]:.4f}")
                logger.info(f"({client_id}) -- Sensitivity on test set: {test_metrics[2]:.4f}")
                logger.info(f"({client_id}) -- AUC on test set: {test_metrics[3]:.4f}")
                logger.info(f"Saving global model to {global_model_path}_{input_model.current_round}...")
                # torch.save(net.state_dict(), f"{global_model_path}_{input_model.current_round}")
                best_metric = test_metrics[0]  # Update best metric based on test set performance
                                                                               
            # # Define loss function
            if local_config.get('loss'):
                if local_config.get('loss').get('name') == 'CrossEntropyLoss':
                    criterion = nn.CrossEntropyLoss(label_smoothing=local_config.get('loss').get('label_smoothing'),
                                                    weight=w_class.to(DEVICE) if local_config.get('loss').get('class_weights', False) else None
                                                    )
                elif local_config.get('loss').get('name') == 'BCEWithLogitsLoss':
                    criterion = nn.BCEWithLogitsLoss()
                elif local_config.get('loss').get('name') == 'BCELoss':
                    criterion = nn.BCELoss()
                logger.info(f"({client_id}) Using loss function: {local_config.get('loss').get('name')}")
            else:
                error_raise("Loss function must be provided")

            # # Use different learning rate for backbone and classifier head if specified
            # # Separate parameters
            backbone_params = []
            finetune_params = []
            for name, param in net.named_parameters():
                if 'classifier' in name or 'features.0.0' in name or 'head' in name:
                    finetune_params.append(param)
                else:
                    backbone_params.append(param)

            # # Define optimizer
            if local_config.get('optimizer'):
                if local_config.get('optimizer').get('name') == 'SGD':
                    optimizer = optim.SGD([
                        {'params': backbone_params, 'lr': local_config.get('optimizer').get('lr_backbone', lr)},
                        {'params': finetune_params, 'lr': local_config.get('optimizer').get('lr_finetune', lr)}
                    ], momentum=local_config.get('optimizer').get('momentum'), weight_decay=local_config.get('optimizer').get('weight_decay'))
                elif local_config.get('optimizer').get('name') == 'Adam':
                    optimizer = optim.Adam([
                        {'params': backbone_params, 'lr': local_config.get('optimizer').get('lr_backbone', lr)},
                        {'params': finetune_params, 'lr': local_config.get('optimizer').get('lr_finetune', lr)}
                    ], weight_decay=local_config.get('optimizer').get('weight_decay'))
                logger.info(f"({client_id}) Using optimizer: {local_config.get('optimizer').get('name')}")
            else:
                error_raise("Optimizer must be provided")
            
            # # Define metric
            if local_config.get('metric'):
                if local_config.get('metric').get('name') == 'Accuracy':
                    logger.info(f"({client_id}) Using metric: {local_config.get('metric').get('name')}")
            else:
                error_raise("Metric must be provided")

            # # Send model to device
            net.to(DEVICE)
            steps = local_epochs * len(train_loader)

            training_loss = []
            logger.info(f"({client_id}) Starting Training for {local_epochs} epochs...")

            local_epochs = 40 if local_config.get('personalized', False) and input_model.current_round == 19 else local_epochs  # If personalized and best metric is high, train for more epochs
            for epoch in range(local_epochs):  # loop over the dataset multiple times
                logger.info(f"-------------------------------------------")
                logger.info(f"({client_id}) Epoch {epoch + 1}...")
                running_loss = 0.0
                predicted_total = []
                ground_total = []

                for i, data in enumerate(train_loader, 0):
                    inputs = torch.cat([data['image']], dim=1).to(DEVICE)
                    inputs = torch.squeeze(inputs, dim=len(inputs.shape) - 1)  # remove redundant channel dimension if exists
                    labels = data['label'].to(DEVICE) 

                    # # For debugging: visualize input images
                    # # IMAGES LOOK QUITE DIFFERENT IN TERMS OF INTENSITY DISTRIBUTION 
                    # breakpoint()
                    # import matplotlib.pyplot as plt
                    # stacked_imgs = np.hstack([inputs[j,0,:,:].cpu() for j in range(4)])
                    # plt.imshow(stacked_imgs, cmap='gray')
                    # plt.show()
                    # continue

                    try:
                        optimizer.zero_grad()

                        outputs = net(inputs) # [0] needed only if using ViT model to get logits

                        # print(f'Out: {outputs}, Labels: {labels}') # check outputs and labels size and values for debugging
                        loss = criterion(outputs, labels)
                        loss.backward()

                        optimizer.step()

                        # Accuracy calculation
                        _, predicted_classes = torch.max(outputs.detach().cpu(), 1)
                        labels = labels.detach().cpu()
                        predicted_total.extend(predicted_classes.numpy().tolist())
                        ground_total.extend(labels.numpy().tolist())

                    except Exception as e:
                        print(f"({client_id}) Error during training: {e}")

                    running_loss += loss.item()
                epoch_accuracy = balanced_accuracy_score(ground_total, predicted_total)
                epoch_confusion = confusion_matrix(ground_total, predicted_total)
                epoch_sensitivity = epoch_confusion[1,1] / (epoch_confusion[1,1] + epoch_confusion[1,0])
                epoch_specificity = epoch_confusion[0,0] / (epoch_confusion[0,0] + epoch_confusion[0,1])
                epoch_loss = running_loss / len(train_loader)
                training_loss.append(epoch_loss)

                logger.info(f"Epoch loss: {epoch_loss:.4f}")
                logger.info(f"Epoch balanced accuracy: {epoch_accuracy:.4f}")
                logger.info(f"Epoch specificity: {epoch_specificity:.4f}")
                logger.info(f"Epoch sensitivity: {epoch_sensitivity:.4f}")
                logger.info(f"-------------------------------------------")

                if (epoch+1) % local_config.get('val_interval', 10) == 0:
                    local_metrics = evaluate(model_args, net.state_dict(), val_loader)
                    logger.info(f"({client_id}) -- Balanced accuracy on validation set: {local_metrics[0]:.4f}")
                    logger.info(f"({client_id}) -- Specificity on validation set: {local_metrics[1]:.4f}")
                    logger.info(f"({client_id}) -- Sensitivity on validation set: {local_metrics[2]:.4f}")
                    logger.info(f"({client_id}) -- AUC on validation set: {local_metrics[3]:.4f}")
                    if local_metrics[0] > best_metric:
                        logger.info(f"({client_id}) New best model found {local_metrics[0]:.4f} at round {input_model.current_round} (previous best was {best_metric:.4f})")
                        best_metric = local_metrics[0]
                    logger.info(f"Saving model to {client_model_path}_{input_model.current_round}...")

                if (epoch+1) % 5 == 0:
                    logger.info(f"({client_id}) Saving model to {workdir}/{client_id}/embed_net_round_{input_model.current_round}_epoch_{epoch+1}.pth...")
                    torch.save(net.state_dict(), f"{workdir}/{client_id}/embed_net_round_{input_model.current_round}_epoch_{epoch+1}.pth")

            logger.info(f"({client_id}) Finished Training")

            output_model = flare.FLModel(
                params=net.cpu().state_dict(),
                # metrics={"metric": global_model_metric},
                meta={"NUM_STEPS_CURRENT_ROUND": steps},
            )

            flare.send(output_model)

        # elif flare.is_evaluate():
        #     global_model_metric = evaluate(net, input_model.params)
        #     print(f"({client_id}) Metric: {global_model_metric}")
        #     flare.send(flare.FLModel(metrics={"metric": global_model_metric}))

        # elif flare.is_submit_model():
        #     model_name = input_model.meta["submit_model_name"]
        #     if model_name == ModelName.BEST_MODEL:
        #         try:
        #             weights = torch.load(model_path)
        #             net = get_model(local_config.get('model'))
        #             net.load_state_dict(weights)
        #             flare.send(flare.FLModel(params=net.cpu().state_dict()))
        #         except Exception as e:
        #             error_raise("Unable to load best model")
        #     else:
        #         error_raise(f"Unknown submit model name: {model_name}")


if __name__ == "__main__":
    logger = get_script_logger()
    main()