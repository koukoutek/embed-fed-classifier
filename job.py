import os
import argparse
import pandas as pd

from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.fuel.utils.log_utils import get_script_logger
from pathlib import Path
from model import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning Job Runner")
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file')
    return parser.parse_args()


if __name__ == "__main__":
    logger = get_script_logger()
    working_dir = os.getcwd()

    args = parse_args()
    config = load_config(args.config)

    experiment_config_dir = Path(f"{config.get('workdir')}_config")
    if not os.path.exists(experiment_config_dir):
        os.makedirs(experiment_config_dir)

    # To run multiple experiments, save a copy of the config file in the experiment directory
    save_config(config, os.path.join(experiment_config_dir, 'server_config.yml'))
    client_config = load_config(config.get('client_config_path'))
    save_config(client_config, os.path.join(experiment_config_dir, 'client_config.yml'))
    
    n_clients = config.get('n_clients')
    num_rounds = config.get('num_rounds')   
    train_script = config.get('client_script')
    
    seed = config.get('seed', 0)
    set_seed(manual_seed=seed)

    model_args = config.get('model')
    model = get_model(model_args)

    if config.get('pretrained', False):
        pretrained_model_path = config.get('pretrained_model_path')
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            model.load_state_dict(torch.load(pretrained_model_path))
            logger.info(f"Loaded pretrained model from {pretrained_model_path}")
        else:
            logger.warning(f"Pretrained model path {pretrained_model_path} does not exist. Proceeding without loading pretrained weights.")

    if config.get('recipe') == 'fedavg':
        job = FedAvgJob(name='fedavg', n_clients=n_clients, num_rounds=num_rounds, initial_model=model)

    train_data_path = Path(config.get('train_dataset_path'))
    test_data_path = Path(config.get('test_dataset_path'))
    meta = pd.read_csv(config.get('meta_data_path'))

    cases = [c for c in train_data_path.rglob('*') if c.is_file()]
    series_instance_uids = [c.parents[0].name for c in cases] # there are 4 duplicates. all is 7351

    # get duplicated
    import collections
    duplicated = [item for item, count in collections.Counter(series_instance_uids).items() if count > 1]

    #######################################################################################################
    #######################################################################################################
    # Reading meta from /mnt/d/Users/kokouk/Projects/EMBEDFedClassifier/metadata/EMBED_cleaned_metadata.csv
    meta['anon_dicom_path'] = meta['anon_dicom_path'].astype(str).str.split('/').str[-1]
    meta['anon_dicom_path'] = meta['anon_dicom_path'].astype(str).str.replace('.dcm', '', regex=False)
    meta = meta[meta['anon_dicom_path'].isin(series_instance_uids)] # this returns 7647 (duplicates are out)

    meta = meta[meta['loc_num'].isin(config.get('client_list'))] # client_list = [1,2,3,5,6,7,8,9] # if only 4 largest sites: [1,2,5,6] - losing around 200 cases
    meta = meta[meta['asses'].isin(['N', 'B', 'M', 'K'])]  # binary classification - losing around 200 cases

    meta_site_dicom = meta[['loc_num', 'anon_dicom_path']]
    meta_site_dicom = meta_site_dicom[meta_site_dicom['anon_dicom_path'].isin(series_instance_uids)]
    meta_site_dicom.drop_duplicates(subset=['anon_dicom_path'], inplace=True)
    #######################################################################################################
    #######################################################################################################

    client_list = config.get('client_list')
    
    for i, site in enumerate(client_list):
        client_model_path = config.get('workdir') + f'/EMBED_net_client_{site}.pth'
        global_model_path = config.get('workdir') + f'/EMBED_net_global.pth'
        # client_cases = sites[i]
        client_cases = meta_site_dicom[meta_site_dicom['loc_num'] == site]['anon_dicom_path'].tolist()
        client_cases = ','.join(client_cases)
        script_args = f"--train_dataset_path {config.get('train_dataset_path')} \
                        --test_dataset_path {config.get('test_dataset_path')} \
                        --batch_size {config.get('batch_size')} \
                        --learning_rate {config.get('learning_rate')} \
                        --client_model_path {client_model_path} \
                        --global_model_path {global_model_path} \
                        --client_config_path {config.get('client_config_path')} \
                        --client_cases {client_cases} \
                        --workdir {Path(config.get('workdir'))}" 
        executor = ScriptRunner(script=train_script, script_args=script_args)
        job.to(executor, f"site-{site}")

    # # threads=1 or 0 to avoid deadlocks from GPUs and to be able for matplotlib to work properly
    job.simulator_run(workspace=config.get('workdir'), threads=0) 
