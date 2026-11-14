import argparse
import yaml

from collect_component import SEGCOLLECTOR

def run_train(args):
    train_config = args.train_config
    with open(train_config, 'r') as f:
        config = yaml.safe_load(f)['CONFIG']
    
    for models in ['wcenet','fbampsm']:
        config['MODEL'] = models

        trainer, epochs, eval_epoch = SEGCOLLECTOR()(config)
        print(f'train {config["MODEL"]} on {config["DATASET"]} for {epochs} epochs, 'f'eval every {eval_epoch} epoch(s), save to {trainer.save_dir}')
        return trainer.train(epochs, eval_epoch)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Admin tool for model training and evaluation')
    parser.add_argument('--train_config', type=str, required=True, help='training config')

    run_train(parser.parse_args())
