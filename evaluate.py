import os
import numpy as np
import torch
from fairseq.models.roberta import RobertaModel
from fairseq.models.roberta import quantize

from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.transformer_layer import TransformerEncoderLayerBase
from scipy.stats import pearsonr
import pandas as pd

databins = {'CoLA': 'CoLA-bin', 'MNLI': 'MNLI-bin', 'MRPC': 'MRPC-bin', 'QNLI': 'QNLI-bin', 'QQP': 'QQP-bin', 'RTE': 'RTE-bin', 'SST-2': 'SST-2-bin', 'STS-B': 'STS-B-bin', 'WNLI': 'WNLI-bin'}
metrics = {}
cfg = "/home/mshalimay/finetuning/mnli.yaml"

def load_model(dataset, inbits=8, epochs=10, case=None):
    checkpoint_path = f"/home/mshalimay/checkpoints_experiment1/{dataset}/bits_{inbits}_epoch_{epochs}"
    checkpoint = torch.load(os.path.join(checkpoint_path, 'checkpoint_best.pt'))
    print(f"Best {checkpoint['extra_state']['best']}")
    model = RobertaModel.from_pretrained(
        checkpoint_path,
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path=databins[dataset]
    )

    if case is None:
        case = "base"
    if dataset not in metrics[case]:
        metrics[case][dataset] = {}
    metrics[case][dataset]["best"] = checkpoint['extra_state']['best']
    return model

def evaluate_sts_b(model, dataset):
    if dataset == 'STS-B':
        predictions, actual_scores = [], []
        with open(f'glue_data/STS-B/dev.tsv', 'r', encoding='utf-8') as fin:
            fin.readline()  # Skip the header
            for line in fin:
                tokens = line.strip().split('\t')
                sent1, sent2, actual_score = parse_line(tokens, dataset)
                tokens = model.encode(sent1, sent2)
                prediction = model.predict('sentence_classification_head', tokens).squeeze().item()
                predictions.append(prediction)
                actual_scores.append(actual_score)

        pearson_corr, _ = pearsonr(predictions, actual_scores)
        print(f'| STS-B Pearson correlation: {pearson_corr:.4f}')


def evaluate_model(model, dataset, label_fn, case=None, stop=-1):
    ncorrect, nsamples = 0, 0
    model.cuda()
    model.eval()

    file_path = f'glue_data/{dataset}/dev.tsv' if dataset != 'MNLI' else f'glue_data/MNLI/dev_matched.tsv'

    if dataset == 'STS-B':
        evaluate_sts_b(model, dataset)
        return
    
    with open(file_path, 'r') as fin:
        fin.readline()  # Skip the header
        with torch.no_grad():
            for line in fin:
                tokens = line.strip().split('\t')
                sent1, sent2, target = parse_line(tokens, dataset)
                if sent2:
                    tokens = model.encode(sent1, sent2)
                else:
                    tokens = model.encode(sent1)
                prediction = model.predict('sentence_classification_head', tokens).argmax().item()
                prediction_label = label_fn(prediction)
                ncorrect += int(prediction_label.strip() == target.strip())
                nsamples += 1

                # print progress
                if nsamples % 100 == 0:
                    print(f'Processed {nsamples} samples')
                    # current accuracy
                    print(f'| Current Accuracy: {ncorrect/nsamples:.4f}')

                if nsamples == stop:
                    break
                
    accuracy = float(ncorrect) / float(nsamples)
    print(f'| {dataset} Accuracy: {accuracy:.4f}')

    if case is None:
        case = "base"
    metrics[case][dataset]["accuracy"] = accuracy
    
    return accuracy

def parse_line(tokens, dataset):
    if dataset == 'MNLI':
        # Adjusted for MNLI column indices, if necessary
        sent1, sent2, target = tokens[8], tokens[9], tokens[-1]
        return sent1, sent2, target

    elif dataset == 'QQP':
        sent1, sent2, target = tokens[3], tokens[4], tokens[5]  # Adjust for QQP column indices
        return sent1, sent2, target
    
    elif dataset == 'MRPC':
        sent1, sent2, target = tokens[3], tokens[4], tokens[0]
        return sent1, sent2, target
    elif dataset == 'QNLI' or dataset == 'RTE' or dataset == 'WNLI':
        sent1, sent2, target = tokens[1], tokens[2], tokens[-1]
        return sent1, sent2, target
    elif dataset == 'SST-2':
        sent1, target = tokens[0], tokens[1]
        return sent1, None, target
    elif dataset == 'CoLA':
        sent1, target = tokens[3], tokens[1]
        return sent1, None, target
    elif dataset == 'STS-B':
        sent1, sent2, target = tokens[7], tokens[8], float(tokens[-1])  
        return sent1, sent2, target
    else:
        raise ValueError(f"Dataset {dataset} not supported.")

def main():
    datasets = ['RTE', 'CoLA', 'MRPC', 'SST-2', 'QNLI', 'QQP', 'MNLI']
    epochs = 10
    cases = [4, 8]
    stop = -1

    for case in cases:
        metrics[case] = {}
        for dataset in datasets:
            try:
                model = load_model(dataset, case, epochs, case=case)

                quantize.set_measure(model, False, False, None)

                # quantize attention only
                # quantize.set_measure(model, True, True, mods_measure=TransformerEncoderLayerBase.__name__) 

                # quantize fc only
                quantize.set_measure(model, True, True, MultiheadAttention.__name__)
                
                label_fn = lambda label: model.task.label_dictionary.string([label + model.task.label_dictionary.nspecial])
                evaluate_model(model, dataset, label_fn, case=case, stop=stop)
            except Exception as e:
                print(f"Error: {e}")
                metrics[case][dataset] = {}
                metrics[case][dataset]["best"] = np.nan
                metrics[case][dataset]["accuracy"] = np.nan

            save_metrics(metrics, case)
    
def save_metrics(metrics, annotate):
    filename = f"metrics_{annotate}.xlsx"

    rows = []
    for case, datasets in metrics.items():
        for dataset, data in datasets.items():
            row = {
                'Case': case,
                'Dataset': dataset,
                'Best': data['best'],
                'Accuracy': data.get('accuracy', 'N/A')  
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_excel(filename, index=False,float_format="%.6f")

if __name__ == '__main__':
    main()
