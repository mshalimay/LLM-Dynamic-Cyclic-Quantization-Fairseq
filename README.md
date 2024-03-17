# LLM Dynamic and Cyclic Quantization with Fairseq

This repo customizes [fairseq](https://github.com/facebookresearch/fairseq) selected modules to allow for dynamic and cyclic quantization of NN models. This repo focus on the `Roberta` model, but codes can be extended to other models. 


Below two examples of experiments possible with this codebase (more details provided in `Examples.md`):
- Finetuning of `Roberta` for GLUE tasks, post-training dynamic quantization and evaluation. Interesting questions to answer:
   - How much we lose with post-training quantization? (it depends)
   - Which layers are more sensible, attention layers? Feed forward layers?

- Finetuning of `Roberta` for GLUE tasks and training-aware cyclic quantization. Interesting questions to answer:
  - How to quantize? Fixed or varying number of bits?
  - How much we loose with training-aware quantization? (not much, in fact...)
  - Is it possible to perform better than full precision counterpart with quantization? (yes!)

For sake of space, this repo only includes the modified files from `fairseq` codebase. These files sould be added to the original codebase for experimentation.


# Codebase description
The following files were modified/included in `fairseq` for quantization:

- `fairseq/fairseq/models/roberta/quantize.py`: added `nn modules` and functions for quantization.

- `fairseq/fairseq/models/roberta/model.py`. Modified to include:
	- Additional arguments:
        - `--freeze`: 
        - `--quantw`: if True, all weights are quantized.
        - `--quantin`: if True, all activations are quantized.
        - `--bitsin`: number of bits for quantization of activations.
        - `--bitsw`: number of bits for quantization of weights.
        - `--measurein`: operates in measurement mode for activations
        - `--measurew`: operates in measurement mode for weights

	- Functions to freeze weights and wrap layers with quantized versions

- `fairseq/fairseq/cfg_cpt.json`: parametrization for dynamic/cyclic quantization

- `fairseq/fairseq/trainer.py`:
	- modified to include cyclic precision scheduler for quantization

- `fairseq/fairseq_cli/train.py`: 
    - modified to compute total number of iterations and compute parameters for cyclic precision scheduler

- `download_glue.py`: donwload GLUE data for finetuning

- `evaluate.py`: Performs infernce on GLUE tasks given a finetuned model 


- `experiment1.bash`: 
    - replicate results from experiment 1.


- `experiment3.bash`:
    - replicate results from experiment 2.

# Quantization implementation
Dynamic Quantization is implemented as follows:
- Each layer to be quantized is wrapped with a `QuantizedLayer` `nn.module` (see `quantize.py`).
- This wrapper alters the forward method of the original layer to potentially use quantized inputs, weights and quantized outputs.
- The wrapper encapsulates a `QuantMeasure` layer, which is responsible for calculating and storing range and zero point parameters for dynamic quantization.
- Two options for range and zero point (qparams) calculation:
	- 1) Compute averages during training and use these averages for quantization during training (`dynamic_qparams` = True)
	- 2)  Compute qparams based on current input/weight only (`dynamic_qparams`=True)

# CPT implementation
- For cyclic-precision quantization, I follow [[1]](#references), and use a `cosine` scheduler for cycling between number of bits in quantization.

CPT is realized via:
- Inclusion of the class `CyclicPrecisionScheduler` in `trainer.py`. This class stores the cyclic state and updates the precision.
- The `total number of iterations` (necessary to calculate the number of cycles) is calculated on the fly in the `train.,py` script, based on the number of epochs and size of the batches used during training
- A`cfg_cpt.json` configuration file with CPT configurations must be provided with the following parameters:
    - `wbit_min`: the minimum number of bits allowed for weight quantization
    - `wbit_max`: the maximum number of bits allowed for weight quantization
    - `inbit_min`: the minimum number of bits allowed for activations quantization
    - `inbit_max`: the maximum number of bits allowed for activations quantization
    - `num_cyclic_period`: The total number of cycles (from min to max bit) for the whole run    
    - See `fairseq/fairseq/cfg_cpt.json` and `experiment2.bash` for examples

# Sample architechtures after quantization

Below is the architechture of `fairseq`'s `Roberta` model

### Base Roberta
```
(encoder): RobertaEncoder(
    (sentence_encoder): TransformerEncoder(
      (dropout_module): FairseqDropout()
      (embed_tokens): Embedding(50265, 768, padding_idx=1)
      (embed_positions): LearnedPositionalEmbedding(514, 768, padding_idx=1)
      (layernorm_embedding): FusedLayerNorm(torch.Size([768]), eps=1e-05, elementwise_affine=True)
      (layers): ModuleList(
        (0-11): 12 x TransformerEncoderLayerBase(
          (self_attn): MultiheadAttention(
            (dropout_module): FairseqDropout()
            (k_proj): Linear(in_features=768, out_features=768, bias=True)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (self_attn_layer_norm): FusedLayerNorm(torch.Size([768]), eps=1e-05, elementwise_affine=True)
          (dropout_module): FairseqDropout()
          (activation_dropout_module): FairseqDropout()
          (fc1): Linear(in_features=768, out_features=3072, bias=True)
          (fc2): Linear(in_features=3072, out_features=768, bias=True)
          (final_layer_norm): FusedLayerNorm(torch.Size([768]), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    (lm_head): RobertaLMHead(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (layer_norm): FusedLayerNorm(torch.Size([768]), eps=1e-05, elementwise_affine=True)
    )
  )
  (classification_heads): ModuleDict(
    (sentence_classification_head): RobertaClassificationHead(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (dropout): Dropout(p=0.0, inplace=False)
      (out_proj): Linear(in_features=768, out_features=2, bias=True)
    )
  )
)
```
Below is the NN structure of the quantized version of **Roberta Model** with quantization in all layers.
Obs: although both `MultiHeadAttention`  and its linear layers are quantized, there is no duplicate quantization. See *Quantization Implementation* for details.

### Quantized Roberta
```
RobertaHubInterface(
  (model): RobertaModel(
    (encoder): RobertaEncoder(
      (sentence_encoder): TransformerEncoder(
        (dropout_module): FairseqDropout()
        (embed_tokens): Embedding(50265, 768, padding_idx=1)
        (embed_positions): LearnedPositionalEmbedding(514, 768, padding_idx=1)
        (layernorm_embedding): FusedLayerNorm(torch.Size([768]), eps=1e-05, elementwise_affine=True)
        (layers): ModuleList(
          (0-11): 12 x TransformerEncoderLayerBase(
            (self_attn): QuantizedLayer(
              (layer): MultiheadAttention(
                (dropout_module): FairseqDropout()
                (k_proj): QuantizedLayer(
                  (layer): Linear(in_features=768, out_features=768, bias=True)
                  (quant_input_0): QuantMeasure()
                  (quant_weight_0): QuantMeasure()
                  (quant_out): QuantMeasure()
                )
                (v_proj): QuantizedLayer(
                  (layer): Linear(in_features=768, out_features=768, bias=True)
                  (quant_input_0): QuantMeasure()
                  (quant_weight_0): QuantMeasure()
                  (quant_out): QuantMeasure()
                )
                (q_proj): QuantizedLayer(
                  (layer): Linear(in_features=768, out_features=768, bias=True)
                  (quant_input_0): QuantMeasure()
                  (quant_weight_0): QuantMeasure()
                  (quant_out): QuantMeasure()
                )
                (out_proj): QuantizedLayer(
                  (layer): Linear(in_features=768, out_features=768, bias=True)
                  (quant_input_0): QuantMeasure()
                  (quant_weight_0): QuantMeasure()
                  (quant_out): QuantMeasure()
                )
              )
              (quant_input_0): QuantMeasure()
              (quant_input_1): QuantMeasure()
              (quant_input_2): QuantMeasure()
              (quant_weight_0): QuantMeasure()
              (quant_weight_1): QuantMeasure()
              (quant_weight_2): QuantMeasure()
              (quant_weight_3): QuantMeasure()
            )
            (self_attn_layer_norm): FusedLayerNorm(torch.Size([768]), eps=1e-05, elementwise_affine=True)
            (dropout_module): FairseqDropout()
            (activation_dropout_module): FairseqDropout()
            (fc1): QuantizedLayer(
              (layer): Linear(in_features=768, out_features=3072, bias=True)
              (quant_input_0): QuantMeasure()
              (quant_weight_0): QuantMeasure()
            )
            (fc2): QuantizedLayer(
              (layer): Linear(in_features=3072, out_features=768, bias=True)
              (quant_input_0): QuantMeasure()
              (quant_weight_0): QuantMeasure()
            )
            (final_layer_norm): FusedLayerNorm(torch.Size([768]), eps=1e-05, elementwise_affine=True)
          )
        )
      )
      (lm_head): RobertaLMHead(
        (dense): Linear(in_features=768, out_features=768, bias=True)
        (layer_norm): FusedLayerNorm(torch.Size([768]), eps=1e-05, elementwise_affine=True)
      )
    )
    (classification_heads): ModuleDict(
      (sentence_classification_head): RobertaClassificationHead(
        (dense): Linear(in_features=768, out_features=768, bias=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (out_proj): Linear(in_features=768, out_features=2, bias=True)
      )
    )
  )
)
```

# Usage
- Setup the Fairseq environment following the [official guideline](https://github.com/facebookresearch/fairseq/tree/main?tab=readme-ov-file#requirements-and-installation).
- Substitute / modify original codebase with the files provided in this repo.
- To download GLUE data, use `download_glue.py`
- To process the downloaded data, use `preproc.bash`
- To evaluate the models, use `evaluate.py` 
- To replicate experiment1, run `experiment1.bash`
- To replicate experiment2, run `experiment2.bash`
- **Important:** Please adjust the paths in the scripts accordingly. Try to use absolute paths because `fairseq` does not work well with relative paths


# References
[1] Fu, Y., Guo, H., Li, M., Yang, X., Ding, Y., Chandra, V., & Lin, Y. (2021). Cpt: Efficient deep neural network training via cyclic precision. arXiv preprint arXiv:2101.09868.

[2] Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019). Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.

https://github.com/facebookresearch/fairseq