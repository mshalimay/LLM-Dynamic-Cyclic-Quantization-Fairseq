# Examples
Below two examples of experiments possible with this codebase:
- Finetuning of `Roberta` for GLUE tasks, post-training dynamic quantization and evaluation. Interesting questions to answer:
   - How much we loose by quantizing? (it depends)
   - Which layers are more sensible to quantization, attention layers? Feed forward layers?

- Finetuning of `Roberta` for GLUE tasks and training-aware cyclic quantization. Interesting questions to answer:
  - How to quantize? Fixed or varying number of bits?
  - How much we loose with training-aware quantization? (not much, in fact...)
  - Is it possible to perform better than full precision counterpart with quantization? (yes!)

# General notes on experiments
- Many hyperparameters choices follows `fairseq` [GLUE Examples](https://github.com/facebookresearch/fairseq/tree/main/examples/roberta).

# Usage
- Setup the Fairseq environment following the [official guideline](https://github.com/facebookresearch/fairseq/tree/main?tab=readme-ov-file#requirements-and-installation).
- Substitute / modify original codebase with the files provided in this repo.
- To download GLUE data, use `download_glue.py`
- To process the downloaded data, use `preproc.bash`
- To evaluate the models, use `evaluate.py` 
- To replicate experiment1, run `experiment1.bash`
- To replicate experiment2, run `experiment2.bash`
- **Important:** Please adjust the paths in the scripts accordingly. Try to use absolute paths because `fairseq` does not work well with relative paths

# Experiment 1: Post-training of fine-tuned Roberta
- In this experiment, Roberta base model is finetuned for GLUE tasks as follows:
  - weights for pre-trained Roberta are frozen. 
  - weights for classification head are tuned.
  - during fine-tuning/training, weights and activations zero-points and ranges (`qparams`) are computed for post-training quantization using an exponential moving average strategy. 

- After finetuning, the model is quantized to 4 and 8 bits using the recorded `qparams` and the model is evaluated on the GLUE tasks, with three variations:
    (i) quantizing all `Roberta` layers; 
    (ii) quantizing only the `MultihHeadAttention` layers; 
    (iii) quantizing only the `Feed-forward` (FC) layers. 

- This answer two questions: 
    - How much of accuracy drop due to quantization? 
    - What modules are more sensible to quantization in terms of accuracy drop?

The table below shows the results. The column `full` shows the accuracies on the same tasks without quantization (but with frozen weights). 

**Comments**
- In all cases, accuracies are comparable to the model without quantization. 
	- For instance, with all layers quantized, the drop in average accuracy is of 1.1pp for 4 bit precision - mainly influenced by the drop in SST-2 and MPRC tasks.
	- For 8-bit precision, average precision is only 0.1pp lower than the full precision counterpart, with the larger drop in SST-2 task (0.5pp)
- The model showed more sensitivity to quantization of the feed forward (`FC`) layers. 
	- This can be seen by the difference in average accuracy from the middle and rightmost tables to the leftmost table, reported separately as *Delta Acc*.
	- For the 4-bit case, by **not** quantizing:
		- the `FC` layers (middle), we can recover approximately 0.54pp in accuracy; 
		- the `MultiHeadAttention` layers (right), we can recover 0.37 in accuracy
	- For the 8-bit case, the delta is pretty close to zero. That aligns with the fact that the fully quantized have virtually the same performance to the full precision

![Alt text](<Pasted image 20240305223308.png>)

# Experiment 2: Training-aware cyclic quantization
- In this experiment, Roberta base model is finetuned for GLUE tasks as follows:
  - All weights are tuned (i.e., no frozen weights as in experiment 1)
  - During fine-tuning/training, weights and activations zero-points and ranges (`qparams`) are computed using an exponential moving average strategy. 
  - Activations are quantized to 16 bits precision.
  - For weights, precision alternatives:
    1) varies cyclically 4 to 8 bit
    2) varies cyclically from 4 to 16 bit
    3) varies cyclically from 8 to 16 bit
    4) fixed at 8 bits
    5) fixed at 16 bits

  
The table below shows the accuracies for selected GLUE tasks in the 5 variations above, alongside the accuracies without quantization. The colormap highlights the higher accuracies (more green => higher)

**Note**: changes relative to `fairseq` configs: max epochs = 5; fp16 disabled.

**Comments**
- the better average accuracy was obtained with the 2nd (cyclic 4-16) setting, surpassing even the average accuracy without quantization. 
- There is a mild variation in this result. For instance, for **CoLa**, case 3 and 5 is better, albeit by a little margin.
- We observe a better accuracy for the cyclic case using broader ranges. This potentially illustrates [[1]](#references) observation that cyclic precision allows for a better search in optimization space.
- However, it is worth mentioning the average gains relative to not cyclic are limited.  I believe this might be due to the usage of already finetuned configurations provided by `fairseq`.

![Alt text](<Pasted image 20240305192135.png>)

# Limitations/Issues
- Due to computing resource limitations, for _Experiment 2_ I limited the number of training epochs to 5 and the number of datasets.

- In `fairseq` base run, the forward method of `MultiHeadAttention` **do not use the forward method of the linear layers within it**. It only pass the inputs and linear layer weights to `pytorch` base functional. If one uses `hydra-train`, this means that `q,k,v` outputs are not quantized, even though specified. For quantization of these, use fairseq's `_xformers_attn_forward` module instead.


# References
[1] Fu, Y., Guo, H., Li, M., Yang, X., Ding, Y., Chandra, V., & Lin, Y. (2021). Cpt: Efficient deep neural network training via cyclic precision. arXiv preprint arXiv:2101.09868.

[2] Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019). Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.

https://github.com/facebookresearch/fairseq