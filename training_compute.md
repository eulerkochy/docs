# Training compute required for SOTA models. 

This doc aims to outline how much compute (in dollars) was required to train certain models. The goal is to build an intuition over training costs for a variety of models, and priortise accordingly. This should also help us aggresively priortize efficiency, and get an intuition on how much efficiency we can squeeze out of the limited compute that we may have. 



## Rule of Thumbs
### An LLM requires 12*N*T FLOP compute to train
where N is the total number of parameters, and T is the number of tokens. A way to see this is that : 
* The majority of operations in LLMs are matrix multiplication
* Each weight contributes to two flops - multiple and accumulate (MAC) in forward pass.
* Backward pass on the other hand requires 4 FLOPs for gradient computation. 
* Each weight interacts with all tokens. Thus 6FLOPS * N * T tokens. 
* Furthermore, we are only able to utilize 40-60% of hardware peak FLOPs in distibuted setting, thereby contributing another 2x factor. 

Going by this, training a 1B model with 1T parameter would take 12 * 10^9 * 10^12 FLOPs/ (10^15 * 0.86 * 10^5 seconds/day) = 140 PFLOPS days.

### Sparse weights are 2x efficient. 
Sparse weights provide 2x FLOPs efficiency in A100, and H100 architectures due to lesser power consumption (?).

### FLOPs for different precision. 
Different precision offer different FLOPs given that they are supported by the hardware. The ratio of 1/FLOPs compared to FP8 is roughly. 
| Precision  | 1/FLOPs |
| ---------- | ------- |
| FP8, INT8  | 1       |
| FP16, BF16 | 2       |
| TF32       | 4       |
| FP32, TF64 | 32      |
| FP64       | 64      |

This means that FP8 roughly has 32 more flops than fp32, provided they're supported by hardware. For example, A100 doesn't support FP8 natively but supports INT8. 

This also means that for lower bit quantizations like INT2/INT4 you can't further improve on computation side, because the final computation anyway happens in the lowest supported precision - INT8. These quantizations still speed up model inference though, since most models are memory bound, and INT2/INT4 quantization can reduce the memory that needs to be transferred from HBM for computation, thereby removing the memory bandwidth bottleneck. 

### GPU hours to PFLOPs days. 
* 1 PFLOP day = 1 H100 day (FP16)
* 1 PFLOP day = 3.3 A100 days (FP16)


## FLOPS required for some SOTA models.

| Model                       | Category            | Model Size                 | DataSize           | FLOPS                        | Flops source                                                                        |
| --------------------------- | ------------------- | -------------------------- | ------------------ | ---------------------------- | ----------------------------------------------------------------------------------- |
| SigLIP-B                    | Vision/Text Encoder | ~100M(Vision), ~100M(Text) | ~3B                | 1.5                          | [Siglip Paper](https://arxiv.org/abs/2303.15343)                                    |
| PaliGemma                   | VLLM                | Gemma-2B + siglip-400M     | 450B tokens        | 190 PFLOPs Days [1]          | [PaliGemma Paper](https://arxiv.org/abs/2407.07726)                                 |
| LLama2                      | LLM                 | 7B/13B/34B/70B             | 1.8T tokens        | 4.6/13.5/21.5 EFLOPS days    | [LLama2 paper](https://arxiv.org/abs/2307.09288)                                    |
| LLama3                      | LLM                 | 8B/70B/405B                | 15T tokens         | 16.7/146/845 EFLOPs days [2] | [LLama3 paper](https://arxiv.org/pdf/2407.21783)                                    |
| Gemma                       | LLM                 | 2B/7B                      | 3T/7T              | 0.8/7 EFLOPs days  [2]       | [Gemma Paper](https://arxiv.org/abs/2403.08295)                                     |
| Gemma2                      | LLM                 | 27B/9B/2B                  | 13T/8T/2T          | 49/10/0.5 EFLOPs days [2]    | [Gemma2 Paper](https://arxiv.org/abs/2408.00118)                                    |
| DeepseekV3                  | LLM                 | 671B, 37B activated        | 14.8 T             | 229  EFLOPS days             | [Tech Report](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf) |
| StableDiffusion-2(MosaicML) | Image Generation    | 0.86B                      | 1.1T image/caption | 308PFLOPs                    | [MosaicML Blog](https://www.databricks.com/blog/diffusion) [3]                      |


TODO(pshishodia): Add numbers for text embedding models. Roughly we can train a good enough text embedding model with 100M parameters and 100B tokens in 1.4 PFLOPs days.

---------


[1] PaliGemma paper also mentioned that it was trained in 32 bit precision, and that they kept the initial pretraining to 1B examples, even though 10x, 30x lower examples were good enough. This means that a good VLM can be trained for much cheaper ~20 PFLOPs days.

[2] Actual FLOPs numbers aren't provided. We estimate the FLOPs using the fact that 1B LLM trained on 1T tokens takes 140 PFLOPs days, and that total compute is proportional to model size & compute. See Rule of thumbs for reference. 

[3] MosaicML also shares the strategis that helped them improve the efficiency. 

## Appendix
### FLOPS Calculation
1. Deepseek V3 : 2.788 M H800 hours x 1.979 FP8 PFLOPS / 24 hours = 229 EFLOPs days
2. SigLIP (B/16, 72.1% acc) : 64 TPUv4 days x 0.275 PFLOPS = 17 PFLOPs days. Note that this is different from the best SigLIP variant so-400m which uses both vision & text encoders of ~400M parameters.
3. PaliGemma (Both Siglip & Gemma are trained.) : 256 TPUv5e * 3.75 days * 0.197 PFLOPs = 190 PFLOPS days. 
4. LLAMA2 7B (Assuming FP16 training): 184320 A100 hours * 0.3 PFLOPS / 24 = 2.3 Similarly for 13B/34B/70B, it took 4.6/13.5/21.5 EFLOPS days, same for other LLAMA variants as well. 
5. Stable Diffusion 2 (MosaicML) : 23835 A100 hours x 0.312 PFLOPs / (24 hours/day) = 309 PFLOPs days.

