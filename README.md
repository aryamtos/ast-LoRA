
# Audio Spectrogram Transformer with LoRA

-[Introduction](#Introduction) 

-[Getting Starting](#Getting-Starting)

-[Citation](#Citation)

-[Contact](#Contact)


<h2>Introduction</h2> 


This repository hosts an implementation of the Audio Spectrogram Transformer (AST) based on the official implementation provided by the authors of the paper "Parameter-Efficient Transfer Learning of Audio Spectrogram Transformers" -- <a href="https://github.com/umbertocappellazzo/PETL_AST">PETL_AST</a>. The paper introduces Parameter-Efficient Transfer Learning (PETL) methods tailored to the Audio Spectrogram Transformer architecture. In this repository, particular emphasis is placed on the LoRA (Low Rank) adapter, which has demonstrated the best results.

<h3>After all, what is LoRA?</h3>

Low Rank Adaption is the most widely used PETL methods. Provides the approach to fine-tuning a model for downstream tasks and the capability of transfer learning with fewer resources.
<h4>Context</h4>
     -- Neural networks use  dense layers with weight matrices for computation
     --These weight matrices are typically  "full-rank"(use all dimensions)
<h4>Solution</h4>
     -- The pre-trained models have a low "intrisic dimension" meaning they might not need full-rank weight matrices.<br/>
     -- The pre-trained parameters of the original model (<strong>W</strong>) are frozen. These weights (<strong>W</strong>) will  not be modified</br>
     -- A new set of parameters is added to the network <strong>WA</strong> and <strong>WB</strong> ( low-rank weight vectors) where the  dimensions are  represented as      
             <strong>dxr</strong> and <strong>rxd</strong> ( r - low rank dimension; d - original dimension) <br/>


<p align="center">      
<img width="300" id="image" alt="Module AST" src="https://github.com/aryamtos/ast-brazilian-LoRA/assets/46492977/ae4ab8ad-0224-4af3-ba2b-0065d4fee381">
</p>



<h2>Getting Starting</h2>

Step 1. Clone or download this repository and set it as the working directory, create a virtual environment and install the dependencies.

```
cd ast-lora/ 
python3 -m venv venvast
source venvast/bin/activate
pip install -r requirements.txt 
```

Step 2: Running Experiment

We just need set some parameters in train.yaml:

<ul>
  <li>``` lr_LoRA ```  </li>
  <li>``` weight_decay ```</li>
  <li>``` final_output ```</li>
  <li>``` patch_size ```</li>
  <li>``` hidden_size ```</li>
</ul>

In main.sh just need set data path ( train, validation and test ) and some parameters. 

```
bash  main.sh

```


## Citation  

Citing the original paper:

```  
@misc{cappellazzo2024efficient,
      title={Efficient Fine-tuning of Audio Spectrogram Transformers via Soft Mixture of Adapters}, 
      author={Umberto Cappellazzo and Daniele Falavigna and Alessio Brutti},
      year={2024},
      eprint={2402.00828},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}

```

## Contact

If you have a question, or just want to share how you have use this, send me an email at ariadnenascime6@gmail.com

