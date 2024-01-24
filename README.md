# Llama
<h1>Language Models</h1>

This repository is the implementation of several versions of language models:
<ol>
<li>Basic language model (bigram) that is based on a single embedding neural layer that is trained on Shakespeare's data. The model just finds the probability distribution of the occurrence of each symbol in the vocabulary after each of all the other symbols. Based on this probability and by knowing the current letter we can sample from the probability distribution we have for the current letter followed by each of the other letters. It is a basic way to create a language model that predicts the next letter based on the current letter.</li>

<li>
Nano GPT language model. Following the paper "Attention is all you need" and openAI's GPT-2 and GPT-3. We developed a decoder-only transformer that can generate text. The architecture of the decoder is similar to that presented in the "Attention is all you need" article. The model is also trained on Shakespeare's data. 
</li>
<li> Llama that is presented in the  
    <a href="https://arxiv.org/pdf/2302.13971.pdf" target="_blank">Llama</a> article published by Meta AI in Feb 2023. LLaMA (Large Language Model Meta AI) is a family of large language models (LLMs) ranging from 7B to 65B parameters. Like other LLMs, Llama is transformer-based. As the authors mentioned in their paper, they want to show that it is possible to train state-of-the-art models using publicly available datasets exclusively, without resorting to proprietary and inaccessible datasets. They demonstrate that LLaMA-13B outperforms GPT-3(175B) on most benchmarks and LLaMA-65B is competitive with the best models, Chinchilla-70B and PaLM-540B.

</li>
</ol>
