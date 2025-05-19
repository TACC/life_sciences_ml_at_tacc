LLMs in Biology Applications
============================

As we saw in the previous section, Large Language Models (LLMs) have demonstrated remarkable
capabilities in summarizing, processing, and generating human-like text. Their success hinges on the
assumption that natural language can be modeled as a system of sequential words/sounds (tokens), and
that the precise arrangement of those tokens is what gives rise to context and meaning.

If we look elsewhere in nature, we find it is rich with sequential information and processes - DNA
sequences, transcriptomes, protein sequences, signal transduction cascades, metabolic pathways, and
even small molecules (interpreted as SMILES strings). Do these sequences in nature have their own
"vocabulary" conferring some biological importance? **If we treat these sequences as language, can
we uncover hidden signals or patterns with LLMs?**

In this section, we will look at how LLMs are currently being used in a few different areas in the
life sciences. The hands-on examples are driven by current topics found in the literature. By the
end of this section, you should be able to:

* Pull existing models from Hugging Face using the ``transformers`` library
* Apply DNABERT-2 to a DNA sequence classification task
* Generate new protein sequences using ProtGPT2

.. note:: 

   The following examples should be done in a Jupyter Notebook on Vista with the ``Day4-pt-251``
   kernel. If you are doing this elsewhere, you need to first install the ``transformers`` library:

   .. code-block:: console

       $ pip install --user transformers


Example 1: DNA Sequence Context
-------------------------------

**DNABERT-2** is an advanced transformer-based foundation model designed for analyzing DNA sequences
across multiple species. It builds upon the original DNABERT model by introducing significant
improvements in tokenization and architecture, and it is trained on genomes from 135 different
species. This allows DNABERT-2 to perform well on a wide range of genomic tasks, including promoter
prediction, gene expression analysis, DNA methylation detection, and more.


How it Works
^^^^^^^^^^^^

DNABERT and DNABERT-2 are transformer-based architectures modeled after Google's BERT architecture.
BERT (Bidirectional Encoder Representations from Transformers) deviates from the traditional 
transformer architecture in that it only uses the encoder portion of the original model. Traditional
transformers include both an encoder and decoder for sequence-to-sequence tasks like translation,
but BERT-like models are designed solely for understanding language, not generating it.

The original DNABERT tokenized k-mer sequences with fixed length as input, and added a CLS token (a
tag representing meaning of entire sentence), a SEP token (sentence separator) and MASK tokens (to
represent masked k-mers in pre-training). The input is embedded and fed through 12 transformer
blocks. The outputs can be interepreted for sentence-level classification or token-level
classification.

.. figure:: ./images/dnabert_model.png
   :width: 800
   :align: center

   The origina DNABERT model [1]_ (shown above) was improved in DNABERT-2 [2]_


Some of the key innovations of DNABERT-2 over the original DNABERT include:

1. **Byte Pair Encoding (BPE) Tokenization:** Instead of fixed-length k-mer tokenization, which can
   be computationally inefficient and may not capture variable-length patterns effectively.
   DNABERT-2 replaces this with BPE, a data-driven compression algorithm that identifies and merges
   frequently occurring nucleotide sequences. This approach reduces sequence length by approximately
   five times on average, enhancing computational efficiency and preserving meaningful biological
   patterns.

2. **Enhanced Model Architecture:** DNABERT-2 improves upon the original model by replacing the
   original embeddigns with Attention with Linear Biases (ALiBi) embeddings, which overcome
   limitations with input length and allow the model to handle inputs of unlimited length without
   performance issues. Also, DNABERT-2 incorporates Flash Attention, an optimized mechanism for 
   improved IO to accelerate training and inference and reducing memory requirements.

3. **Genome Understanding Evaluation (GUE) Benchmark:** The original DNABERT was trained on human
   sequences alone. DNABERT-2 is trained on genomces from 135 different species, enhancing its
   generalizability across diverse organisms. To assess and compare genome models effectively, the
   developers introduced the GUE benchmark, a massive amalgam of 36 datasets for 9 different genomic
   tasks.


DNABERT-2 is trained and evaluated on massive amounts of data:

* Trained on genomes from **135 species** comprising **>32B bases**
* Evaluated with the GUE benchmark for **9 different genome classification tasks**
* In total, there were **262B training tokens** and **117M parameters** in the model

With these improvements, DNABERT-2 was able to outperform its predecessor on 23 out of 28 datasets 
from the GUE benchmark in tasks like promoter prediction, gene expression analysis, DNA methylation
detection, chromatin state classification, and variant effect prediction.


Try it Out
^^^^^^^^^^

Let's perform a simple classification task using DNABERT-2. 

.. tip::

   You may also need to install this library from within your Jupyter Notebook:

   .. code-block:: python

      >>> pip install --user einops


First, use the ``transformers.pipeline`` method to load the 117M parameter model from
`Hugging Face <https://huggingface.co/zhihan1996/DNABERT-2-117M>`__:

.. code-block:: python

   >>> import torch
   >>> from transformers import AutoTokenizer, AutoModel

   >>> tokenizer = AutoTokenizer.from_pretrained('zhihan1996/DNABERT-2-117M', trust_remote_code=True)
   >>> model = AutoModel.from_pretrained('zhihan1996/DNABERT-2-117M', trust_remote_code=True)


Then, tokenize the DNA sequence with the efficient BPE method used by the model. This fragments the
sequence into variable length k-mers and assigns each k-mer a unique token ID (embedding):

.. code-block:: python

   >>> dna = 'ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC'  # Example sequence
   >>> inputs = tokenizer(dna, return_tensors = 'pt')['input_ids']
   >>> print(inputs)

.. code-block:: text

   tensor([[  2, 101, 1543, 352, 583, ..., 103,   3]])


Run the model to get the hidden states. The model will return a tensor as output.

.. code-block:: python

   >>> hidden_states = model(inputs)[0] # [1, sequence_length, 768]
   >>> print(hidden_states.shape)
   >>> print(hidden_states[0])


.. code-block:: text

   torch.Size([1, 17, 768])
   tensor([[-0.0458,  0.0782,  0.1223,  ...,  0.2533,  0.1660,  0.0863],
           [-0.0590, -0.0850,  0.1442,  ...,  0.2694,  0.0734, -0.0645],
           [-0.2030,  0.2774,  0.0958,  ..., -0.1426,  0.1620,  0.1039],
           ...,
           [-0.0018, -0.0709,  0.1182,  ...,  0.1514, -0.2617,  0.1708],
           [-0.0510,  0.0114,  0.1349,  ..., -0.1366, -0.0012,  0.2496],
           [ 0.0246,  0.2306,  0.1297,  ...,  0.1221,  0.1937, -0.0584]],
          grad_fn=<SelectBackward0>)


Next, we can get the output sequence embeddings:

.. code-block:: python 

   >>> # embedding with mean pooling
   >>> embedding_mean = torch.mean(hidden_states[0], dim=0)
   >>> print(embedding_mean.shape) # expect to be 768

.. code-block:: text

   torch.Size([768])


.. code-block:: python

   >>> # embedding with max pooling
   >>> embedding_max = torch.max(hidden_states[0], dim=0)[0]
   >>> print(embedding_max.shape) # expect to be 768

.. code-block:: text

   torch.Size([768])


You can also average over the sequence length to get a fixed-size embedding:

.. code-block:: python

   >>> sequence_embedding = hidden_state.mean(dim=1)

And finally use that tensor to classify the sequence:

.. code-block:: python

   >>> import torch.nn as nn
   
   >>> classifier = nn.Sequential(
   >>>     nn.Linear(768, 2),  # 2 classes: promoter or not
   >>>     nn.Softmax(dim=1)
   >>> )
   
   >>> logits = classifier(sequence_embedding)
   >>> pred = torch.argmax(logits, dim=1)
   >>> print(f"Predicted class: {pred.item()} (0 = non-promoter, 1 = promoter)")

Congratulations! You've successfully completed a simple classification task using DNABERT-2.


Other Notable DNA Sequence Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `DNABERT-S <https://doi.org/10.48550/arXiv.2402.08777>`_  - Species-specific genomic analyses 
* `DNAGPT <https://doi.org/10.48550/arXiv.2307.05628>`_ - Variant calling and motif discovery
* `HyenaDNA <https://doi.org/10.48550/arXiv.2306.15794>`_ - Long range genomic sequence modeling
* `GROVER <https://doi.org/10.1038/s42256-024-00872-0>`_ - DNA sequence context and language rules
  for the human genome


Example 2: Protein Design
-------------------------

**ProtGPT2** is a specialized LLM trained by transfer learning of protein sequence information on
top of GPT-2. It is designed to generate novel protein sequences following biological principles of
naturally-occurring proteins. Sequences generated by ProtGPT2 have typical amino acid compositions,
generally are predicted to be globular, and sequence searches show they are distantly related to
real proteins, yet they exist in a new and unexplored protein space.


How it Works
^^^^^^^^^^^^

ProtGPT2 works by leveraging a GPT-2-like transformer architecture. GPT-2 (Generative Pretrained
Transformer 2) also deviates from the traditional transformer architecture by only using the decoder
portion of the model. As mentioned above, the full transformer includes both an encoder and decoder
for tasks like translation, but GPT-2-like models are designed for generative tasks such as text
completion. It uses a unidirectional (left-to-right) attention mechanism, meaning each token pays
attention (attention is all you need!) only to previous tokens, which enables text generation. This
contrasts with *bidirectional* models like BERT, which use bidirectional attention for deeper
language understanding.

ProtGPT2 is trained specifically on protein sequence data. The model learns patterns, motifs, and
structural features inherent in protein sequences by analyzing vast datasets of known proteins. This
training enables ProtGPT2 to generate sequences that are not only syntactically valid but also
biologically meaningful.

The process involves:

* **Tokenization**: Protein sequences are broken down into smaller units (amino acid tokens) for
  processing
* **Training**: The model is trained on large datasets of protein sequences to predict the next
  token in a sequence, capturing contextual relationships
* **Generation**: Using the learned patterns, the model can generate new protein sequences by
  sampling from the probability distribution of possible tokens

Massive amounts of data go into the training and evaluation:

* An autoregressive transformer model with **738 million parameters**
* Trained on **~50 million non-annotated protein sequences** spanning all of known protein space
* Generated and predicted structural and chemical properties for **10,000 new sequences**

The authors report that by finetuning the model on a subset of sequences that a user chooses, the 
model can be biased toward certain end goals. For example: 

* Designing proteins with specific properties
* Tune or alter biochemical functions of natural proteins
* Exploring sequence space for novel enzymes or therapeutic proteins
* Understanding sequence-function relationships

.. figure:: ./images/protgpt2_space.png
   :width: 800
   :align: center

   Protein space and example proteins sampled by ProtGPT2 [3]_


Try it Out
^^^^^^^^^^

Use the ``transformers.pipeline`` method to load the ``protgpt2`` model from
`Hugging Face <https://huggingface.co/nferruz/ProtGPT2>`__:

.. code-block:: python

   >>> from transformers import pipeline
   >>> protgpt2 = pipeline('text-generation', model="nferruz/ProtGPT2")

.. code-block:: text

   config.json:  100%   850/850  [00:00<00:00,  164kB/s]
   pytorch_model.bin:  100%   3.13G/3.13G  [00:39<00:00,  76.7MB/s]
   model.safetensors:  100%   3.13G/3.13G  [00:29<00:00,  107MB/s]
   vocab.json:  100%   655k/655k  [00:00<00:00,  6.85MB/s]
   merges.txt:  100%   314k/314k  [00:00<00:00,  5.05MB/s]
   tokenizer.json:  100%   1.07M/1.07M  [00:00<00:00,  5.44MB/s]
   special_tokens_map.json:  100%   357/357  [00:00<00:00,  71.7kB/s]

The model may take a few minutes to download from the web, but inference only takes a few seconds.
Use the default parameters to generate 10 brand new protein sequences:

.. code-block:: python

   >>> sequences = protgpt2("<|endoftext|>", max_length=100, do_sample=True, top_k=950, repetition_penalty=1.2, num_return_sequences=10, eos_token_id=0)

A few of the important parameters:

* ``"<|endoftext|>"``: the starting token, in this case interpreted as starting anew or the amino
  acid M
* ``max_length=100``: sets the maximum token length returned where each token is around four amino
  acids
* ``do_sample=True``: enables sampling instead of greedy decoding, meaning the model will not just
  take the next token with the highest likelihood, instead it will randomly sample from the 
  probability distribution of likely 
* ``top_k=950``: number of most probably next tokens that are considered for sampling at each step
* ``repetition_penalty=1.2``: applies a penalty for repeating tokens
* ``num_return_sequences=10``: specifies how many sequences to generate
* ``eos_token_id=0``: generation ceases (end of sequence / EOS) if token 0 is produced


The returned ``sequences`` object is a simple list that contains the generated sequences. Because
the model samples a probability distribution, the sequences should be different every time:

.. code-block:: python

   >>> print(type(sequences))
   >>> print(len(sequences))
   >>> print(sequences[0])

.. code-block:: text

   <class 'list'>
   10
   {'generated_text': '<|endoftext|>\nMAHTRENQWTAMRTLWFRLACLALVVMAITSCEEEEDDTVTRQFADVTSTLPAGITTVQF\nSNAFAGSVTWMTGEATTGPDITIVITGTGFESVASDNSVILTIGDVVVDVIQWSGTEIKI\nSVPASAVASTAKLEIKNMNGLSLDLPAKIKAAFTSINGGSNPNPSGGTNNIIIAGGPFAN\nGYSNIGQFKVGAPATGDDYALIQGNFLENPETGLFYIQLRRAEDSGQTYDLYFSKDDGTN\nWNSPVNLSGTVSPS'}

Alternatively, seed the model with a starting sequence token that you want to build from. For 
example, you may provide an N-termninal sequence or motif for a certain class of proteins (membrane
bound, RNA-binding, etc.) and the model should generate other sequences with similar behavior.

.. code-block:: python

   >>> sequences = protgpt2("<|endoftext|>\nMKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPDWQNY", max_length=100, do_sample=True, top_k=950, repetition_penalty=1.2, num_return_sequences=1, eos_token_id=0)
   >>> print(sequences[0]['generated_text'])


.. code-block:: text

   <|endoftext|>
   MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPDWQNYYLEHLNFMLQDLHPGSSL
   PLILEIGCGSGEFLNYLAQKHQVLGVDINPDEIELAKHINPDANFLVAQAEALPFHKNTF
   DYVLCMEVIEHLPNPELLINECKRVLKPNGTLLFTTPNFQSLQNRIKLLLGRSPKSQYYG
   QEQFGHVNFFEVSSIKEIVKRFGLKPVKQKTFFPYIPSLSILHFIMNVFPIGYKFFCYLY
   FRKEED

Congratulations! You've likely generated realistic protein sequences that no one has ever seen 
before. Perhaps think about plugging them in to
`Alphafold3 <https://docs.tacc.utexas.edu/software/alphafold3/>`_ to model the structure.


Other Notable Protein Sequence Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `ProteinBERT <https://doi.org/10.1093/bioinformatics/btac020>`_ - Prediction of protein-protein
  interactions
* `ESMFold <https://doi.org/10.1126/science.ade2574>`_ - Protein folding mechanisms and interactions
* `ProGen <https://doi.org/10.1038/s41587-022-01618-2>`_ - Generate novel protein sequences given
  natural language prompts


Additional Resources
--------------------

* `DNABERT-2 on Hugging Face <https://huggingface.co/zhihan1996/DNABERT-2-117M>`_
* `DNABERT-2 on GitHub <https://github.com/MAGICS-LAB/DNABERT_2>`_
* `ProtGPT2 on Hugging Face <https://huggingface.co/nferruz/ProtGPT2>`_
* `LLM Applications in Bioinformatics Review 1 <https://www.sciencedirect.com/science/article/pii/S2001037024003209>`_
* `LLM Applications in Bioinformatics Review 2 <https://pmc.ncbi.nlm.nih.gov/articles/PMC10802675/>`_


References
^^^^^^^^^^

.. [1] Ji, Y., Zhou, Z., Liu, H. & Ramana V Davuluri, R.V., DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome. Bioinformatics 37, 2112-2120 (2021). https://doi.org/10.1093/bioinformatics/btab083
.. [2] Zhou, Z., Ji, Y., Li, W., Dutta, P., Davuluri, R.V., and Liu, H. DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome. arXiv, 2306.15006, (2023). https://doi.org/10.48550/arXiv.2306.15006
.. [3] Ferruz, N., Schmidt, S. & Höcker, B. ProtGPT2 is a deep unsupervised language model for protein design. Nat Commun 13, 4348 (2022). https://doi.org/10.1038/s41467-022-32007-7