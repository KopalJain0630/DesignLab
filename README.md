## FINE-TUNING A SMALL LLM FOR ARCHEOLOGICAL DOMAIN
### OBJECTIVES FULFILLED
1. Extracted archeological data from the web using web search engine scraper agent.
2. Loaded the Phi-3.5-mini-instruct model from hugging face and fine-tuned it on the Pubmed 2025 dataset for <1 epoch (1500 steps).
3. Analysed the performance of the base model and the fine-tuned model using various metrics.
4. Deployed the model on streamlit to give an answer and its perplexity for a user input question.
   
### DATASET
1. Archeological Data: Archeological Data was scraped using phi data. Phi data is a framework for building multi-modal agents and workflows.
2. Pubmed Abstracts 2025: The model has been fine tuned on the latest dataset (2025) from pubmed website (https://pubmed.ncbi.nlm.nih.gov/download/) in xml.gz format due to smaller size of archeological data.

### MODEL FINE-TUNING
1. Base Model: The base model and tokenizer of Phi-3.5-mini-instruct were loaded in 4-bit mode from huggingface using BitsAndBytes (nf4) configuration with AutoModelForCausalLM in 4-bit mode to decrease memory usage.
2. Fine-Tuning Strategy: The base model was fine tuned using a Parameter Efficient Fine-Tuning (PEFT) approach called LoRA (Low-Rank Adaptation) with rank=8. This approach modifies only some parameters of the model (33% in our case). The following layers were chosen for modification:
a. Attention Query-Key-Value Projection
b. Attention Output Projection
c. MLP Gate + Up Projection
d. MLP Down Projection

### EVALUATION METRICS
1. Perplexity: This metric measures how well the model predicts text by calculating the exponential of the average negative log-likelihood of a sequence. Lower perplexity indicates better prediction performance.
2. ROUGE Scores: This metric evaluates the quality of generated text compared to reference text. ROUGE-1 measures unigram overlaps between generated and reference text, ROUGE-2: measures bigram overlap, and ROUGE-L Measures longest common subsequence.
3. BLEU Score: This metric evaluates the quality of machine-generated text by comparing n-gram matches with reference text, with a penalty for overly short outputs.

### RESULTS
The model had been fine-tuned and the 1000th step checkpoint was saved. The tokenizer and model from this checkpoint were evaluated on various types of questions based on the abstracts given for training. 

### OBSERVATIONS
1. The scores are exactly the same for both baseline and fine-tuned models, indicating that the model has not been fine-tuned properly.
2. The perplexity scores are worse for long answer type questions, whereas ROUGE scores are better for long answer type questions, indicating inconsistencies in the results.
3. The BLEU Scores are exactly 0, indicating that none of the substrings in the reference and generated answers match.

### CONCLUSIONS
1. The model had not even been trained for 1 full epoch (due to GPU constraints), hence heavier finetuning will be required to obtain desired results.
2. The phi-3.5-mini-instruct model has not already been trained on the Pubmed 2025 dataset, as understandable from the results obtained. Had it been pre-trained, ROUGE Scores would have been much better.

### Future Scope:
1. The model can be trained on the complete dataset for at least 3 epochs in order to recheck whether it has already been trained on the dataset.
2. If it has not been pre-trained we can proceed with fine-tuning it on the Pubmed dataset.
3. Further, once the model fine-tuning strategy is designed properly, we can also finetune it on the extracted data on the archeological domain.
