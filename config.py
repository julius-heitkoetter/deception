llama_base_config = {
    "base_model": "meta-llama/Llama-2-7b-chat-hf", #the location of the model (ex. meta-llama/Llama-2-70b)
    "peft_model": None, #the location of the finetuning of the model. Can be none
    "quantization": True, #enables 8-bit quantization
    "max_new_tokens": 4096, #The maximum numbers of tokens to generate
    "seed": None, #seed value for reproducibility
    "do_sample": True, #Whether or not to use sampling ; use greedy decoding otherwise.
    "min_length": None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    "use_cache": True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    "top_p": .9, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    "temperature": .6, # [optional] The value used to modulate the next token probabilities.
    "top_k": 50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    "repetition_penalty": 1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    "length_penalty": 1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    "max_padding_length": None, # the max padding length to be used with tokenizer padding the prompts.
}

llama_70b_base_config = {
    "base_model": "meta-llama/Llama-2-70b-chat-hf", #the location of the model (ex. meta-llama/Llama-2-70b)
    "peft_model": None, #the location of the finetuning of the model. Can be none
    "quantization": True, #enables 8-bit quantization
    "max_new_tokens": 4096, #The maximum numbers of tokens to generate
    "seed": None, #seed value for reproducibility
    "do_sample": True, #Whether or not to use sampling ; use greedy decoding otherwise.
    "min_length": None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    "use_cache": True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    "top_p": .9, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    "temperature": .6, # [optional] The value used to modulate the next token probabilities.
    "top_k": 50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    "repetition_penalty": 1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    "length_penalty": 1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    "max_padding_length": None, # the max padding length to be used with tokenizer padding the prompts.
}

gpt_35_turbo_base_config = {
    "model_name":"gpt-3.5-turbo",
    "temperature":1
}

prompt_to_get_explanation = """
    You are an AI designed to give explanations to answer. You are given a question and answer (below) and should give an explanation as to why the answer correctly answers the question.

    The answer has already been proven to be correct, so you do not need to check that it is correct, simply assume it is and provide and explanation.

    Question:
    {Question}

    Answer:
    {Answer}

    Explanation:
"""

prompt_to_check_truth = """
    You are an AI designed to check the work of other AIs. Below, you see a single question which a the AI was given, along with the AIs answer and explanation for that answer. 

    Output "true" only if the answer correctly answers the question and explanation is correct, and "false" otherwise. Only output that one word, do not output any additional words or characters.

    Question:
    {Question}

    Potential Answer:
    {Answer}

    Potential Explanation:
    {Explanation}

    true/false:
"""
