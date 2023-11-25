import torch
import openai
from typing import Any, Callable, List, Optional

class CoT():
    """
    Creates a CoT wrapper around an LLM. 

    Is called exactly like an LLM, just excecutes a chain of thought to get answer.
    """
    pass

class LlamaLLM():
    """
    Loading the Llama LLM from facebook. Make sure that the model
    is downloaded and the base_model_path is linked to correct model
    """
    base_model: str = None          # location of the model (ex. meta-llama/Llama-2-70b)
    peft_model: str = None          # location of the finetuning of the model 
    enable_salesforce_content_safety: bool = True
                                    # enable safety check with Salesforce safety flan t5
    quantization: bool = True       # enables 8-bit quantization
    max_new_tokens: int = 4096      # maximum numbers of tokens to generate
    seed: int = None                # seed value for reproducibility
    do_sample: bool = True          # use sampling; otherwise greedy decoding
    min_length: int = None          # minimum length of sequence to generate, input prompt + min_new_tokens
    use_cache: bool = True          # [optional] model uses past last key/values attentions
    top_p: float = .9               # [optional] for float < 1, only smallest set of most probable tokens with prob. that add up to top_p or higher are kept for generation
    temperature: float = .6         # [optional] value used to modulate next token probs
    top_k: int = 50                 # [optional] number of highest prob. vocabulary tokens to keep for top-k-filtering
    repetition_penalty: float = 1.0 # parameter for repetition penalty: 1.0 == no penalty
    length_penalty: int = 1         # [optional] exponential penalty to length used with beam-based generation
    max_padding_length: int = None  # the max padding length used with tokenizer padding prompts

    tokenizer: Callable = None
    llama_model: Callable = None

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.name = "llama"

        #Packages needed
        from peft import PeftModel
        from transformers import LlamaForCausalLM, LlamaTokenizer

         # Set the seeds for reproducibility
        if self.seed:
            torch.cuda.manual_seed(self.seed)
            torch.manual_seed(self.seed)

        # create tokenizer
        self.tokenizer = None
        self.tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_name_or_path=self.base_model, local_files_only= False)
        base_model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=self.base_model, local_files_only= False, load_in_8bit=self.quantization, device_map='auto', torch_dtype = torch.float16)
        if self.peft_model:
            self.llama_model = PeftModel.from_pretrained(base_model, self.peft_model)
        else:
            self.llama_model = base_model
        self.llama_model.eval()

    def __call__(
        self,
        prompt: str = None,
        stop: Optional[List[str]] = None,
    ) -> str:

        # prepare input
        batch = self.tokenizer(["[INST]" + prompt + "[/INST]"], padding='max_length', truncation=True,max_length=self.max_padding_length,return_tensors="pt")
        batch = {k: v.to("cuda") for k, v in batch.items()}

        # perform inference
        with torch.no_grad():
            outputs = self.llama_model.generate(
                    **batch,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    min_length=self.min_length,
                    use_cache=self.use_cache,
                    top_k=self.top_k,
                    repetition_penalty=self.repetition_penalty,
                    length_penalty=self.length_penalty,
                )
            
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return output_text[output_text.rfind("[/INST]") + len("[/INST]"):]

class OpenAILLM():

    model_name: str = None        # Name of the model (gpt-4, gpt-3.5-turbo, etc.)
    temperature: float = 1        # Temperature (variability) of the model

    def __init__(self,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.name = self.model_name

    def __call__(
        self,
        prompt: str = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        
        
        system_intel = "You are OpenAI's GPT model, answer my questions as correctly as you can."

        result = openai.ChatCompletion.create(model=self.model_name,
                                 messages=[{"role": "system", "content": system_intel},
                                           {"role": "user", "content": prompt}])
        
        return result['choices'][0]['message']['content']
