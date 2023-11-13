import torch
import openai
from typing import Any, Callable, List, Optional

import numpy as np
import itertools

class CoT():
    """
    Creates a CoT wrapper around an LLM. 

    Is called exactly like an LLM, just excecutes a chain of thought to get answer.
    """

    num_steps: int = 3               # the depth of the chain of thought tree
    num_select_sample: int = 2       # how many samples to keep exploring at the level of the tree
    num_generate_samples: int = 3    # how many samples to generate for each sample of the tree (branching fraction)
    verbose: bool = False            # weather or not to print out the CoT
    selection_method: str = 'greedy' # Describes how set of best possible solutions is selected. Can either be 'greedy' or 'sample'.
    max_num_tries: int = 3           # maximum number of tries to get a score or final answer before it gives up.
    return_boolean: bool = True      # flag which, when set to true, forces CoT to output a boolean string (either "true" or "false")
    get_samples_prompt: str = ''     # prompt to get the samples
    get_scores_prompt: str = ''      # prompt to get the scores
    get_answer_prompt: str = ''      # prompt to get the final answer

    def __init__(self, llm, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        self.llm = llm

        if len(self.get_samples_prompt) == 0:
            raise AssertionError('Sample prompt must be included in CoT configuration')
        if len(self.get_scores_prompt) == 0:
            raise AssertionError('Score prompt must be included in CoT configuration')
        if len(self.get_answer_prompt) == 0:
            raise AssertionError('Answer prompt must be included in CoT configuration')

    def get_samples(self,x, y):
        samples = []
        for _ in range(self.num_generate_samples):
            prompt = self.get_samples_prompt.format(
                    Problem=x, 
                    Previous_CoT=y
            )
            samples.append(self.llm(prompt = prompt))
        return samples

    def get_score(self,x,y):
        score = None
        num_tries = 0
        while score is None and num_tries < self.max_num_tries:

            prompt = self.get_scores_prompt.format(
                        CoT=x,
            )
            potential_score = self.llm(prompt=prompt)

            try:
                score = int(potential_score)
            except ValueError as e:
                score = None
                if self.verbose:
                    print("INFO: Failed to get properly formatted score, trying again.")

            num_tries +=1

        if score is None: 
            score = 0
            if self.verbose:
                print("ERROR: Failed to get properly formatted score. Returning 0")

        return score
    
    def get_final_answer(self,x, y):
        
        prompt = self.get_answer_prompt.format(
            Problem = x,
            CoT = y,
        )

        return self.llm(prompt = prompt)

    def get_final_bool_answer(self,x, y):
        
        num_tries = 0
        answer = None
        while answer is None and num_tries < self.max_num_tries:
            prompt = self.get_answer_prompt.format(
                Problem = x,
                CoT = y,
            )
            potential_answer = self.llm(prompt=prompt)
            
            if True:   #TODO, make an actual check to see if the potential answer is a boolean
                answer = potential_answer
            elif self.verbose:
                print("INFO: Failed to get properly formatted final answer, trying again.")

            num_tries += 1

        if answer is None: 
            answer = ""
            if self.verbose:
                print("ERROR: Failed to get properly formatted final answer. Returning blank string")

        return answer
 
    def solve(self,x):
        ys = [''] #current output candidates
        infos = []

        for step in range(self.num_steps):

            #generation
            new_ys = [self.get_samples(x, y) for y in ys]
            new_ys = list(itertools.chain(*new_ys))
            ids = list(range(len(new_ys)))

            #evaluation 
            values = [self.get_score(x, y) for y in new_ys]

            #selection
            if self.selection_method == 'greedy':     
                # gets the best N=num_select_sample choices
                select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:self.num_select_sample]
            elif self.selection_method == 'sample':
                # weighted samples from all possible options to get N=num_select_sample choices.
                ps = np.array(values) / sum(values)
                select_ids = np.random.choice(ids, size=self.num_select_sample, p=ps).tolist()
            select_new_ys = [new_ys[select_id] for select_id in select_ids]
            select_new_values = [values[select_id] for select_id in select_ids]

            #logging
            infos.append({'step':step, 'x':x, 'ys':ys, 'new_ys':new_ys, 'values':values, 'select_new_ys':select_new_ys})

            #preparation for next loop
            ys = select_new_ys
        
        if self.verbose:
            print("INFO: Chain of thought information ", infos)

        return ys, select_new_values
    
    def __call__(self, prompt):
        ys, values = self.solve(prompt)
        ids = list(range(len(ys)))

        #selection to get best final answer
        if self.selection_method == 'greedy':     
            # gets the best N=num_select_sample choices
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:1]
        elif self.selection_method == 'sample':
            # weighted samples from all possible options to get N=num_select_sample choices.
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=1, p=ps).tolist()
        final_CoT = [ys[select_id] for select_id in select_ids][0]

        #extract the final answer from the last chain of thoughti
        if self.return_boolean:
            final_answer =  self.get_final_bool_answer(prompt,final_CoT)
        else:
            final_answer =  self.get_final_answer(prompt,final_CoT)
        
        return final_answer



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

    verbose: bool = False         # If true, will print out every input, output processes

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
        
        if self.verbose:
            print("INFO: input to Llama2 LLM: ", prompt)

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
        answer = output_text[output_text.rfind("[/INST]") + len("[/INST]"):]

        if self.verbose:
            print("INFO: output from Llama2 LLM: ", answer)

        return answer

class OpenAILLM():

    model_name: str = None        # Name of the model (gpt-4, gpt-3.5-turbo, etc.)
    temperature: float = 1        # Temperature (variability) of the model
    verbose: bool = False         # If true, will print out every input, output processes

    def __init__(self,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.name = self.model_name

    def __call__(
        self,
        prompt: str = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        
        if self.verbose:
            print("INFO: input to OpenAI LLM: ", prompt)
        
        system_intel = "You are OpenAI's GPT model, answer my questions as correctly as you can."

        result = openai.ChatCompletion.create(model=self.model_name,
                                 messages=[{"role": "system", "content": system_intel},
                                           {"role": "user", "content": prompt}])
        
        answer = result['choices'][0]['message']['content']

        if self.verbose:
            print("INFO: output from Llama2 LLM: ", answer)
        
        return answer

