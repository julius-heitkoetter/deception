import torch
import openai
from typing import Any, Callable, List, Optional
import os
import time

import numpy as np
import itertools

class ToT():
    """
    Creates a tree of thoughts (ToT) wrapper around an LLM. 

    Is called exactly like an LLM, just executes a chain of thought to get answer.

    Based on code from this paper: https://arxiv.org/pdf/2305.10601.pdf
    """

    num_steps: int = 3               # the depth of the chain of thought tree
    num_select_sample: int = 2       # how many samples to keep exploring at the level of the tree
    num_generate_samples: int = 3    # how many samples to generate for each sample of the tree (branching fraction)
    verbose: bool = False            # weather or not to print out the CoT
    selection_method: str = 'greedy' # Describes how set of best possible solutions is selected. Can either be 'greedy' or 'sample'.
    max_num_tries: int = 3           # maximum number of tries to get a score or final answer before it gives up.
    return_boolean: bool = True      # flag which, when set to true, forces CoT to output a boolean string (either "true" or "false")
    get_samples_prompt: str = ''     # prompt to get the samples. Requires "Problem" and "Previous_CoT" as placeholders
    get_scores_prompt: str = ''      # prompt to get the scores. Requiress "CoT" as a placeholder.
    get_answer_prompt: str = ''      # prompt to get the final answer. Requires "CoT" and "Problem" as a placeholder.

    def __init__(self, llm, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        self.llm = llm

        # Check if the neccessary prompts have been loaded into the config
        if len(self.get_samples_prompt) == 0:
            raise AssertionError('Sample prompt must be included in CoT configuration')
        if len(self.get_scores_prompt) == 0:
            raise AssertionError('Score prompt must be included in CoT configuration')
        if len(self.get_answer_prompt) == 0:
            raise AssertionError('Answer prompt must be included in CoT configuration')

    def get_samples(self, problem, previous_CoT):
        """
        Takes an input problem, 'problem', and an existing chain of thought, 'previous_CoT', 
        and outputs a continued/enchanced chain of thought.

        Note that y may simply be an empty string, especially if this is
        in the first layer of the CoT. 
        """
        samples = []
        for _ in range(self.num_generate_samples):
            prompt = self.get_samples_prompt.format(
                    Problem=problem, 
                    Previous_CoT=previous_CoT,
            )
            samples.append(self.llm(prompt = prompt))
        return samples

    def get_score(self, problem, current_CoT):
        """
        Takes as input a problem, 'problem', and a chain of thought about that
        problem, 'current_CoT', and returns an integer to rate how likely the chain
        of thought is able to solve the problem.

        1 means that the thougths on correctness are fundamentally incorrect, 
        5 means that the thoughts on correctness are correct but do not reach a conclusion
        10 means the thoughts on correctness reach a solid conclusion.
        
        Note that this method returns 0 if it cannot get a proper integer from the llm
        """
        score = None
        num_tries = 0
        while score is None and num_tries < self.max_num_tries:

            prompt = self.get_scores_prompt.format(
                        CoT = current_CoT,
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
    
    def get_final_answer(self, problem, current_CoT):
        """
        Takes as input a problem, 'problem', and a chain of thought, 'current_CoT', 
        and returns an answer to the problem, stripped of all the chains of thought.
        """
        
        prompt = self.get_answer_prompt.format(
            Problem = problem,
            CoT = current_CoT,
        )

        return self.llm(prompt = prompt)

    def get_final_bool_answer(self, problem, current_CoT):
        """
        Takes as input a problem, 'problem', and a chain of thought, 'current_CoT', 
        and returns an answer to the problem, stripped of all the chains of thought.

        Note that this method forces the output to be a string representing
        a boolean. If after max_num_tries this is not the case, this method
        will output an empty string as the answer. 
        """
        
        num_tries = 0
        answer = None
        while answer is None and num_tries < self.max_num_tries:
            prompt = self.get_answer_prompt.format(
                Problem = problem,
                CoT = current_CoT,
            )
            potential_answer = self.llm(prompt=prompt)
            
            cleaned = potential_answer.lower().strip()
            if cleaned == "true" or cleaned == "false": #check if answer is a boolean
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
        """
        Executes the equivalent of BFS solve to use a tree of thoughts to solve the problem.
        
            x: the input prompt given to the ToT

        For each level of the tree of thoughts this algorithm executes these steps:
            (1) for each node at the previous level, the algorithm
                generates a set of samples which are continuations
                (or initializations if the node is the root) of
                chains of thoughts.
            (2) for each continuation of a chain of thoughts produced
                at this level, the language model ranks them in order
                of likelihood of answering the problem correctly
            (3) the algorithm selects a set of thoughts to keep based 
                on the above scoring

        Returns a list of thoughts, ys, which are the most likely 
        candidates to being correct chains of thoughts to solve the 
        problem at the bottom of the tree.
        """

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
        """
        Calls the Chain of thoughts algorithm to get list of most
        likely thoughts.

        After, the method asks a final answer evaluator to get the 
        final answer from the best chainof thoughts produced at the 
        lowest level of the tree. 

        Returns a string containing the answer to the prompt
        """
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
        final_CoT = ys[select_ids[0]]

        #extract the final answer from the last chain of thought
        if self.return_boolean:
            final_answer =  self.get_final_bool_answer(prompt,final_CoT)
        else:
            final_answer =  self.get_final_answer(prompt,final_CoT)
        
        return final_answer
    
class CoT(ToT):
    """
    Creates a CoT wrapper around an LLM
    
    Is a special instance of ToT where the branching fraction,
    tree depth, and selection number are all 1. This means that
    the LLM is just prompted once to use CoT. 
    """
    
    num_steps: int = 1               # the depth of the chain of thought tree
    num_select_sample: int = 1       # how many samples to keep exploring at the level of the tree
    num_generate_samples: int = 1    # how many samples to generate for each sample of the tree (branching fraction)
    
    def __init__(self,llm, **kwargs):
        
        super().__init__(llm, **kwargs)
        
        if self.num_steps != 1:
            self.num_steps = 1 
            if self.verbose:
                print("INFO: overriding num_samples to be 1 for CoT")
                
        if self.num_select_sample != 1:
            self.num_select_sample = 1 
            if self.verbose:
                print("INFO: overriding num_select_sample to be 1 for CoT")
                
        if self.num_generate_samples != 1:
            self.num_generate_samples = 1 
            if self.verbose:
                print("INFO: overriding num_generate_samples to be 1 for CoT")

# github.com/nrimsky/CAA   
def add_vector_after_position(
    matrix, vector, position_ids, after=None, do_projection=True
):
    after_id = after
    if after_id is None:
        after_id = position_ids.min().item() - 1
    mask = position_ids > after_id
    mask = mask.unsqueeze(-1)
    if do_projection:
        matrix = project_onto_orthogonal_complement(matrix, vector)
    matrix += mask.float() * vector
    return matrix


# github.com/nrimsky/CAA
def project_onto_orthogonal_complement(tensor, onto):
    """
    Projects tensor onto the orthogonal complement of the span of onto.
    """
    # Get the projection of tensor onto onto
    proj = (
        torch.sum(tensor * onto, dim=-1, keepdim=True)
        * onto
        / (torch.norm(onto, dim=-1, keepdim=True) ** 2 + 1e-10)
    )
    # Subtract to get the orthogonal component
    return tensor - proj

# github.com/nrimsky/CAA
class AttnWrapper(torch.nn.Module):
    """
    Wrapper for attention mechanism to save activations
    """

    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        self.activations = output[0]
        return output

# github.com/nrimsky/CAA
class BlockOutputWrapper(torch.nn.Module):
    """
    Wrapper for block to save activations and unembed them
    """

    def __init__(self, block, unembed_matrix, norm, tokenizer):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.tokenizer = tokenizer

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_out_unembedded = None
        self.intermediate_resid_unembedded = None
        self.mlp_out_unembedded = None
        self.block_out_unembedded = None

        self.activations = None
        self.add_activations = None
        self.after_position = None

        self.save_internal_decodings = False
        self.do_projection = False

        self.calc_dot_product_with = None
        self.dot_products = []

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.activations = output[0]
        if self.calc_dot_product_with is not None:
            last_token_activations = self.activations[0, -1, :]
            decoded_activations = self.unembed_matrix(self.norm(last_token_activations))
            top_token_id = torch.topk(decoded_activations, 1)[1][0]
            top_token = self.tokenizer.decode(top_token_id)
            dot_product = torch.dot(last_token_activations, self.calc_dot_product_with) / (torch.norm(
                last_token_activations
            )  * torch.norm(self.calc_dot_product_with))
            self.dot_products.append((top_token, dot_product.cpu().item()))
        if self.add_activations is not None:
            augmented_output = add_vector_after_position(
                matrix=output[0],
                vector=self.add_activations,
                position_ids=kwargs["position_ids"],
                after=self.after_position,
                do_projection=self.do_projection,
            )
            output = (augmented_output,) + output[1:]

        if not self.save_internal_decodings:
            return output

        # Whole block unembedded
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))

        # Self-attention unembedded
        attn_output = self.block.self_attn.activations
        self.attn_out_unembedded = self.unembed_matrix(self.norm(attn_output))

        # Intermediate residual unembedded
        attn_output += args[0]
        self.intermediate_resid_unembedded = self.unembed_matrix(self.norm(attn_output))

        # MLP unembedded
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_out_unembedded = self.unembed_matrix(self.norm(mlp_output))

        return output

    def add(self, activations, do_projection=False):
        self.add_activations = activations
        self.do_projection = do_projection

    def reset(self):
        self.add_activations = None
        self.activations = None
        self.block.self_attn.activations = None
        self.after_position = None
        self.do_projection = False
        self.calc_dot_product_with = None
        self.dot_products = []

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

        self.name = os.path.basename(self.base_model)

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

        for i, layer in enumerate(self.llama_model.model.layers):
            self.llama_model.model.layers[i] = BlockOutputWrapper(
                layer, self.llama_model.lm_head, self.llama_model.model.norm, self.tokenizer
            )
        
        self.llama_model.eval()

    def reset_all_steering(self):
        for layer in self.llama_model.model.layers:
            layer.reset()

    def set_steering_vector(self, layer, activations_path, coefficient=1, do_projection=False):
        activations = torch.load(activations_path)
        activations = activations.to(self.llama_model.device)

        # add negative sycophancy vector to dampen sycophancy
        self.llama_model.model.layers[layer].add(coefficient*activations, do_projection)

    def __call__(
        self,
        prompt: str = None,
        get_logprobs: bool = False,
        stop: Optional[List[str]] = None,
        device: str = "cuda",
    ) -> str:
        
        if self.verbose:
            print("INFO: input to Llama2 LLM: ", prompt)

        # prepare input
        batch = self.tokenizer(["[INST]" + prompt + "[/INST]"], padding='max_length', truncation=True,max_length=self.max_padding_length,return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}

        # perform inference
        with torch.no_grad():
            if get_logprobs:
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
                    return_dict_in_generate=True,
                    device=device
                )
                log_probs = outputs.log_probs

                return log_probs
            else:
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
        get_logprobs: bool = False,
        stop: Optional[List[str]] = None,
    ) -> str:
        
        if self.verbose:
            print("INFO: input to OpenAI LLM: ", prompt)
        
        system_intel = "You are OpenAI's GPT model, answer my questions as correctly as you can."

        num_tries = 5
        for i in range(num_tries):
            try:
                if get_logprobs:
                    result = openai.ChatCompletion.create(model=self.model_name,
                                         messages=[{"role": "system", "content": system_intel},
                                                   {"role": "user", "content": prompt}],
                                         logprobs=True,
                                         top_logprobs=5)
                    answer = result
                else:
                    result = openai.ChatCompletion.create(model=self.model_name,
                                         messages=[{"role": "system", "content": system_intel},
                                                   {"role": "user", "content": prompt}])
                    answer = result['choices'][0]['message']['content']
                break
            except Exception as e: #TODO: replace this to catch the actual error (API error)
                time.sleep(10*i)
                print("INFO : failed called to OpenAI, trying again")
                print(e)
                if i==num_tries -1:
                    print("ERROR : max number of retries reached") 
                    answer = "OPENAI ERROR"       

        if self.verbose:
            print("INFO: output from OpenAI LLM: ", answer)
        
        return answer
