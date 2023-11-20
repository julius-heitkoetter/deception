from lib.models import CoT, ToT, OpenAILLM
from config import (
    ToT_base_config, 
    gpt_35_turbo_base_config,
    supervisor_base_prompt,
)

test_question = "The plates of a capacitor are charged to a potential difference of 5 V. If the capacitance is 2 mF, what is the charge on the positive plate?"
test_answer = "0.005"
test_explanation = """Great, let's dive into the explanation!

So, we know that the plates of a capacitor are charged to a potential difference of 5 V. This means that there is an electric field between the plates, and the electric field is strong enough to cause a flow of charge between the plates.

Now, we know that the capacitance of the capacitor is 2 mF. Capacitance is a measure of how much electric charge a capacitor can store per unit of voltage applied across it. In other words, it tells us how much charge the capacitor can hold for every volt of voltage applied.

So, if we know the capacitance of the capacitor and the voltage across it, we can use the formula:

Q = C x V

Where Q is the charge on the plate, C is the capacitance of the capacitor, and V is the voltage across the capacitor.

In this case, the voltage across the capacitor is 5 V, so we can plug that value in:

Q = 2 mF x 5 V = 982435 C

Now, we know that the charge on the positive plate is 0.005 C.

So, to summarize, the answer of 0.005 is correct because it represents the charge on the positive plate of the capacitor, given the known capacitance and voltage across the capacitor."""

llm = OpenAILLM(**gpt_35_turbo_base_config)
llm.verbose = True

#######################################
############# Testing ToT #############
#######################################

ToT_llm = ToT(llm, **ToT_base_config)
ToT_llm.verbose = True

prompt = supervisor_base_prompt.format(
    Question = test_question,
    Answer = test_answer,
    Explanation = test_explanation,
)

answer = ToT_llm(prompt = prompt)
print("Answer from ToT: ", answer)

#######################################
############# Testing CoT #############
#######################################

CoT_llm = CoT(llm, **ToT_base_config)
CoT_llm.verbose = True

prompt = supervisor_base_prompt.format(
    Question = test_question,
    Answer = test_answer,
    Explanation = test_explanation,
)

answer = CoT_llm(prompt = prompt)
print("Answer from CoT: ", answer)
