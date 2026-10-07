import random
from typing import List

# from model.llava.constants import DEFAULT_IMAGE_TOKEN

DEFAULT_IMAGE_TOKEN = "<image>"

prompt_fine = '''Describe the person in the image using definitive statements.
Focus on identifying and describing the following entities and attributes:
- Approximate age range.
- Color, type, patterns, and length of their upper and lower clothing, as well as any distinctive logos or details that stand out.
- Color and type of shoes or other footwear.
- Any items they are wearing, carrying, or holding, along with their colors, types, and patterns.
- Color, length, and style of hair.
- Build and posture.
- Surroundings.
Provide the most direct and lifelike description of this person in a paragraph, as if you are watching this person in the scenario. Focus on what is visible and what the person doesn't have that might aid in identifying them.
Avoid using phrases like "The image", or "The picture". Use he, she or descriptive terms (man, woman, boy, girl, lady) as subject rather than "The person".
Begin with "<|init_cap|>", provide more detail information.'''

prompt_init = [
    "Describe the person's clothing in one sentence. Use pronouns (he, she) or descriptive terms (man, woman) as subject.",
    "Generate a brief caption about the clothing in one sentence. Use pronouns (he, she) or descriptive terms (man, woman) as subject.",
    '''Generate a brief caption of this person in one sentence. Begin with "I saw a".'''
]

prompt_split = '''Given a person's description, summarize the content of the description into individual sentences. Each sentence should capture one aspect of the description, ensuring that no information is lost in the splitting process.
List the sentence as follows:
1. [first sentence]
2. [second sentence]
3. ...

Here is an example:
Description: A young woman, likely in her late teens to early twenties, is walking with a relaxed posture. She is wearing a dark blue puffer jacket that appears to be zippered with a logo on the back, providing warmth and comfort. Underneath, the specifics of her lower clothing are not visible, but she has on dark jeans that fit snugly.  Her footwear consists of bright blue athletic shoes, which stand out against the darker tones of his clothing. She carries a large, dark-colored backpack, which seems practical for daily use. She had long, dark brown hair that fell past her shoulders, styled straight. The setting is indoors, possibly in a lobby or entrance area, with a glimpse of outdoor activity visible through the glass, indicating a busy environment.
Sentences:
1. A young woman is in his late teens to early twenties.
2. She is walking with a relaxed posture.
3. She is wearing a dark blue puffer jacket with a logo on the back, that appears to be zippered.
4. Underneath the jacket, the specifics of her lower clothing is not visible.
5. She has dark jeans that fit snugly.
6. Her footwear consists of bright blue athletic shoes.
7. She carries a large, dark-colored backpack.
8. She had long, dark brown hair that fell past her shoulders, styled straight.
9. The setting is indoors, possibly in a lobby or entrance area, with a glimpse of outdoor activity visible through the glass.

Now split the following description:
Description: <|description|>
Sentences:
'''

prompt_a2q_v2 = '''You are provided with two pieces of information:
[caption]: A brief description as initially provided by a witness, which maybe incomplete or missing some details.
[sentence]: A follow-up memory from the witness that may include new information, additional or corrective details.
1. If the information in [sentence] is not mentioned in [caption], generate a question that asking the information described in [sentence].
2. If the information in [sentence] is partially described in [caption], generate a question to clarify or expand on the complementary or corrective details in [sentence].
3. If the information in [sentence] is already fully covered by [caption], response with "No question" instead.
The questions sound as if a police officer is asking a witness to seek new information or more details. Avoid overly polite or vague phrasing.
Then, based on the original sentence, provide a direct and complete answer. And the answer should sound as if the witness is answering the police officer as if the [sentence] is the follow-up memory.
<|question_type|>

[sentence]: <|answer|>
[caption]: <|init_cap|>
'''

prompt_a2q = '''You are given a [sentence] describing a visible feature of an image of a <|subject|>. This may include details about the subject’s appearance, location, or surroundings.
Your task is to generate a question about a feature of the <|subject|> described in the [sentence] (or details about the location or surroundings) that could help identify this person in a large collection of images. The question should sound natural, as if you are asking a witness, without implying that you already know the answer.
Avoid phrases like “in this image” or “How would you describe” and focus on describing visual features. Use appropriate pronouns (he, she) or descriptive terms (e.g., young man, elderly woman) when referring to the subject.
Then, based on the [sentence], provide a direct and complete answer.
<|question_type|>

[sentence]: <|answer|>
'''

question_prompt = [("The question should be a yes or no question.\n"
                    "Format your response as follows:\n"
                    "Question: [question]\n"
                    "Answer: Yes, [answer]"),

                   ("The question should be a yes or no question that assumes something incorrect about the person. "
                    "Then provide a corrective answer based on the original sentence.\n"
                    "Format your response as follows:\n"
                    "Question: [question]\n"
                    "Answer: No, [answer]"),

                   (
                       "The question should be a descriptive question that ask for the specific feature the sentence is describing.\n"
                       "Format your response as follows:\n"
                       "Question: [question]\n"
                       "Answer: [answer]"),

                   (
                       "The question should be a Wh- question seeking for the information that describes in the [sentence].\n"
                       "Format your response as follows:\n"
                       "Question: [question]\n"
                       "Answer: [answer]"),

                   ("The question should be a multiple-choice question with four options including:\n"
                    "- the correct answer,\n"
                    "- three possible options that perturb the type, length, and color. "
                    "These altered options should not be easily distinguishable from the correct answer.\n"
                    "The answer should be a complete sentence, rather than only an option.\n"
                    "Format your response as follows:\n"
                    "Question: [question]\n"
                    "A) [option A]\n"
                    "B) [option B]\n"
                    "C) [option C]\n"
                    "D) [option D]\n"
                    "Answer: [answer]")]

prompt_answer_generator = '''Imagine you are a witness who has seen one person and someone is asking the information about that person. You are answering a question about that person and this is what you remember:
<|context|>

Answer the question only according to what you remember.
Instruction:
1. Answer directly in a complete sentence that provides the information to the question, without including any extra information.
2. Avoid giving answers that are just `Yes', `No', or a single word or phrase. Always answer in full sentences, even for multiple-choice questions.
3. If an entity or attribute is not included in your memory, assume the person does not possess it.
Question: <|question|>
Answer:'''

prompt_answer_visual_generator = '''<|question|>
Hint:
1. Avoid giving answers that are just `Yes', `No', or a single word or phrase. Always answer in full sentences, even for multiple-choice questions.
2. If the question assume something wrong about the person, directly give the corrective information of that aspect.
'''

SYSTEM_MESSAGE_REID = '''<|im_start|>system
You are an advanced question generator designed to assist in identifying a specific individual from a collection of candidates.'''

def get_image_placeholder(num_candidates: int):
    candidate_images_placeholder = f"[Candidate Person] {DEFAULT_IMAGE_TOKEN}\n[Similar Persons] "
    candidate_images_placeholder += ''.join([f"#{j + 1}: {DEFAULT_IMAGE_TOKEN}\n" for j in range(num_candidates - 1)])
    return candidate_images_placeholder.strip('\n')


def get_conversation(initial_query, questions, answers):
    conversation = f"[Conversation] Witness: {initial_query}\n"
    for (q, a) in zip(questions, answers):
        q = q.replace('\n', ' ')
        conversation += f"You: {q}\nWitness: {a}\n"
    return conversation


def wrap_question_prompt(prompt_base: str,
                         initial_query: List[str],
                         questions: List[List[str]],
                         answers: List[List[str]],
                         num_candidates: int,
                         batch=False):
    if isinstance(prompt_base, list):
        # print("random choosing prompt")
        prompt_base = random.choice(prompt_base)
    prompt_base = prompt_base.replace("<|candidate_images|>", get_image_placeholder(num_candidates))
    if batch:
        prompts = []
        assert len(initial_query) == len(questions) and len(answers) == len(questions)
        for i in range(len(initial_query)):
            prompt = prompt_base.replace("<|context|>", get_conversation(initial_query[i], questions[i], answers[i]))
            prompts.append(prompt)
        return prompts
    else:
        prompt = prompt_base.replace("<|context|>", get_conversation(initial_query, questions, answers))
        return prompt


prompt_question_generator_v3 = [("<|candidate_images|>\n<|context|>\n"
                                 "Based on these information:\n"
                                 "[Candidate Person] The closest match to the provided description and answers.\n"
                                 "[Similar Persons] Individuals who also match but are less similar.\n"
                                 "[Conversation] The initial description from the witness and the ongoing dialogue.\n"
                                 "generate a question that could help differentiate the candidate from the similar individuals.\n"
                                 "Avoid repeating questions that already answered with \"I don't know.\""),
                                ("<|candidate_images|>\n<|context|>\n"
                                 "Based on the following context:\n"
                                 "[Candidate Person] The individual most likely matching the description provided.\n"
                                 "[Similar Persons] Individuals with similar characteristics, but they are less likely to be the candidate.\n"
                                 "[Conversation] The dialogue between the witness and investigator, including all available descriptions.\n"
                                 "Generate a question that explores a specific detail or feature that differentiates the candidate from others. "
                                 "Ensure the question has not already been answered or ruled out with an \"I don't know\" response."
                                 ),
                                ("<|candidate_images|>\n<|context|>\n"
                                 "The following context applies:\n"
                                 "[Candidate Person] The individual who best fits the description so far.\n"
                                 "[Similar Persons] Individuals who have some overlapping characteristics but are less likely to be the target.\n"
                                 "[Conversation] The ongoing witness description and responses to questions.\n"
                                 "Generate a question that focuses on details that could clearly identify the candidate, "
                                 "while avoiding any redundant questions that have already been answered with \"I don't know.\"")
                                ]
