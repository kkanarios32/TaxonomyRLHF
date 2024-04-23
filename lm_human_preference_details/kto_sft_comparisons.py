import json
from openai import OpenAI
client = OpenAI()
from tqdm import tqdm

system_mes = "You are mimicking a human rater who compares two summaries of a Reddit post provided to you, and chooses the better one as commanded by the user."

user_mes_beginning = """"Which of the following summaries does a better job of summarizing the most \
important points in the given forum post, without including unimportant or \
irrelevant details? A good summary is both precise and concise. A concise but inaccurate summary is unhelpful. \
Both summaries will typically be somewhat hard to parse and have many irrelevant or made-up statements. Choose the summary that is\
less full of gibberish and more relevant to the text.\n"""

user_mes_end = """FIRST provide a one-sentence comparison of the two summaries, explaining which \
you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your \
choice. Your response should use the format:
Comparison: <one-sentence comparison and explanation>
Preferred: <"A" or "B">"""

with (open("kto.jsonl", "r") as kto_file):
    with (open("sft.jsonl", "r") as sft_file):
        with open("kto_sft_comparisons.txt", "a") as kto_sft_comparisons:
            kto_data = list(kto_file)
            sft_data = list(sft_file)
            for i in tqdm(range(1000)):
                kto_json = json.loads(kto_data[i])
                sft_json = json.loads(sft_data[i])
                user_mes = user_mes_beginning + kto_json["query"] + "\nSummary A: " + kto_json["response"] +\
                "\nSummary B: " + sft_json["response"] + "\n" + user_mes_end
                completion = client.chat.completions.create(model="gpt-4-turbo", 
                                                            messages=[{"role": "system", "content": system_mes},
                                                                      {"role": "user", "content": user_mes}]
                                                           )
                ans = completion.choices[0].message.content
                idx = ans.find("\nPreferred: ")
                kto_sft_comparisons.write(ans[idx+12:])