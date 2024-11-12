import json
from openai import OpenAI, RateLimitError
import backoff
from tqdm import tqdm

client = OpenAI()


@backoff.on_exception(backoff.expo, RateLimitError, max_time=60)
def translate_text(text):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You will be given a text in English delimited by triple quotes. Translate the text into Mongolian. Do not output the triple quotes.",
            },
            {
                "role": "user",
                "content": f'"""{text}"""',
            },
        ],
    )
    return response.choices[0].message.content.strip()


def translate_instructions(input_file, output_file):
    data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    for item in tqdm(data):
        if "turns" in item:
            item["turns"][0] = translate_text(item["turns"][0])
            item["turns"][1] = translate_text(item["turns"][1])
        if "reference" in item and item["reference"] != "":
            item["reference"] = translate_text(
                item["reference"]
            )

    with open(output_file, "w", encoding="utf-8") as f:
        for d in data:
            json_str = json.dumps(d, ensure_ascii=False)
            f.write(json_str)
            f.write("\n")


if __name__ == "__main__":
    input_file = "question copy.jsonl"
    output_file = "question_translated.jsonl"
    translate_instructions(input_file, output_file)
