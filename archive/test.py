from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-5",
    input="What is the answer to the ultimate question of life, the universe, and everything?",
    text={
        "verbosity": "low"
    }
)

print(response)
