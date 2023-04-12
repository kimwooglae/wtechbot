import interactions
import json
import openai
import os
import time

import pandas as pd
import numpy as np
from openai.embeddings_utils import distances_from_embeddings

df_ko=pd.read_csv('processed/embeddings_ko.csv', index_col=0)
df_ko['embeddings'] = df_ko['embeddings'].apply(eval).apply(np.array)

print('korean embedding loaded')

df_en=pd.read_csv('processed/embeddings_en.csv', index_col=0)
df_en['embeddings'] = df_en['embeddings'].apply(eval).apply(np.array)

print('english embedding loaded')


# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)
token = config['token']
guildId = config['guildId']

bot = interactions.Client(token=token)
model="gpt-3.5-turbo"
stop_sequence=None

def create_context(
    question, df, max_len=1500, debug=False
):
    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0
    prev_distance = 0
    prev_msg = ""

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        if prev_distance == row['distances']:
            # print("이전 목록과 동일 (distance)")
            continue
        elif prev_msg == row['text']:
            # print("이전 목록과 동일 (문자열)")
            continue
        else:
            prev_distance = row['distances']
            prev_msg = row['text']            

            # Add the length of the text to the current length
            cur_len += row['n_tokens'] + 4

            # If the context is too long, break
            if cur_len > max_len:
                break

            if debug:
                print(i, row['distances'], row['text'])
            # Else add it to the text that is being returned
            returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def answer_question_chat(
    df,
    model="gpt-3.5-turbo",
    question="",
    max_len=3000,
    debug=False,
    stop_sequence=None
):

    context = create_context(
        question,
        df,
        max_len=max_len,
        debug=debug
    )
    if debug:
        print("\n\nContext:\n" + context)
        print("\n\n")

    print("context==>", context)
    retries = 3
    retry_cnt = 0
    backoff_time = 10
    while retry_cnt <= retries:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "user", "content": f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"잘 모르겠습니다.\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer in Korean."}
                ],
                temperature=0,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop_sequence,
            )

            return response["choices"][0]["message"]["content"].strip(), context
        except Exception as e:
            print(e)
            print(f"Retrying in {backoff_time} seconds...")
            time.sleep(backoff_time)
            backoff_time *= 1.5
            retry_cnt += 1
        return "잘 모르겠습니다.", context

@bot.command(
    name="wtech",
    description="wtech q&a bot!",
    scope=guildId,
    options = [
        interactions.Option(
            name="question",
            description="질문을 입력하세요",
            type=interactions.OptionType.STRING,
            required=True,
        ),
        interactions.Option(
            name="model",
            description="model을 선택하세요.(기본값: gpt-3.5-turbo))",
            type=interactions.OptionType.STRING,
            choices=[interactions.Choice(name="gpt-3.5-turbo", value="gpt-3.5-turbo"), interactions.Choice(name="gpt-4", value="gpt-4")],
            required=False,
        ),
        interactions.Option(
            name="lang",
            description="한국어 데이터를 사용할지, 영어 데이터를 사용할지 선택하세요. (기본값: ko)",
            type=interactions.OptionType.STRING,
            choices=[interactions.Choice(name="ko", value="ko"), interactions.Choice(name="en", value="en")],
            required=False,
        ),
    ]
)
async def wtech(ctx: interactions.CommandContext, question: str, model:str='gpt-3.5-turbo', lang:str='ko'):
    try:
        await ctx.defer()
        print("\nmodel==>", model)
        print("lang==>", lang)
        print("question==>", question)
        df = df_en
        if lang == 'ko':
            df = df_ko
        response, context = answer_question_chat(df, question=question, model=model, debug=False)
        print("answer==>", response)
        await ctx.send(("# W-Tech GPT 답변\n" + response + "\n\n# 관련 정보\n" + context)[:2000])
    except Exception as e:
        print(e)
        await ctx.send("에러가 발생했습니다.")

bot.start()