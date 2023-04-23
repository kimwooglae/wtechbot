import interactions
import json
import openai
import time

import pandas as pd
import numpy as np
from openai.embeddings_utils import distances_from_embeddings

import re

# CLEANR = re.compile('<.*?>')
CLEANR = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")


def cleanhtml(raw_html):
    cleantext = re.sub(CLEANR, "", raw_html)
    return cleantext


df_ko = pd.read_csv("processed/embeddings_ko.csv", index_col=0)
df_ko["embeddings"] = df_ko["embeddings"].apply(eval).apply(np.array)

print("korean embedding loaded")

df_ko_cleansing = pd.read_csv("processed/embeddings_cleaning_ko.csv", index_col=0)
df_ko_cleansing["embeddings"] = (
    df_ko_cleansing["embeddings"].apply(eval).apply(np.array)
)

print("korean cleaning embedding loaded")

df_guide_ko = pd.read_csv("processed/embeddings_guide_ko.csv", index_col=0)
df_guide_ko["embeddings"] = df_guide_ko["embeddings"].apply(eval).apply(np.array)

print("korean guide embedding loaded")

df_en = pd.read_csv("processed/embeddings_en.csv", index_col=0)
df_en["embeddings"] = df_en["embeddings"].apply(eval).apply(np.array)

print("english embedding loaded")

df_api_ko = pd.read_csv("processed/embeddings_api_ko.csv", index_col=0)
df_api_ko.drop(
    df_api_ko[df_api_ko["component"] == "WebSquare.uiplugin.grid"].index, inplace=True
)
df_api_ko["embeddings"] = df_api_ko["embeddings"].apply(eval).apply(np.array)

print("api embedding loaded")

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)
token = config["token"]
guildId = config["guildId"]

bot = interactions.Client(token=token)
model = "gpt-3.5-turbo"
stop_sequence = None


def create_context(question, df, max_len=1500, skip_cnt=0, debug=False):
    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(
        input=question, engine="text-embedding-ada-002"
    )["data"][0]["embedding"]

    # Get the distances from the embeddings
    df["distances"] = distances_from_embeddings(
        q_embeddings, df["embeddings"].values, distance_metric="cosine"
    )

    returns = []
    cur_len = 0
    prev_distance = 0
    prev_msg = ""
    skip_cnt_orig = skip_cnt

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values("distances", ascending=True).iterrows():
        if prev_distance == row["distances"]:
            # print("이전 목록과 동일 (distance)")
            continue
        elif prev_msg == row["text"]:
            # print("이전 목록과 동일 (문자열)")
            continue
        elif skip_cnt > 0:
            # print("skip_cnt > 0")
            skip_cnt -= 1
            continue
        else:
            prev_distance = row["distances"]
            prev_msg = row["text"]

            # Add the length of the text to the current length
            cur_len += row["n_tokens"] + 4

            # If the context is too long, break
            if cur_len > max_len:
                break

            if debug:
                print(i, row["distances"], row["text"])
            # Else add it to the text that is being returned
            returns.append(row["text"])

    return "\n\n---\n\n".join(returns), len(returns) + skip_cnt_orig, len(returns)


def create_guide_context(question, df, max_len=1500, skip_cnt=0, debug=False):
    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(
        input=question, engine="text-embedding-ada-002"
    )["data"][0]["embedding"]

    # Get the distances from the embeddings
    df["distances"] = distances_from_embeddings(
        q_embeddings, df["embeddings"].values, distance_metric="cosine"
    )

    returns = []
    links = []
    cur_len = 0
    prev_distance = 0
    prev_msg = ""
    skip_cnt_orig = skip_cnt

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values("distances", ascending=True).iterrows():
        if prev_distance == row["distances"]:
            # print("이전 목록과 동일 (distance)")
            continue
        elif prev_msg == row["text"]:
            # print("이전 목록과 동일 (문자열)")
            continue
        elif skip_cnt > 0:
            # print("skip_cnt > 0")
            skip_cnt -= 1
            continue
        else:
            prev_distance = row["distances"]
            prev_msg = row["text"]

            # Add the length of the text to the current length
            cur_len += row["n_tokens"] + 4

            # If the context is too long, break
            if cur_len > max_len:
                break

            if debug:
                print(i, row["distances"], row["text"])
            # Else add it to the text that is being returned
            returns.append(cleanhtml(row["text"]))
            links.append(row["link"])

    return returns, links, len(returns) + skip_cnt_orig


def search_context(df, question, max_cnt=5, debug=False):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(
        input=question, engine="text-embedding-ada-002"
    )["data"][0]["embedding"]

    # Get the distances from the embeddings
    df["distances"] = distances_from_embeddings(
        q_embeddings, df["embeddings"].values, distance_metric="cosine"
    )

    components = []
    names = []
    descriptions = []
    cur_cnt = 0
    prev_distance = 0
    prev_msg = ""

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values("distances", ascending=True).iterrows():
        if prev_distance == row["distances"]:
            # print("이전 목록과 동일 (distance)")
            continue
        elif prev_msg == row["text"]:
            # print("이전 목록과 동일 (문자열)")
            continue
        else:
            prev_distance = row["distances"]
            prev_msg = row["text"]

            # Add the length of the text to the current length
            cur_cnt += 1

            # If the context is too long, break
            if cur_cnt > max_cnt:
                break

            if debug:
                print(i, row["distances"], row["text"])

            components.append(row["component"])
            names.append(row["name"])
            descriptions.append(cleanhtml(row["description"]))

    return components, names, descriptions


def answer_question_chat(
    df,
    model="gpt-3.5-turbo",
    question="",
    max_len=3000,
    debug=False,
    stop_sequence=None,
    df2=None,
):
    gpt_retries = 3
    gpt_retry_cnt = 0
    skip_cnt = 0
    while gpt_retry_cnt <= gpt_retries:
        print("skip_cnt==>", skip_cnt)
        context, skip_cnt_next, context_len = create_context(
            question, df, max_len=max_len, skip_cnt=skip_cnt, debug=debug
        )

        context2, _, _ = create_context(
            question, df2, max_len=max_len, skip_cnt=skip_cnt, debug=debug
        )
        if debug:
            print("\n\nContext:\n" + context)
            print("\n\n")

        print("context==>", context)
        api_retries = 3
        api_retry_cnt = 0
        backoff_time = 10
        while api_retry_cnt <= api_retries:
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": f'Answer the question based on the context below, and if the question can\'t be answered based on the context, say "잘 모르겠습니다."\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer in Korean.',
                        }
                    ],
                    temperature=0,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=stop_sequence,
                )
                response_message = response["choices"][0]["message"]["content"].strip()
                print("response_message==>", response_message)
                print(response_message.find("잘 모르겠습니다."))
                if response_message.find("잘 모르겠습니다.") == -1:
                    return response_message, context, skip_cnt, context_len, context2
                else:
                    gpt_retry_cnt += 1
                    skip_cnt = skip_cnt_next
                    break
            except Exception as e:
                print(e)
                print(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time *= 1.5
                api_retry_cnt += 1

    return "잘 모르겠습니다.", context, skip_cnt, context_len, context2


@bot.command(
    name="api",
    description="API 검색용 W-Tech GPT 봇입니다!",
    scope=guildId,
    options=[
        interactions.Option(
            name="question",
            description="질문을 입력하세요",
            type=interactions.OptionType.STRING,
            required=True,
        ),
        interactions.Option(
            name="cnt",
            description="반환할 API 개수를 입력하세요. (기본값: 10))",
            type=interactions.OptionType.INTEGER,
            required=False,
        ),
    ],
)
async def api(ctx: interactions.CommandContext, question: str, cnt: int = 10):
    try:
        await ctx.defer()
        print("\nmodel==>", model)
        print("question==>", question)
        print("cnt==>", str(cnt))
        df = df_api_ko
        component_name, names, descriptions = search_context(
            df, question=question, max_cnt=cnt, debug=True
        )
        print("component_name==>", component_name)
        print("names==>", names)
        print("descriptions==>", descriptions)
        total_len = len(question[:4000])
        embeds_len = 1
        embeds = []
        embeds.append(interactions.Embed(title="질문", description=question[:4000]))
        for i in range(len(descriptions)):
            if embeds_len >= 10:
                break
            if total_len + len(descriptions[i][:4000]) > 5800:
                break

            embeds.append(
                interactions.Embed(
                    title=component_name[i] + " " + names[i],
                    description=descriptions[i][:4000],
                )
            )
            embeds_len += 1
            total_len += len(descriptions[i][:4000])

        components = []
        components.append(
            interactions.Button(
                label="API",
                style=interactions.ButtonStyle.LINK,
                url="https://docs.inswave.com/websquare/websquare.html?w2xPath=/support/api/ws5_sp5/api.xml",
            )
        )

        await ctx.send(embeds=embeds, components=components)

    except Exception as e:
        print(e)
        await ctx.send("에러가 발생했습니다.")


@bot.command(
    name="w",
    description="W-Tech GPT 봇입니다!",
    scope=guildId,
    options=[
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
            choices=[
                interactions.Choice(name="gpt-3.5-turbo", value="gpt-3.5-turbo"),
                interactions.Choice(name="gpt-4", value="gpt-4"),
            ],
            required=False,
        ),
        interactions.Option(
            name="data",
            description="어떤 데이터를 사용할지 선택하세요. (ko:한국어 원본 데이터, ko_cleaning:한국어 cleaning 데이터, en: 영어 데이터) (기본값: ko_cleaning)",
            type=interactions.OptionType.STRING,
            choices=[
                interactions.Choice(name="ko", value="ko"),
                interactions.Choice(name="ko_cleaning", value="ko_cleaning"),
                interactions.Choice(name="en", value="en"),
            ],
            required=False,
        ),
        interactions.Option(
            name="include_context",
            description="Context 정보를 반환할지여부. (기본값: N)",
            type=interactions.OptionType.STRING,
            choices=[
                interactions.Choice(name="N", value="N"),
                interactions.Choice(name="Y", value="Y"),
            ],
            required=False,
        ),
    ],
)
async def w(
    ctx: interactions.CommandContext,
    question: str,
    model: str = "gpt-3.5-turbo",
    data: str = "ko_cleansing",
    include_context: str = "N",
):
    try:
        await ctx.defer()
        print("\nmodel==>", model)
        print("data==>", data)
        print("question==>", question)
        print("include_context==>", include_context)
        data_type = "2"
        df = df_en
        if data == "ko":
            data_type = "0"
            df = df_ko
        elif data == "ko_cleansing":
            data_type = "1"
            df = df_ko_cleansing
        response, context, skip_cnt, context_len, context2 = answer_question_chat(
            df, question=question, model=model, debug=False, df2=df_ko
        )
        print("answer==>", response)
        total_len = len(response[:4000]) + len(question[:4000])
        embeds_len = 2
        embeds = []
        embeds.append(interactions.Embed(title="질문", description=question[:4000]))
        embeds.append(
            interactions.Embed(
                title="W-Tech 답변",
                description=response[:4000],
                footer=interactions.EmbedFooter(
                    text="powered by gpt4\tc"
                    + str(context_len)
                    + "."
                    + str(skip_cnt)
                    + "."
                    + data_type
                    + ""
                ),
            )
        )
        if include_context == "Y":
            contextList = context.split("\n\n---\n\n")
            for i in range(len(contextList)):
                if embeds_len >= 10:
                    break
                if total_len + len(contextList[i][:4000]) > 5800:
                    break

                embeds.append(
                    interactions.Embed(
                        title="관련정보 - " + str((i + 1)),
                        description=contextList[i][:4000],
                    )
                )
                embeds_len += 1
                total_len += len(contextList[i][:4000])

        components = []
        if include_context == "N":
            components.append(
                interactions.Button(
                    label="관련정보",
                    custom_id="wtech_gpt_context",
                    style=interactions.ButtonStyle.PRIMARY,
                )
            )

        components.append(
            interactions.Button(
                label="동영상",
                custom_id="wtech_gpt_youtube",
                style=interactions.ButtonStyle.PRIMARY,
            )
        )

        components.append(
            interactions.Button(
                label="가이드",
                custom_id="wtech_gpt_guide",
                style=interactions.ButtonStyle.PRIMARY,
            )
        )

        components.append(
            interactions.Button(
                label="개발 가이드",
                style=interactions.ButtonStyle.LINK,
                url="https://docs1.inswave.com/sp5_user_guide",
            )
        )
        await ctx.send(embeds=embeds, components=components)

    except Exception as e:
        print(e)
        await ctx.send("에러가 발생했습니다.")


@bot.component("wtech_gpt_context")
async def button_response_detail(ctx):
    await ctx.defer()
    print(ctx)
    question = ctx.message.embeds[0].description
    skip_cnt = int(ctx.message.embeds[1].footer.text.split("\t")[1].split(".")[1])
    data_type = int(ctx.message.embeds[1].footer.text.split("\t")[1].split(".")[2])
    print("question==>", question)
    print("skip_cnt==>", skip_cnt)
    df = df_ko
    if data_type == 0:
        df = df_ko
    elif data_type == 1:
        df = df_ko_cleansing
    else:
        df = df_en

    context, _, _ = create_context(
        question, df, max_len=3000, skip_cnt=skip_cnt, debug=False
    )
    print("context==>", context)
    total_len = 0
    embeds_len = 0
    embeds = []
    contextList = context.split("\n\n---\n\n")
    for i in range(len(contextList)):
        if embeds_len >= 10:
            break
        if total_len + len(contextList[i][:4000]) > 5800:
            break

        embeds.append(
            interactions.Embed(
                title="관련정보 - " + str((i + 1)), description=contextList[i][:4000]
            )
        )
        embeds_len += 1
        total_len += len(contextList[i][:4000])

    components = []

    components.append(
        interactions.Button(
            label="개발 가이드",
            style=interactions.ButtonStyle.LINK,
            url="https://docs1.inswave.com/sp5_user_guide",
        )
    )

    await ctx.send(embeds=embeds, components=components)


@bot.component("wtech_gpt_guide")
async def button_guide_detail(ctx):
    await ctx.defer()
    print(ctx)
    question = ctx.message.embeds[0].description
    skip_cnt = 0
    # skip_cnt = int(ctx.message.embeds[1].footer.text.split("\t")[1].split(".")[1])
    # data_type = int(ctx.message.embeds[1].footer.text.split("\t")[1].split(".")[2])
    print("question==>", question)
    print("skip_cnt==>", skip_cnt)
    df = df_guide_ko

    contextList, linkList, _ = create_guide_context(
        question, df, max_len=3000, skip_cnt=skip_cnt, debug=False
    )

    total_len = 0
    embeds_len = 0
    embeds = []
    for i in range(len(contextList)):
        if embeds_len >= 10:
            break
        if total_len + len(contextList[i][:4000]) > 5800:
            break

        embeds.append(
            interactions.Embed(
                title="가이드 - " + str((i + 1)),
                url="https://docs1.inswave.com/sp5_user_guide/" + linkList[i],
                description=contextList[i][:4000],
            )
        )
        embeds_len += 1
        total_len += len(contextList[i][:4000])

    components = []

    components.append(
        interactions.Button(
            label="개발 가이드",
            style=interactions.ButtonStyle.LINK,
            url="https://docs1.inswave.com/sp5_user_guide",
        )
    )

    await ctx.send(embeds=embeds, components=components)


@bot.component("wtech_gpt_youtube")
async def button_youtube_response(ctx):
    await ctx.defer()
    print(ctx)
    question = ctx.message.embeds[0].description
    skip_cnt = 0
    # skip_cnt = int(ctx.message.embeds[1].footer.text.split("\t")[1].split(".")[1])
    print("question==>", question)
    print("skip_cnt==>", skip_cnt)
    df = df_ko
    context, _, _ = create_context(
        question, df, max_len=6000, skip_cnt=skip_cnt, debug=False
    )
    print("context==>", context)
    urls = ""
    youtube_cnt = 0
    contextList = context.split("\n\n---\n\n")
    for i in range(len(contextList)):
        if contextList[i].find("https://youtu.be") != -1:
            chunks = contextList[i].split("https://youtu.be")
            for idx in range(len(chunks)):
                if idx > 0 and urls.find(chunks[idx].split(")")[0]) == -1:
                    youtube_cnt += 1
                    urls += "https://youtu.be" + chunks[idx].split(")")[0] + " "
                    if youtube_cnt >= 5:
                        break
        if youtube_cnt >= 5:
            break

    components = []
    components.append(
        interactions.Button(
            label="개발 가이드",
            style=interactions.ButtonStyle.LINK,
            url="https://docs1.inswave.com/sp5_user_guide",
        )
    )

    await ctx.send(content=urls, components=components)


bot.start()
