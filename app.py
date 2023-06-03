import os
import threading
import json
import requests
import openai
import time

import pandas as pd
import numpy as np

from openai.embeddings_utils import distances_from_embeddings

from flask import Flask, request, send_from_directory
from flask_cors import CORS
import re

# CLEANR = re.compile('<.*?>')
CLEANR = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")


def cleanhtml(raw_html):
    cleantext = re.sub(CLEANR, "", raw_html)
    return cleantext


app = Flask(__name__)
CORS(app, origins=[f"https://41ec6e2ada58.ngrok.app", "https://chat.openai.com"])

df_ko = pd.read_csv("processed/embeddings_ko.csv", index_col=0)
df_ko["embeddings"] = df_ko["embeddings"].apply(eval).apply(np.array)

print("korean embedding loaded")

# df_ko_cleansing = pd.read_csv("processed/embeddings_cleaning_ko.csv", index_col=0)
# df_ko_cleansing["embeddings"] = (
#     df_ko_cleansing["embeddings"].apply(eval).apply(np.array)
# )

# print("korean cleaning embedding loaded")

# df_en = pd.read_csv("processed/embeddings_en.csv", index_col=0)
# df_en["embeddings"] = df_en["embeddings"].apply(eval).apply(np.array)

# print("english embedding loaded")

df_api_ko = pd.read_csv("processed/embeddings_api_ko.csv", index_col=0)
df_api_ko.drop(
    df_api_ko[df_api_ko["component"] == "WebSquare.uiplugin.grid"].index, inplace=True
)
df_api_ko["embeddings"] = df_api_ko["embeddings"].apply(eval).apply(np.array)

print("api embedding loaded")


model = "gpt-4"
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


@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route("/logo.png", methods=["GET"])
def plugin_logo():
    return send_from_directory(os.path.dirname(__file__), "img/logo.png")


@app.route("/.well-known/ai-plugin.json", methods=["GET"])
def plugin_manifest():
    return send_from_directory(os.path.dirname(__file__), "ai-plugin.json")


@app.route("/openapi.yaml", methods=["GET"])
def openapi_spec():
    return send_from_directory(os.path.dirname(__file__), "openapi.yaml")


@app.route("/query", methods=["POST"])
def openapi_query():
    responseBody = {"results": ""}
    try:
        body = request.get_json()
        print(body)
        print(body["query"])
        question = body["query"]
        max_len = 4000
        print("question==>", question)
        print("max_len==>", max_len)
        df = df_ko
        context, _, _ = create_context(
            question, df, max_len=max_len, skip_cnt=0, debug=False
        )
        print("context==>", context)

        responseBody["results"] = context

    except Exception as e:
        print(e)
        responseBody["results"] = "에러가 발생했습니다."
    return responseBody


@app.route("/api/sayHello", methods=["POST"])
def sayHello():
    body = request.get_json()
    print(body)
    print(body["userRequest"]["utterance"])

    responseBody = {
        "version": "2.0",
        "template": {"outputs": [{"simpleText": {"text": "Hello World!"}}]},
    }
    return responseBody


@app.route("/api/api", methods=["POST"])
def api():
    responseBody = {
        "version": "2.0",
        "template": {"outputs": [{"simpleText": {"text": ""}}]},
    }
    try:
        body = request.get_json()
        print(body)
        print(body["userRequest"]["utterance"])
        question = body["action"]["detailParams"]["question"]["value"]
        cnt = 10
        print("\nmodel==>", model)
        print("question==>", question)
        print("cnt==>", cnt)
        df = df_api_ko
        component_name, names, descriptions = search_context(
            df, question=question, max_cnt=cnt, debug=True
        )
        print("component_name==>", component_name)
        print("names==>", names)
        print("descriptions==>", descriptions)

        total_len = 0
        msg = ""
        for i in range(len(descriptions)):
            if (
                total_len
                + (
                    3
                    + len(component_name[i].rsplit(",", 1)[-1])
                    + len(names[i])
                    + len(descriptions[i][:100])
                )
                > 1000
            ):
                break

            if i > 0:
                msg += "\n\n"

            info = (
                (descriptions[i][:97] + "..")
                if len(descriptions[i]) > 97
                else descriptions[i]
            )
            msg = (
                msg
                # + str(int(i + 1))
                # + "\ufe0f\u20e3 "
                # + "■ "
                + "["
                + component_name[i].rsplit(".", 1)[-1]
                + "."
                + names[i]
                + "]\n"
                + info
            )
            total_len += (
                6
                + len(component_name[i].rsplit(".", 1)[-1])
                + len(names[i])
                + len(info)
            )

        responseBody["template"]["outputs"][0]["simpleText"]["text"] = msg

    except Exception as e:
        print(e)
        responseBody["template"]["outputs"][0]["simpleText"]["text"] = "에러가 발생했습니다."
    return responseBody


@app.route("/api/w", methods=["POST"])
def w():
    responseBody = {
        "version": "2.0",
        "useCallback": True,
    }
    try:
        body = request.get_json()
        timer = threading.Timer(1.0, w_callback, args=(body,))
        timer.start()
    except Exception as e:
        print(e)
    return responseBody


def w_callback(body):
    requestBody = {
        "version": "2.0",
        "template": {"outputs": [{"simpleText": {"text": ""}}]},
    }
    try:
        print(body)
        print(body["userRequest"]["utterance"])
        question = body["action"]["detailParams"]["question"]["value"]
        callbackUrl = body["userRequest"]["callbackUrl"]
        data = "ko"
        max_len = 6000
        include_context = "N"

        print("\nmodel==>", model)
        print("data==>", data)
        print("question==>", question)
        print("include_context==>", include_context)
        df = df_ko
        # data_type = "0"
        # if data == "ko":
        #     data_type = "0"
        #     df = df_ko
        # elif data == "ko_cleansing":
        #     data_type = "1"
        #     df = df_ko_cleansing
        # else:
        #     data_type = "2"
        #     df = df_en

        response, context, skip_cnt, context_len, context2 = answer_question_chat(
            df, question=question, model=model, debug=False, df2=df_ko, max_len=max_len
        )
        print("answer==>", response)
        msg = (response[:997] + "..") if len(response) > 997 else response
        requestBody["template"]["outputs"][0]["simpleText"]["text"] = msg
        headers = {"Content-Type": "application/json; charset=utf-8"}
        print("callbackUrl==>", callbackUrl)
        print("headers==>", headers)
        print("requestBody==>", requestBody)
        res = requests.post(callbackUrl, data=json.dumps(requestBody), headers=headers)
        print(res)

        print("\n\n\n")
    except Exception as e:
        print(e)
