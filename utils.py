import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm
import json
import re, time
import matplotlib.pyplot as plt

def calculate_token_count(messages, api_key, api_gw_key):
    url = f'https://clovastudio.apigw.ntruss.com/v1/api-tools/chat-tokenize/HCX-003'

    headers = {
        'X-NCP-CLOVASTUDIO-API-KEY': api_key,
        'X-NCP-APIGW-API-KEY': api_gw_key,
        'Content-Type': 'application/json'
    }

    data = {
        "messages": messages
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        response_data = response.json()
        if response_data['status']['code'] == '20000':
            total_token_count = 0
            for message in response_data['result']['messages']:
                total_token_count += message['count']
            return total_token_count
        else:
            return None
    else:
        return None
    
def call_clova_api(api_key, api_gw_key, messages):
    url = 'https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-003'

    headers = {
        'X-NCP-CLOVASTUDIO-API-KEY': api_key,
        'X-NCP-APIGW-API-KEY': api_gw_key,
        'Content-Type': 'application/json',
    }

    data = {
        "topK": 0,
        "includeAiFilters": True,
        "maxTokens": 100,
        "temperature": 0.25,
        "messages": messages,
        "repeatPenalty": 4,
        "topP": 0.8
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code != 200:
            return None, f"API 호출 실패: {response.status_code}, {response.text}"

        if not response.text:
            return None, "Empty response received"

        try:
            response_json = response.json()

        except json.JSONDecodeError as e:
            return None, f"JSON decoding error: {e} - Response text: {response.text[:100]}"

        if isinstance(response_json, dict):
            return response_json, None
        else:
            return None, "Unexpected response format"

    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {e}"
    
def process_response_content(result_content):
    """ 
    """
    try:
        parsed_content = json.loads(result_content)
        pred = parsed_content.get("분류", "")
        prob = parsed_content.get("확률", "")
        return pred, prob
    except json.JSONDecodeError:
        raise ValueError("Unexpected content format: Not a valid JSON")

def process_dataframe(df, category, prompt, api_key, api_gw_key):
    df_filtered = df[df['category'] == category]

    results = []
    errors = []

    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc=f"Processing {category} rows"):
        try:
            context = row['content']
            len_context = row['len_context']

            if len_context > 7000:
                context = context[-4500:]

            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": context},
            ]

            response_json, error = call_clova_api(api_key, api_gw_key, messages)

            if error:
                raise Exception(error)

            if 'result' not in response_json or 'message' not in response_json['result']:
                raise Exception("Unexpected response format: 'result' or 'message' key missing")

            result_content = response_json['result']['message'].get('content', '')

            try:
                pred, prob = process_response_content(result_content)
            except ValueError as ve:
                raise Exception(f"Failed to process content for docid {row['docid']} in {category} - {str(ve)}")

            results.append({
                "docid": row['docid'],
                "category": category,
                "title": row['title'],
                "link": row['link'],
                "content": row['content'],
                "len_content": len_context,
                "label": row['label'],
                "pred": pred,
                "prob": prob
            })

            print(f"Success: docid {row['docid']} in {category} 처리 완료")
            time.sleep(5)  # 각 요청 간 지연 시간

        except Exception as e:
            errors.append({
                "docid": row['docid'],
                "category": category,
                "errors": str(e),
                "time": datetime.now().strftime('%Y%m%d %H:%M:%S')
            })

            print(f"Error: docid {row['docid']} in {category} 처리 실패 - {str(e)}")

    result_df = pd.DataFrame(results)
    errors_df = pd.DataFrame(errors)

    return result_df, errors_df

def retry_failed_rows(errors_df, df, result_df, prompts, api_key, api_gw_key, max_retries=3):
    retry_count = 0
    retry_wait_time = 6

    all_logs = []

    while not errors_df.empty and retry_count < max_retries:
        retry_count += 1
        print(f"Retrying failed rows, attempt {retry_count}")

        current_attempt_logs = []
        new_errors = []

        for _, error_row in tqdm(errors_df.iterrows(), total=len(errors_df), desc=f"Retrying attempt {retry_count}"):
            try:
                matching_rows = df[(df['docid'] == error_row['docid']) & (df['category'] == error_row['category'])]

                if matching_rows.empty:
                    log_message = f"No matching row found for docid {error_row['docid']} and category {error_row['category']}"
                    print(log_message)
                    current_attempt_logs.append({
                        "docid": error_row['docid'],
                        "category": error_row['category'],
                        "status": "Error",
                        "error_stage": "Retry",
                        "message": log_message,
                        "time": datetime.now().strftime('%Y%m%d %H:%M:%S')
                    })
                    new_errors.append({
                        "docid": error_row['docid'],
                        "category": error_row['category'],
                        "errors": log_message,
                        "error_stage": "Retry",
                        "time": datetime.now().strftime('%Y%m%d %H:%M:%S')
                    })
                    continue

                row = matching_rows.iloc[0]
                context = row['content']
                len_context = row['len_context']

                if len_context > 7000:
                    context = context[-4500:0]

                messages = [
                    {"role": "system", "content": prompts[row['category']]},
                    {"role": "user", "content": context},
                ]

                response_json, error = call_clova_api(api_key, api_gw_key, messages)

                if error:
                    if '429' in error:
                        log_message = f"429 Too Many Requests: Waiting for {retry_wait_time} seconds before retrying..."
                        print(log_message)
                        time.sleep(retry_wait_time)
                        raise Exception("API 호출 실패: 429, Too Many Requests")
                    elif '40005' in error or '40006' in error:
                        log_message = f"Skipping retry for docid {error_row['docid']} in {error_row['category']} due to policy issue: {error}"
                        print(log_message)
                        current_attempt_logs.append({
                            "docid": error_row['docid'],
                            "category": error_row['category'],
                            "status": "Error",
                            "error_stage": "Retry",
                            "message": log_message,
                            "time": datetime.now().strftime('%Y%m%d %H:%M:%S')
                        })
                        new_errors.append({
                            "docid": error_row['docid'],
                            "category": error_row['category'],
                            "errors": error,
                            "error_stage": "Retry",
                            "time": datetime.now().strftime('%Y%m%d %H:%M:%S')
                        })
                        continue
                    else:
                        raise Exception(error)

                if not response_json:
                    raise Exception("Received empty response from the API")

                if 'result' not in response_json or 'message' not in response_json['result']:
                    raise Exception("Unexpected response format: 'result' or 'message' key missing")

                result_content = response_json['result']['message'].get('content', '')

                if not isinstance(result_content, str):
                    raise Exception("Unexpected content format")

                pred, prob = process_response_content(result_content)

                result_data = {
                    "docid": row['docid'],
                    "category": row['category'],
                    "title": row['title'],
                    "link": row['link'],
                    "content": row['content'],
                    "len_content": len_context,
                    "label": row['label'],
                    "pred": pred,
                    "prob": prob
                }

                original_index = result_df[result_df['docid'] < row['docid']].index.max() + 1
                if pd.isna(original_index):
                    original_index = 0

                result_df = pd.concat([
                    result_df.iloc[:original_index],  # 기존 데이터프레임에서 해당 위치까지
                    pd.DataFrame([result_data]),      # 새롭게 추가할 데이터
                    result_df.iloc[original_index:]   # 해당 위치 이후의 데이터
                ]).reset_index(drop=True)

                log_message = f"Success on retry: docid {row['docid']} in {row['category']} 처리 완료"
                print(log_message)
                current_attempt_logs.append({
                    "docid": row['docid'],
                    "category": row['category'],
                    "status": "Success",
                    "error_stage": "Retry",
                    "message": log_message,
                    "time": datetime.now().strftime('%Y%m%d %H:%M:%S')
                })

            except Exception as e:
                log_message = f"Error on retry: docid {error_row['docid']} in {error_row['category']} 처리 실패 - {str(e)}"
                new_errors.append({
                    "docid": error_row['docid'],
                    "category": error_row['category'],
                    "errors": str(e),
                    "error_stage": "Retry",
                    "time": datetime.now().strftime('%Y%m%d %H:%M:%S')
                })
                print(log_message)
                current_attempt_logs.append({
                    "docid": error_row['docid'],
                    "category": error_row['category'],
                    "status": "Error",
                    "error_stage": "Retry",
                    "message": log_message,
                    "time": datetime.now().strftime('%Y%m%d %H:%M:%S')
                })

            time.sleep(retry_wait_time)

        all_logs.extend(current_attempt_logs)

        errors_df = pd.DataFrame(new_errors).reset_index(drop=True)

    logs_df = pd.DataFrame(all_logs)

    return result_df, errors_df, logs_df