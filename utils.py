import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm
import json
import re, time
import matplotlib.pyplot as plt

def fetch_article_content(url):
    """
    한국일보 뉴스 url로 부터 기사 수집.

    사용 예시:
        tqdm.pandas()
        df['content'], df['len_context'] = zip(*df['link'].progress_apply(fetch_article_content))
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p', {'class': 'editor-p'})
        content = " ".join([p.get_text(strip=True) for p in paragraphs])
        return content, len(content) if content else ("Content not found", 0)
    except Exception as e:
        return f"Error fetching content: {e}", 0
    
def calculate_token_count(messages, api_key, api_gw_key):
    """
    Clova Token 계산기:

    사용 예시:
    messages = [
        {
            "role": "user",
            "content": "This is test"
        }
    ]
    calculate_token_count(messages, api_key, api_gw_key)
    """
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
    JSON 구조화 함수
    """
    try:
        parsed_content = json.loads(result_content)
    except json.JSONDecodeError:
        def fix_reason_field(s):
            pattern = r'("근거"\s*:\s*)(.*?)(\s*[\n\r]*\s*})'
            match = re.search(pattern, s, re.DOTALL)
            if match:
                key = match.group(1)  
                value = match.group(2).strip() 
                end = match.group(3)   

                if not (value.startswith('"') and value.endswith('"')):
                    value = value.replace('"', '\\"')
                    value = f'"{value}"'

                return s[:match.start(1)] + key + value + end
            else:
                return s

        fixed_content = fix_reason_field(result_content)

        try:
            parsed_content = json.loads(fixed_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 파싱 실패: {str(e)}")

    pred = parsed_content.get("분류", "")
    score = parsed_content.get("점수", "")
    reason = parsed_content.get("근거", "")

    return pred, score, reason

def process_dataframe(df, category, prompt, api_key, apigw_api_key):
    """
        배치 처리를 위한 데이터 처리 함수:

    파라미터:
        df: 처리할 데이터 프레임
        category: 분류 카테고리
        prompt: 프롬프트
        api_key: API 호출에 사용할 API 키
        api_gw_key: API Gateway 호출에 사용할 API 키

    리턴:
        result_df: 결과 데이터 프레임
        errors_df: 오류 로그 

    사용 예시:
        prompts = {
        "난이도": prompt_1,
        "논조": prompt_2,
        "논쟁성": prompt_3
        }

        final_result_df_list = []
        final_errors_df_list = []

        for category, prompt in prompts.items():
            print(f"Processing category: {category}")

            result_df, errors_df = process_dataframe(df, category, prompt, api_key, api_gw_key)

            # 각 카테고리별 결과를 리스트에 저장
            final_result_df_list.append(result_df)
            final_errors_df_list.append(errors_df)

        # 각 카테고리별 결과를 하나의 데이터프레임으로 병합
        final_result_df = pd.concat(final_result_df_list).reset_index(drop=True)
        final_errors_df = pd.concat(final_errors_df_list).reset_index(drop=True)

        # 최종 결과를 docid 순서로 정렬
        final_result_df = final_result_df.sort_values(by='docid').reset_index(drop=True)

        # 최종 결과 및 에러 데이터프레임 저장
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_result_filename = f'final_result_df_{current_time}.csv'
        final_errors_filename = f'final_errors_df_{current_time}.csv'
        final_result_df.to_csv(final_result_filename, index=False)
        final_errors_df.to_csv(final_errors_filename, index=False)

        print(f"All categories processed and results saved as {final_result_filename} and {final_errors_filename}.")
    """
    df_filtered = df[df['category'] == category]

    results = []
    errors = []

    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc=f"Processing {category} rows"):
        try:
            context = row['content']
            len_context = row['len_context']

            if len_context > 6000:
                context = context[:6000]

            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": context},
            ]

            response_json, error = call_clova_api(api_key, apigw_api_key, messages)

            if error:
                raise Exception(error)

            if 'result' not in response_json or 'message' not in response_json['result']:
                raise Exception("Unexpected response format: 'result' or 'message' key missing")

            result_content = response_json['result']['message'].get('content', '')

            try:
                pred, score, reason = process_response_content(result_content)
            except ValueError as ve:
                raise Exception(f"Failed to process content for docid {row['id']} in {category} - {str(ve)}")

            results.append({
                "docid": row['id'],
                "category": category,
                "title": row['title'],
                "link": row['web_link'],
                "content": row['content'],
                "len_content": len_context,
                "label": row['label'],
                "pred": pred,
                "score": score,
                "reason": reason
            })
            print(f"Success: docid {row['id']} in {category} 처리 완료")
            time.sleep(2)  

        except Exception as e:
            errors.append({
                "docid": row['id'],
                "category": category,
                "errors": str(e),
                "time": datetime.now().strftime('%Y%m%d %H:%M:%S')
            })

            print(f"Error: docid {row['id']} in {category} 처리 실패 - {str(e)}")

    result_df = pd.DataFrame(results)
    errors_df = pd.DataFrame(errors)

    return result_df, errors_df

def retry_failed_rows(errors_df, df, result_df, prompts, api_key, api_gw_key, max_retries=3):
    """
    데이터 재처리 함수:

    파라미터:
        errors_df: process_dataframe() 수행 후 에러 로그
        df: 처리할 데이터 프레임
        result_df: process_dataframe() 수행 결과
        prompts: 분류 카테고리별로 사용할 프롬프트를 담은 딕셔너리
        api_key: API 호출에 사용할 API 키
        api_gw_key: API Gateway 호출에 사용할 API 키
        max_retries: 재시도 최대 횟수

    리턴:
        result_df: category 별 재처리 결과
        errors_df: category 별 에러 로그
        logs_df: 전체 에러 로그

    사용 예시:

        #40005와 40006 오류를 제외한 모든 오류에 대해 재처리 수행
        retry_errors_df = final_errors_df[~final_errors_df['errors'].str.contains('40005|40006')]

        # 재처리 수행
        final_result_df, final_errors_df, retry_logs_df = retry_failed_rows(retry_errors_df, df, final_result_df, prompts, api_key, api_gw_key)

        # 재처리 후 결과와 로그 저장
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_result_filename = f'final_result_df_after_retry_{current_time}.csv'
        final_errors_filename = f'final_errors_df_after_retry_{current_time}.csv'
        retry_logs_filename = f'retry_logs_{current_time}.csv'

        final_result_df.to_csv(final_result_filename, index=False)
        final_errors_df.to_csv(final_errors_filename, index=False)
        retry_logs_df.to_csv(retry_logs_filename, index=False)

        print(f"Retry processing complete. Results saved as {final_result_filename}, {final_errors_filename}, and {retry_logs_filename}.")
    """
    retry_count = 0
    retry_wait_time = 2

    all_logs = []

    while not errors_df.empty and retry_count < max_retries:
        retry_count += 1
        print(f"Retrying failed rows, attempt {retry_count}")

        current_attempt_logs = []
        new_errors = []

        for _, error_row in tqdm(errors_df.iterrows(), total=len(errors_df), desc=f"Retrying attempt {retry_count}"):
            try:
                matching_rows = df[(df['id'] == error_row['docid']) & (df['category'] == error_row['category'])]

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

                if len_context > 6000:
                    context = context[:6000]

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

                pred, score, reason = process_response_content(result_content)
                result_data = {
                    "docid": row['id'],
                    "category": row['category'],
                    "title": row['title'],
                    "link": row['web_link'],
                    "content": row['content'],
                    "len_content": len_context,
                    "label": row['label'],
                    "pred": pred,
                    "score": score,
                    "reason": reason
                }
                original_index = result_df[result_df['docid'] < row['id']].index.max() + 1
                if pd.isna(original_index): 
                    original_index = 0

                result_df = pd.concat([
                    result_df.iloc[:original_index],  # 기존 데이터프레임에서 해당 위치까지
                    pd.DataFrame([result_data]),      # 새롭게 추가할 데이터
                    result_df.iloc[original_index:]   # 해당 위치 이후의 데이터
                ]).reset_index(drop=True)

                log_message = f"Success on retry: docid {row['id']} in {row['category']} 처리 완료"
                print(log_message)
                current_attempt_logs.append({
                    "docid": row['id'],
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

def process_single_row(row, prompt, api_key, apigw_api_key):
    try:
        context = row['content']
        len_context = row['len_context']

        if len_context > 6000:
            context = context[:6000]

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": context},
        ]

        response_json, error = call_clova_api(api_key, apigw_api_key, messages)

        if error:
            raise Exception(error)
        if 'result' not in response_json or 'message' not in response_json['result']:
            raise Exception("Unexpected response format: 'result' or 'message' key missing")

        result_content = response_json['result']['message'].get('content', '')
        # print(result_content)

        pred, score, reason = process_response_content(result_content)

        return pred, score, reason

    except Exception as e:
        print(f"Error: docid {row['id']} 처리 실패 - {str(e)}")