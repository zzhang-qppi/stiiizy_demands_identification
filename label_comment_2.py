import pandas as pd
import numpy as np
import openai
from retry import retry
import time
import os
import re
import logging
from datetime import datetime
import tiktoken


# 获取回答
@retry(exceptions=Exception, tries=10, delay=5)  # 若出错，重试20次，每次暂停5秒
def get_response_from_gpt(m_comments, m_criteria):
    # comment: a string of a single comment
    # questions: a list of questions about the comment to be fed to GPT

    # sys_message = f'''You evaluate this comment "{m_comment}" against a list of questions.
    # It is very important that you answer the questions purely based on the content of the comment.
    # Don't make any interpretation beyond the exact words in the comment. Answer in yes and no only.
    # '''

    # sys_message = f'''You evaluate this consumer's comment about a e-cigarette product against a list of
    # {len(m_questions)} criteria. The comment is "{m_comment}". You take each criterion from the list
    # and answer a question that take a form like this: Does this comment mention [the criterion]?
    # I need {len(m_questions)} answers in a list indexed by numbers in the exact same order as the criteria list
    # and separated by one new line character.
    # You should give me one and only one answer for each criterion! Give me the answers in yes/no only.
    # Don't give me the questions.'''

    prompt = prompt_formulator(m_comments, m_criteria)

    print(prompt)
    return openai.ChatCompletion.create(
        messages=[
            {"role": "system",
             "content": "You evaluate comments on an e-cigarette product against a list of criteria I provide you."},
            {"role": "user", "content": prompt},
        ],
        model="gpt-4",
        temperature=0,
        request_timeout=300,
    )


def prompt_formulator(m_comments, m_criteria):
    if type(m_comments) is str:
        indexed_comment_string = m_comments
    else:
        indexed_comment_string = "comment's index | comment\n----|----------\n" + "\n".join(
            [
                f"{num_to_letter_index(i)} | '" + m_comments[i] + "'"
                for i in m_comments.index
            ]
        )
    indexed_criteria_string = "criterion's index | criterion\n----|------\n" + "\n".join(
        [f"{i} | " + m_criteria[i] for i in m_criteria.index]
    )
    # prompt = f'''Here is a buyer's comment of e-cigarette: "{indexed_comment_string}".
    # Which ones of the following criteria does this comment mention? Answer simply with the numerical indices.
    # \n'''
    prompt = f"""Here is a list of buyer's comments on an e-cigarette product:\n"{indexed_comment_string}" 
    
    In the exact order, take one comment at a time and answer:
    Which ones of the following criteria does this comment mention? 
    Answer simply with the numerical indices!
    If the comment doesn't mention any of the criteria, return an empty list for that answer.
    Your response should consist of lines in the format as such:
    [a comment's index] : [a list of indices of the mentioned criteria by the comment]
    
    \n{indexed_criteria_string}"""
    return prompt

def num_to_letter_index(num: int):
    # number to letter, e.g. 29---‘ad’

    quotient = num // 26
    remainder = num % 26
    if quotient == 0:
        return chr(97 + num)
    else:
        return chr(97 + quotient - 1) + chr(97 + remainder)


def letter_index_to_num(ind: str):
    # letter to number

    num = -1
    rev_ind = list(reversed(ind))
    for i in range(len(ind)):
        num += (26 ** i) * (ord(rev_ind[i]) - 96)
    return num


def insert_response_into_target_matrix(response, outmatrix, comment_indices):
    # Intelligently integrate GPT response into a target matrix
    num_of_comments = len(comment_indices)
    list_of_resp = (
        (response["choices"][0]["message"]["content"])
        .strip("\n")
        .split("\n", num_of_comments - 1)  # Restrict the number of lines to at most the number of comments,
        # so that it is easier to copy into the matrix later
    )
    # There should be a same number of lines of response and comments
    if len(list_of_resp) == num_of_comments:
        print("correct number of responses")
    else:
        print("wrong number of responses")
        print(len(list_of_resp), " responses")
        print(num_of_comments, " comments")
    for resp in list_of_resp:
        comment_index_split = resp.split(":")[0]
        criteria_index_split = resp.split(":")[1]
        comment_index = letter_index_to_num(re.findall("[a-z]+", comment_index_split)[0])
        mentioned_criteria_indices = np.array(re.findall("\d+", criteria_index_split), dtype="i")
        outmatrix.loc[comment_index, :] = 0
        outmatrix.loc[comment_index, mentioned_criteria_indices] = 1
    return outmatrix.copy()


def divide_into_batches(data, size_of_batch):
    # divide the Series of data into several smaller Series
    # to comply with the token limit or runtime efficiency
    # data: the source data that needs to be divided
    # length_of_batch: the number of items in each patch
    if type(data) is pd.io.parsers.readers.TextFileReader:
        while True:
            next_batch = -1
            try:
                next_batch = data.get_chunk(3)
                while tokeniser.encode(next_batch) <= 7200:
                    next_batch = pd.concat((next_batch, data.get_chunk(3)))
            except StopIteration:
                print("batching completed")
            finally:
                yield next_batch

    else:
        if 0 < size_of_batch <= len(data):
            batch_list = []
            # number of batches (which would very possibly be label_numb + 1)
            label_numb = len(data) // size_of_batch
            for i in range(label_numb):
                batch_list.append(
                    data[
                    len(data) // label_numb * i: len(data) // label_numb * (i + 1)
                    ]
                )
            if len(data) // label_numb * label_numb < len(data):
                batch_list.append(data[len(data) // label_numb * label_numb:])
            print("successfully batched criteria into ", len(batch_list), " batches")
        else:
            batch_list = [data]
            print("successfully read criteria. You opted not to batch criteria.")
        return batch_list


def comment_labeling_with_gpt(batched_comments, batched_criteria, criteria_batch_size, comments_batch_size):
    # a generator function that yields results by batch
    # comments: a Series of comments.
    # criteria: a Series of criteria

    # 初始回答汇总列表, concatenate later
    all_answers = []

    counter = 1
    for c in range(len(batched_comments)):
        this_comments = batched_comments[c]
        # initialize the output matrix with np.nan
        for j in range(len(batched_criteria)):
            # 将每个问题上传给ChatGPT并获取回答
            this_criteria = batched_criteria[j]
            answers = pd.DataFrame(columns=this_criteria.index, index=this_comments.index)
            try:
                response = get_response_from_gpt(this_comments, this_criteria)
                print(response)
                print("successfully responded")
                insert_response_into_target_matrix(response, answers, this_comments.index)
                # comments that are missed by GPT will retrain np.nan and will later be reprocessed
                print(
                    f"{counter}; [{this_comments.index[0]}, {this_comments.index[-1]}] comment; "
                    f"[{this_criteria.index[0]}, {this_criteria.index[-1]}] criteria; {datetime.now()}"
                )  # counter提示程序正常运行
                counter += 1

            # 如果出错，保存现有的回答，防止已有进度丢失
            except Exception as err:
                print(err)
            finally:
                yield answers, this_comments, this_criteria



#         all_answers_df = pd.concat(all_answers)
#         return all_answers_df.applymap(
#             lambda x: x.strip("\n") if type(x) == str else x
#         )  # strip the extra \n


def main(comment_file_name, read_dir, save_dir, criteria_file_dir, api_key='', criteria_batch_size=0,
         comments_batch_size=100):
    # criteria_batch_size/comment_batch_size: 批次处理中一批的容量大小，0表示不分批

    # 初始化计时器，读取各种文件
    start = time.time()
    g_criteria = pd.read_csv(criteria_file_dir, index_col=0)["criterion"]
    save_path = os.path.join(save_dir, comment_file_name)
    read_path = os.path.join(read_dir, comment_file_name)

    # Create result file and fill in the header if there isn't one.
    if os.path.isfile(save_path):
        old_results = pd.read_csv(save_path, index_col=0)
        if len(old_results) == 0:
            last_processed_comment_index = 0
        else:
            last_processed_comment_index = len(old_results)
    else:
        pd.DataFrame(columns=g_criteria).to_csv(save_path)
        last_processed_comment_index = 0

    # Continue reading comments from the last one unprocessed
    g_comments_reader = pd.read_csv(read_path, index_col=0, iterator=True)['comment'].loc[last_processed_comment_index:]

    # batch process criteria and comments
    batched_criteria = divide_into_batches(g_criteria, criteria_batch_size)
    batched_comments = divide_into_batches(g_comments_reader, comments_batch_size)

    # openai.api_key = api_key  # ChatGPT密匙
    results_by_batch_generator = comment_labeling_with_gpt(batched_comments, batched_criteria, comments_batch_size, criteria_batch_size)

    while True:
        try:
            new_results, current_comments, current_criteria = next(results_by_batch_generator)
        except StopIteration:
            break
        # 检查无效值
        naindxs = new_results[new_results.isna().any(axis=1)].index
        if len(naindxs) > 0:  # 若有无效值，将对应评论重新处理
            print("nan values are found at ", naindxs.tolist())
            response = get_response_from_gpt(current_comments[naindxs], current_criteria)
            print(response)
            new_results = insert_response_into_target_matrix(response, new_results, naindxs)
        else:
            print("No nan values are found! All comments have been processed!")
        new_results.index = current_comments[new_results.index]
        new_results.to_csv(os.path.join(save_dir, comment_file_name), header=False, mode='a')
        print(
            (time.time() - start) / len(new_results.index) / len(new_results.columns)
        )  # Print average processing time


openai.api_key = ''  # key
tokeniser = tiktoken.encoding_for_model("gpt-4")

# 运行
if __name__ == '__main__':
    logging.basicConfig(filename='mylog.log', encoding='utf-8', level=logging.INFO, filemode='a',
                        format='%(asctime)s %(message)s')
    # main(评论文件名，读取评论文件路径，保存结果路径，标签文件地址)
    # api_key: 选填
    # criteria_batch_size/comment_batch_size: 分批处理中一批次的容量大小，0表示不分批
    # 尽量保持criteria_batch_size=0
    main(
        'weedmaps.csv',
        'data',
        'processed-results',
        'label_and_questions.csv',
        criteria_batch_size=0,
        comments_batch_size=90,
    )
