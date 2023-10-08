import math

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
import collections


# 获取回答
@retry(exceptions=openai.error.Timeout, tries=10, delay=5)  # 若出错，重试20次，每次暂停5秒
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

    print("prompt sent to gpt")
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
                f"{num_to_letter_index(i)} | '" + m_comments.loc[i, 'comment'] + "'"
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


def divide_into_batches(data_reader, size_of_batch):
    # divide the Series of data into several smaller Series
    # to comply with the token limit or runtime efficiency
    # data: the file reader of the source data that needs to be divided
    # length_of_batch: the number of items in each patch
    #                  -1 - don't batch
    #                   0 - smart batch
    #                  >0 - fixed batch size
    if not issubclass(type(data_reader), collections.abc.Iterator):
        raise TypeError("Only an Iterator type is allowed.")
    if size_of_batch == 0:
        while True:
            next_batch = []
            try:
                next_batch = data_reader.get_chunk(3)
                while len(tokeniser.encode(prompt_formulator(next_batch, g_criteria))) <= 6500:
                    a = pd.concat((next_batch, data_reader.get_chunk(3)))
                    next_batch = a
            except StopIteration:
                print("smart batching completed")
            finally:
                yield next_batch
    elif size_of_batch < 0:
        next_batch = data_reader.get_chunk(100)
        while True:
            try:
                a = pd.concat((next_batch, data_reader.get_chunk(100)))
                next_batch = a
            except StopIteration:
                print("You asked not to batch.")
            finally:
                yield next_batch
    elif size_of_batch > 0:
        while True:
            next_batch = []
            try:
                next_batch = data_reader.get_chunk(size_of_batch)
            except StopIteration:
                print(f"fixed batching completed; size = {size_of_batch}")
            finally:
                yield next_batch

    # else:
    #     if 0 < size_of_batch <= len(data):
    #         batch_list = []
    #         # number of batches (which would very possibly be label_numb + 1)
    #         label_numb = len(data) // size_of_batch
    #         for i in range(label_numb):
    #             batch_list.append(
    #                 data[
    #                 len(data) // label_numb * i: len(data) // label_numb * (i + 1)
    #                 ]
    #             )
    #         if len(data) // label_numb * label_numb < len(data):
    #             batch_list.append(data[len(data) // label_numb * label_numb:])
    #         print("successfully batched criteria into ", len(batch_list), " batches")
    #     else:
    #         batch_list = [data]
    #         print("successfully read criteria. You opted not to batch criteria.")
    #     return batch_list


def comment_labeling_with_gpt(batched_comments_generator, m_criteria):
    # a generator function that yields results by batch
    # batched_comments_generator: a generator for batches of comments
    # m_criteria: a Series of criteria

    while True:
        this_comments = next(batched_comments_generator)
        if len(this_comments) == 0:
            print("all comments read")
            yield None, None
        # initialize the output matrix with np.nan
            # 将每个问题上传给ChatGPT并获取回答
        answers = pd.DataFrame(columns=m_criteria.index, index=this_comments.index)
        response = get_response_from_gpt(this_comments, m_criteria)
        print(response)
        print("successfully responded")
        insert_response_into_target_matrix(response, answers, this_comments.index)
        # comments that are missed by GPT will retrain np.nan and will later be reprocessed
        print(
            f"successfully processed at {datetime.now()}:"
            f"[{this_comments.index[0]}, {this_comments.index[-1]}] comments; "
            f"[{m_criteria.index[0]}, {m_criteria.index[-1]}] criteria"
        )
        yield answers, this_comments



#         all_answers_df = pd.concat(all_answers)
#         return all_answers_df.applymap(
#             lambda x: x.strip("\n") if type(x) == str else x
#         )  # strip the extra \n


def main(comment_file_name, read_dir, save_dir, criteria_file_dir, api_key='', comments_batch_size=0):
    # comments_batch_size:
    #                  -1 - don't batch
    #                   0 - smart batch
    #                  >0 - fixed batch size

    # 初始化计时器，读取各种文件
    start = time.time()
    global g_criteria
    g_criteria = pd.read_csv(criteria_file_dir, index_col=0)["criterion"]

    save_path = os.path.join(save_dir, comment_file_name)
    read_path = os.path.join(read_dir, comment_file_name)

    if os.path.isfile(save_path):
        old_results = pd.read_csv(save_path, index_col=0)
        last_processed_comment_length = len(old_results)
    # Create result file and fill in the header if there isn't one.
    else:
        pd.DataFrame(columns=g_criteria).to_csv(save_path)
        last_processed_comment_length = 0

    # Continue reading comments from the last one unprocessed
    g_comments_reader = pd.read_csv(read_path, index_col=0, iterator=True)
    g_comments_reader.get_chunk(last_processed_comment_length)

    # batch comments
    batched_comments_generator = divide_into_batches(g_comments_reader, comments_batch_size)
    # openai.api_key = api_key  # ChatGPT密匙
    results_by_batch_generator = comment_labeling_with_gpt(batched_comments_generator, g_criteria)

    while True:
        new_results, current_comments = next(results_by_batch_generator)
        if new_results is None:
            break
        # 检查无效值
        naindxs = new_results[new_results.isna().any(axis=1)].index
        if len(naindxs) > 0:  # 若有无效值，将对应评论重新处理
            print("nan values are found at ", naindxs.tolist())
            response = get_response_from_gpt(current_comments[naindxs], g_criteria)
            print(response)
            new_results = insert_response_into_target_matrix(response, new_results, naindxs)
        else:
            print("No nan values are found! All comments have been processed!")
        new_results['comment'] = current_comments['comment']
        new_results.to_csv(os.path.join(save_dir, comment_file_name), header=False, mode='a')
        print(
            (time.time() - start) / len(new_results.index) / len(new_results.columns)
        )  # Print average processing time
    return 0


openai.api_key = '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'  # key
tokeniser = tiktoken.encoding_for_model("gpt-4")

# 运行
if __name__ == '__main__':
    logging.basicConfig(filename='mylog.log', encoding='utf-8', level=logging.INFO, filemode='a',
                        format='%(asctime)s %(message)s')
    # main(评论文件名，读取评论文件路径，保存结果路径，标签文件地址)
    # api_key: 选填
    # comments_batch_size:
    #                  -1 - don't batch
    #                   0 - smart batch
    #                  >0 - fixed batch size
    main(
        'yelp.csv',
        'data',
        'processed-results-2',
        'label_and_questions.csv',
        comments_batch_size=0,
    )
