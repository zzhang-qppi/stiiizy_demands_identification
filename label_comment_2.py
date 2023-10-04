import pandas as pd
import numpy as np
import openai
from retry import retry
import time
import os
import re
import logging
from datetime import datetime


# 获取回答
# @retry(exceptions=Exception, tries=20, delay=5)
def get_response_from_gpt(m_comment, m_criteria):
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

    if type(m_comment) is str:
        indexed_comment_string = m_comment
    else:
        indexed_comment_string = '\n'.join(
            [f"{num_to_letter_index(i)}. " + m_comment[i] for i in m_comment.index]
        )
    # prompt = f'''Here is a buyer's comment of e-cigarette: "{indexed_comment_string}".
    # Which ones of the following criteria does this comment mention? Answer simply with the numerical indices.
    # \n'''
    prompt = f'''Here are a list of buyer's comments on an e-cigarette product:\n"{indexed_comment_string}". 
    In the exact order, take one comment at a time and answer:
    Which ones of the following criteria does this comment mention? 
    Answer simply with the numerical indices!
    If the comment doesn't mention any of the criteria, answer -1 for that answer.
    Your response should compose of lines in the format of this:
    [index of a comment] ***: [a list of indices of the mentioned criteria]
    \n'''

    indexed_criteria_string = '\n'.join(
        [f"{i}. " + m_criteria[i] for i in m_criteria.index]
    )
    print(prompt+indexed_criteria_string)
    return openai.ChatCompletion.create(
        messages=[
            {"role": "system", "content": "You evaluate comments on an e-cigarette product against a list of criteria I provide you."},
            {"role": "user", "content": prompt+indexed_criteria_string},
        ],
        model="gpt-4",
        temperature=0.1,
        request_timeout=300,
    )

# number to letter, e.g. 29---‘ad’
def num_to_letter_index(num: int):
    quotient = num // 26
    remainder = num % 26
    if quotient == 0:
        return chr(97+num)
    else:
        return chr(97+quotient-1)+chr(97+remainder)

# letter to number
def letter_index_to_num(ind: str):
    num = -1
    rev_ind = list(reversed(ind))
    for i in range(len(ind)):
        num += (26**i)*(ord(rev_ind[i])-96)
    return num

def insert_response_into_target_matrix(response, outmatrix, num_of_comments):
    # Intelligently integrate GPT response into a target matrix

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
        mentioned_criteria_indices = np.array(re.findall("\d+", resp), dtype="i")
        comment_index = letter_index_to_num(re.findall("[a-z]+",resp.split("***")[0])[0])
        outmatrix.loc[comment_index, :] = 0
        outmatrix.loc[comment_index, mentioned_criteria_indices] = 1
    return outmatrix.copy()


def divide_into_batches(data, length_of_batch):
    # divide the Series of data into several smaller Series
    # to comply with the token limit or runtime efficiency
    # data: the source data that needs to be divided
    # length_of_batch: the number of items in each patch

    if 0 < length_of_batch <= len(data):
        batch_list = []
        # number of batches (which would very possibly be label_numb + 1)
        label_numb = len(data) // length_of_batch
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

def comment_labeling_with_gpt(m_comments, m_criteria, criteria_batch_size=0, comments_batch_size=50):
    # comments: a Series of comments.
    # criteria: a Series of criteria

    # 初始回答汇总列表, concatenate later
    all_answers = []

    # batch process criteria and comments
    batched_criteria = divide_into_batches(m_criteria, criteria_batch_size)
    batched_comments = divide_into_batches(m_comments, comments_batch_size)

    counter = 1
    try:
        for c in range(len(batched_comments)):
            this_comments = batched_comments[c]
            # initialize the output matrix with np.nan
            answers = pd.DataFrame(columns=m_criteria.index, index=this_comments.index)
            for j in range(len(batched_criteria)):
                # 将每个问题上传给ChatGPT并获取回答
                this_criteria = batched_criteria[j]
                response = get_response_from_gpt(this_comments, this_criteria)
                print(response)
                print("successfully responded")
                insert_response_into_target_matrix(response, answers, len(this_comments))
                # comments that are missed by GPT will retrain np.nan and will later be reprocessed
                print(
                    f"{counter}; [{this_comments.index[0]}, {this_comments.index[-1]}] comment; [{this_criteria.index[0]}, {this_criteria.index[-1]}] criteria; {datetime.now()}"
                )  # counter提示程序正常运行
                counter += 1
            all_answers.append(answers.copy())
    except Exception as err:
        # 如果出错，保存现有的回答，防止已有进度丢失
        print(err)
    finally:
        concated_ans = pd.concat(all_answers)
        print(len(concated_ans), 'answers returned from ', len(m_comments), 'comments received')
        return concated_ans
#         all_answers_df = pd.concat(all_answers)
#         return all_answers_df.applymap(
#             lambda x: x.strip("\n") if type(x) == str else x
#         )  # strip the extra \n


def main(comment_file_name, read_dir, save_dir, criteria_file_dir, api_key='', criteria_batch_size=0, comments_batch_size=100):
    # criteria_batch_size/comment_batch_size: 批次处理中一批的容量大小，0表示不分批


    start = time.time()
    g_criteria = pd.read_csv(criteria_file_dir, index_col=0)["criterion"]
    save_path = os.path.join(save_dir, comment_file_name)
    read_path = os.path.join(read_dir, comment_file_name)

    # Create result file and fill in the header if there isn't one.
    if os.path.isfile(save_path):
        old_results = pd.read_csv(save_path, index_col=0)
        if len(old_results) == 0:
            last_comment = 0
        else:
            last_comment = len(old_results)
    else:
        pd.DataFrame(columns=g_criteria).to_csv(save_path)
        last_comment = 0

    # Continue reading comments from the last one unprocessed
    g_comments = pd.read_csv(read_path, index_col=0)['comment'].loc[last_comment:]

#    openai.api_key = api_key  # ChatGPT密匙
    new_results = comment_labeling_with_gpt(g_comments, g_criteria,
                                            criteria_batch_size=criteria_batch_size,
                                            comments_batch_size=comments_batch_size)
    naindxs = new_results[new_results.isna().any(axis=1)].index
    if len(naindxs) > 0:
        print("nan values are found at ", naindxs.tolist())
        response = get_response_from_gpt(g_comments[naindxs], g_criteria)
        print(response)
        new_results = insert_response_into_target_matrix(response, new_results, len(naindxs))
    else:
        print("No nan values are found! All comments have been processed!")
    new_results.index = g_comments[new_results.index]
    new_results.to_csv(os.path.join(save_dir, comment_file_name), header=False, mode='a')
    print(
        (time.time() - start) / len(new_results.index) / len(new_results.columns)
    )  # Print average processing time


openai.api_key = '' # key

# 运行语句
if __name__ == '__main__':
    logging.basicConfig(filename='mylog.log', encoding='utf-8', level=logging.INFO, filemode='a', format='%(asctime)s %(message)s')
    main(
        'weedmaps.csv',
         'data',
         'processed-results-2',
         'label_and_questions.csv',
        criteria_batch_size=0,
        comments_batch_size=100,
    )


