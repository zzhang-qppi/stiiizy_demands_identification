import random

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
def get_response_from_list(m_comment, m_questions):
    # comment: a string of a single comment
    # questions: a list of questions about the comment to be fed to GPT

    #    sys_message = f'''You evaluate this comment "{m_comment}" against a list of questions.
    #    It is very important that you answer the questions purely based on the content of the comment.
    #    Don't make any interpretation beyond the exact words in the comment. Answer in yes and no only.
    #    '''

    # sys_message = f'''You evaluate this consumer's comment about a e-cigarette product against a list of {len(m_questions)} criteria. The comment is "{m_comment}". You take
    # each criterion from the list and answer a question that take a form like this: Does this comment mention [the criterion]?
    # I need {len(m_questions)} answers in a list indexed by numbers in the exact same order as the criteria list and separated by one new line character.
    # You should give me one and only one answer for each criterion! Give me the answers in yes/no only. Don't give me the questions.'''
    if type(m_comment) is str:
        comment_string = m_comment
    else:
        comment_string = '\n'.join(
            [f"{num_to_letter_index(i)}. " + m_comment.iloc[i] for i in range(len(m_comment))]
        )
    # prompt = f'''Here is a buyer's comment of e-cigarette: "{comment_string}".
    # Which ones of the following criteria does this comment mention? Answer simply with the numerical indices.
    # \n'''
    prompt = f'''Here are a list buyer's comments of e-cigarette: "{comment_string}". 
    In the exact order, take one comment at a time and answer:
    Which ones of the following criteria does this comment mention? Answer simply with the numerical indices.
    \n'''

    indexed_criteria = '\n'.join(
        [f"{m_questions.index[i]}. " + m_questions.iloc[i] for i in range(len(m_questions))]
    )
    print(prompt+indexed_criteria)
    return openai.ChatCompletion.create(
        messages=[
            {"role": "system", "content": "You evaluate comments on an e-cigarette product against a list of criteria I provide you."},
            {"role": "user", "content": prompt+indexed_criteria},
        ],
        model="gpt-4",
        temperature=0.1,
        request_timeout=300,
    )


def num_to_letter_index(num: int):
    quotient = num // 26
    remainder = num % 26
    if quotient == 0:
        return chr(97+num)
    else:
        return chr(97+quotient-1)+chr(97+remainder)

def letter_index_to_num(ind: str):
    num = -1
    rev_ind = list(reversed(ind))
    for i in range(len(ind)):
        num += (26**i)*(ord(rev_ind[i])-96)
    return num


def comment_labeling_with_gpt(m_comments, m_criteria, batch_questions=False, batch_comments=True):
    # comments: a DataFrame containing the comments.
    # criteria: a Series of criteria

    # 初始回答汇总列表
    all_answers = []

    # divide the Series of criteria into several smaller Series
    # to comply with the token limit
    if batch_questions:
        batched_criteria = []
        label_numb = len(m_criteria) // 100  # number of batches if 100 questions per batch
        # number of batches (which would very possibly be label_numb + 1)
        for i in range(label_numb):
            batched_criteria.append(
                m_criteria[
                len(m_criteria) // label_numb * i: len(m_criteria) // label_numb * (i + 1)
                ]
            )
        if len(m_criteria) // label_numb * label_numb < len(m_criteria):
            batched_criteria.append(m_criteria[len(m_criteria) // label_numb * label_numb:])
        print("successfully batched criteria into ", len(batched_criteria), " batches")
    else:
        batched_criteria = [m_criteria]
        print("successfully read questions. You opted not to batch questions.")

    # divide the Series of comments into several smaller Series
    # to comply with the token limit
    if batch_comments:
        batched_comments = []
        comment_numb = len(m_comments) // 50  # number of batches if 50 comments per batch
        # number of batches (which would very possibly be comment_numb + 1)
        for i in range(comment_numb):
            batched_comments.append(
                m_comments[
                    len(m_comments) // comment_numb * i : len(m_comments) // comment_numb * (i + 1)
                ]
            )
        if len(m_comments) // comment_numb * comment_numb < len(m_comments):
            batched_comments.append(m_comments[len(m_comments) // comment_numb * comment_numb:])
        print("successfully batched criteria into ", len(batched_comments), " batches")
    else:
        batched_comments = [m_comments]
        print("successfully read questions. You opted not to batch questions.")

    counter = 1
    try:
        for c in range(len(batched_comments)):
            this_comments = batched_comments[c]
            answers = pd.DataFrame(0, columns=m_criteria.index, index=[this_comments.index])
            for j in range(len(batched_criteria)):
                # 将每个问题上传给ChatGPT并获取回答
                this_criteria = batched_criteria[j]
                response = get_response_from_list(this_comments, this_criteria)
                print("successfully responded")
                list_of_resp = (response["choices"][0]["message"]["content"]).split(
                    "\n", len(this_criteria) - 1
                )
                if len(list_of_resp) == len(this_criteria):
                    print("correct number of responses")
                else:
                    print("wrong number of responses")
                    print(len(list_of_resp), " responses")
                    print(len(this_criteria), " comments")
                for resp in list_of_resp:
                    mentioned_criteria_indices = re.findall('\d+', resp)
                    comment_index = letter_index_to_num(resp.split('.')[0])
                    answers.loc[comment_index, mentioned_criteria_indices] = 1
                print(
                    f"{counter}; ({this_comments.index.tolist()}) comment; {this_criteria.index.tolist()}criteria; {datetime.now()}"
                )  # counter提示程序正常运行
                counter += 1
            all_answers.append(answers.copy())
            print(c, 'comment appended.')
    except Exception as err:
        # 如果出错，保存现有的回答，防止已有进度丢失
        print(err)
    finally:
        return pd.concat(all_answers)
#         all_answers_df = pd.concat(all_answers)
#         return all_answers_df.applymap(
#             lambda x: x.strip("\n") if type(x) == str else x
#         )  # strip the extra \n


def refill_na_values(m_results, m_questions):
    # results: a DataFrame where the index is the questions' index
    # and the column is the comments
    try:
        assert len(m_questions) == len(m_results.index)
        m_comments = m_results.columns
        naloc = list(
            zip(*np.where(m_results == '//'))
        )  # a list of integar locations of nan values; [(row, column),]
        print(len(naloc), "nan values are found.")
        for loc in naloc:
            prompt = f"""According to this comment on an e-cigarette product "{m_comments[loc[1]]}", 
            {m_questions[loc[0]]}? Answer in yes and no only.
            """
            m_results.iloc[loc] = openai.ChatCompletion.create(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4",
                temperature=0,
            )["choices"][0]["message"]["content"]
            print(loc, "nan value is replaced as", m_results.iloc[loc])
    except Exception as err:
        print(err)
    finally:
        return m_results


def main(comment_file_name, read_directory, save_directory, questions_directory, api_key):
    start = time.time()
    g_labels_questions = pd.read_csv(questions_directory, index_col=0)
    g_questions = g_labels_questions["criterion"]
    save_path = os.path.join(save_directory, comment_file_name)
    read_path = os.path.join(read_directory, comment_file_name)

    # Create result file and fill in the header if there isn't one.
    if os.path.isfile(save_path):
        old_results = pd.read_csv(save_path, index_col=0)
        if len(old_results) == 0:
            last_comment = 0
        else:
            last_comment = len(old_results)
    else:
        pd.DataFrame(columns=g_questions).to_csv(save_path)
        last_comment = 0

    # Continue reading comments from the last one unprocessed
    g_comments = pd.read_csv(read_path, index_col=0)['comment'].dropna().loc[last_comment:]

    openai.api_key = api_key  # ChatGPT密匙
    new_results = comment_labeling_with_gpt(g_comments, g_questions)
    refill_na_values(
        new_results, g_questions
    )  # Feed questions with nan results to GPT again

    new_results.to_csv(os.path.join(save_directory, comment_file_name), header=False, mode='a')
    print(
        (time.time() - start) / len(new_results.index) / len(new_results.columns)
    )  # Print average processing time


# 运行语句
# if __name__ == '__main__':
#     logging.basicConfig(filename='mylog.log', encoding='utf-8', level=logging.INFO, filemode='a', format='%(asctime)s %(message)s')
#     main('weedmaps.csv', 'data', 'processed-results', 'label_and_questions.csv', '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')

g_labels_questions = pd.read_csv('label_and_questions.csv', index_col=0)['criterion']
g_comments = pd.read_csv('data/weedmaps.csv', index_col=0)['comment'].dropna()
openai.api_key='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'


