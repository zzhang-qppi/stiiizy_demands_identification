import pandas as pd
import numpy as np
import openai
from retry import retry
import time
import os
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

    comment_string = '\n'.join(
        [f"{chr(97+i)}. " + m_comment.iloc[i] for i in range(len(m_comment))]
    )
    prompt = f'''Here is a buyer's comment of e-cigarette: "{m_comment}".
    Give me the numerical indices of the following criteria that this comment mentions. 
    I need answer in the form like "25, 33, 89, 100"\n'''

    indexed_criteria = '\n'.join(
        [f"{m_questions.index[i]}. " + m_questions.iloc[i] for i in range(len(m_questions))]
    )

    return openai.ChatCompletion.create(
        messages=[
            {"role": "system", "content": "You evaluate comments on an e-cigarette product against a list of criteria I provide you."},
            {"role": "user", "content": prompt+indexed_criteria},
        ],
        model="gpt-4",
        temperature=0,
        request_timeout=60,
    )


def comment_labeling_with_gpt(m_comments, m_questions):
    # comments: a DataFrame containing the comments.
    # questions: a Series whose indexes are rough labels and values are formulated questions

    # 初始回答汇总列表
    all_answers = []

    # divide one Series of questions into several smaller Series
    # of questions to comply with the 4096 token limit
    batched_questions = []
    numb = 4
    # number of batches (which would likely be numb + 1)
    for i in range(numb):
        batched_questions.append(
            m_questions[
                len(m_questions) // numb * i : len(m_questions) // numb * (i + 1)
            ]
        )
    if len(m_questions) // numb * numb < len(m_questions):
        batched_questions.append(m_questions[len(m_questions) // numb * numb:])
    print("successfully batched into ", len(batched_questions), " batches")

    counter = 1
    try:
        for c in range(len(m_comments)):
            comment = m_comments.iloc[c]
            answers = pd.DataFrame("//", columns=m_questions.index, index=[comment])
            for j in range(len(batched_questions)):
                # 将每个问题上传给ChatGPT并获取回答
                this_batch = batched_questions[j]
                response = get_response_from_list(comment, this_batch)
                print("successfully responded")
                list_of_resp = (response["choices"][0]["message"]["content"]).split(
                    "\n", len(this_batch) - 1
                )
                if len(list_of_resp) == len(this_batch):
                    print("correct number of responses")
                else:
                    print("wrong number of responses")
                    print(len(list_of_resp), " responses")
                    print(len(this_batch), " questions")
                answers.loc[comment, this_batch.index[: len(list_of_resp)]] = list_of_resp
                print(
                    f"{counter}; {m_comments.index[c]}th comment; {j}th batch; {datetime.now()}"
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

    print(g_comments)
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
openai.api_key=''
print(get_response_from_list(g_comments[11:15], g_labels_questions[:20])["choices"][0]["message"]["content"])

