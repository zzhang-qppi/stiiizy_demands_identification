import pandas as pd
import numpy as np
import openai
from retry import retry
import time


# 获取回答
@retry(Exception, tries=20, delay=5)
def get_response_from_list(m_comment, m_questions):
    # comment: a string of a single comment
    # questions: a list of questions about the comment to be fed to GPT

    #    sys_message = f'''You evaluate this comment "{m_comment}" against a list of questions.
    #    It is very important that you answer the questions purely based on the content of the comment.
    #    Don't make any interpretation beyond the exact words in the comment. Answer in yes and no only.
    #    '''
    sys_message = f'''You evaluate this comment '{m_comment}' against a list of {len(m_questions)} questions. 
    I need you {m_comment} answers in a list indexed by numbers and separated by one new line character. 
    Give me one and only one answer for each question! Answer in yes and no only.'''

    prompt = '\n'.join(
        [f"{i+1}. " + m_questions.iloc[i] for i in range(len(m_questions))]
    )

    return openai.ChatCompletion.create(
        messages=[
            {"role": "system", "content": sys_message},
            {"role": "user", "content": prompt},
        ],
        model="gpt-4",
        temperature=0,
    )


def comment_labeling_with_gpt(m_comments, m_questions):
    # comments: a DataFrame containing the comments.
    # questions: a Series whose indexes are rough labels and values are formulated questions

    # 初始回答汇总列表
    all_answers = []

    # divide one Series of questions into several smaller Series
    # of questions to comply with the 4096 token limit
    batched_questions = []
    numb = 6  # number of batches
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
            answers = pd.DataFrame("//", index=m_questions.index, columns=[comment])
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
                answers.loc[
                    this_batch.index[: len(list_of_resp)], comment
                ] = list_of_resp
                print(
                    f"{counter}; {m_comments.index[c]}th comment; {j}th batch"
                )  # counter提示程序正常运行
                counter += 1
            all_answers.append(answers.copy())
    except Exception as err:
        # 如果出错，保存现有的回答，防止已有进度丢失
        print(err)
    finally:
        all_answers_df = pd.concat((all_answers), axis=1)
        return all_answers_df.applymap(
            lambda x: x.strip("\n") if type(x) == str else x
        )  # strip the extra \n


def refill_na_values(m_results, m_questions):
    # results: a DataFrame where the index is the questions' index
    # and the column is the comments
    try:
        assert len(m_questions) == len(m_results.index)
        m_comments = m_results.columns
        naloc = list(
            zip(*np.where(m_results.isnull()))
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


start = time.time()

g_labels_questions = pd.read_csv("secondary_labels_questions.csv", index_col=0)
g_questions = g_labels_questions["questions"]
g_comments = pd.read_csv("xros3mini_comments.csv", index_col=0).iloc[:2, 0]
openai.api_key = "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"  # ChatGPT密匙

new_results = comment_labeling_with_gpt(g_comments, g_questions)
old_results = pd.read_csv('results/xros3mini_secondary_test_responses.csv', index_col=0)
refill_na_values(
    new_results, g_questions
)  # Feed questions with nan results to GPT again

results = pd.concat((old_results, new_results), axis=1)  # Concatenate the old and new results
results.to_csv("results/xros3mini_secondary_test_responses.csv")
print(
    (time.time() - start) / len(new_results.index) / len(new_results.columns)
)  # Print average processing time
