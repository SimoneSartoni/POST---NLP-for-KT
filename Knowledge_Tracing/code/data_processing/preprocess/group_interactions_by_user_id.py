import pandas as pd


def generate_sequences_of_same_length(df, seq_len, min_seq_len, output_filepath="/content/"):
    grouped = df.groupby("user_id")
    user_id = []
    problem_id = []
    question_id = []
    skill = []
    correct = []
    start_time, end_time, elapsed_time = [], [], []
    timestamp = []
    chunk_count = 0
    n_seq_longer_than_seq_len = 0
    number = 0.0
    length = 0.0
    for name, group in grouped:
        index = 0
        number += 1.0
        length += len(group['question_id'].values)
        while index + seq_len < len(group['question_id'].values):
            if index == 0:
                n_seq_longer_than_seq_len += 1
            chunk_count += 1
            user_id.append(name)
            problem_id.append(group['problem_id'].values[index:index+seq_len])
            question_id.append(group['question_id'].values[index:index+seq_len])
            skill.append(group['skill'].values[index:index+seq_len])
            correct.append(group['correct'].values[index:index+seq_len])
            start_time.append(group['start_time'].values[index:index+seq_len])
            end_time.append(group['end_time'].values[index:index+seq_len])
            elapsed_time.append(group['elapsed_time'].values[index:index+seq_len])
            timestamp.append(group['timestamp'].values[index:index+seq_len])
            index += seq_len
        if len(group['question_id'].values) - index > min_seq_len:
            user_id.append(name)
            problem_id.append(group['problem_id'].values[index:])
            question_id.append(group['question_id'].values[index:])
            skill.append(group['skill'].values[index:])
            correct.append(group['correct'].values[index:])
            start_time.append(group['start_time'].values[index:])
            end_time.append(group['end_time'].values[index:])
            elapsed_time.append(group['elapsed_time'].values[index:])
            timestamp.append(group['timestamp'].values[index:])
    print("Average sequence length is:", float(length)/float(number))
    print("Number of users with sequences longer than "+str(seq_len)+" is: " + str(n_seq_longer_than_seq_len))
    print("Number of times we chunk sequences with seq_len="+str(seq_len)+" is: " + str(chunk_count))
    data = {'question_id': question_id, 'problem_id': problem_id, 'user_id': user_id, 'correct': correct,
            'skill': skill, 'start_time': start_time, 'end_time': end_time, 'elapsed_time': elapsed_time,
            'timestamp': timestamp}
    new_df = pd.DataFrame(data=data)
    new_df.to_csv(output_filepath + 'interactions_grouped_with_same_length.csv')
    return new_df

