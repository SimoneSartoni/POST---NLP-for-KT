import pandas as pd


def generate_sequences_of_same_length(df, seq_len, output_filepath="/kaggle/working/"):
    grouped = df.groupby("user_id")
    user_id = []
    problem_id = []
    question_id = []
    skill = []
    correct = []
    start_time, end_time, elapsed_time = [], [], []
    timestamp = []
    for name, group in grouped:
        index = 0
        while index + seq_len < len(group['question_id'].values):
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
        user_id.append(name)
        problem_id.append(group['problem_id'].values[index:])
        question_id.append(group['question_id'].values[index:])
        skill.append(group['skill'].values[index:])
        correct.append(group['correct'].values[index:])
        start_time.append(group['start_time'].values[index:])
        end_time.append(group['end_time'].values[index:])
        elapsed_time.append(group['elapsed_time'].values[index:])
        timestamp.append(group['timestamp'].values[index:])
    data = {'question_id': question_id, 'problem_id': problem_id, 'user_id': user_id, 'correct': correct,
            'skill': skill, 'start_time': start_time, 'end_time': end_time, 'elapsed_time': elapsed_time,
            'timestamp': timestamp}
    new_df = pd.DataFrame(data=data)
    new_df.to_csv(output_filepath + 'interactions_grouped_with_same_length')
    return new_df

