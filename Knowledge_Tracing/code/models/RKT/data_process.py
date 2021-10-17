# Passages needed to recompute pro_pro_skills
# Initially I will use file with the ones already computed by RKT authors
# In the future I will modify the code and try to compute better ones.

import os
import pandas as pd
import numpy as np
from scipy import sparse


class DataProcess:
    def __init__(self, data_folder='./', output_folder='./', file_name='skill_builder_data_corrected_collapsed.csv',
                 min_inter_num=3, dataset_name='assistment_2012'):
        print("Process Dataset %s" % data_folder)
        self.min_inter_num = min_inter_num
        self.data_folder = data_folder
        self.file_name = file_name
        self.output_folder = output_folder
        self.dataset_name = dataset_name

    def process_csv(self):
        # pre-process original csv file for assist dataset

        # read csv file
        data_path = os.path.join(self.data_folder, self.file_name)
        df = pd.read_csv(data_path, low_memory=False, encoding="ISO-8859-1")
        print('original records number %d' % len(df))

        # delete empty skill_id
        df = df.dropna(subset=['skill_id'])
        df = df[~df['skill_id'].isin(['noskill'])]
        print('After removing empty skill_id, records number %d' % len(df))

        # delete scaffolding problems
        df = df[df['original'].isin([1])]
        print('After removing scaffolding problems, records number %d' % len(df))

        # delete the users whose interaction number is less than min_inter_num
        users = df.groupby(['user_id'], as_index=True)
        delete_users = []
        for u in users:
            if len(u[1]) < self.min_inter_num:
                delete_users.append(u[0])
        print('deleted user number based min-inters %d' % len(delete_users))
        df = df[~df['user_id'].isin(delete_users)]
        print('After deleting some users, records number %d' % len(df))
        # print('features: ', df['assistment_id'].unique(), df['answer_type'].unique())

        df.to_csv(os.path.join(self.output_folder, '%s_processed.csv' % self.file_name))

    def pro_skill_graph(self):
        df = pd.read_csv(os.path.join(self.output_folder, '%s_processed.csv' % self.file_name), low_memory=False,
                         encoding="ISO-8859-1")
        problems = df['problem_id'].unique()
        pro_id_dict = dict(zip(problems, range(len(problems))))
        print('problem number %d' % len(problems))

        pro_type = df['problem_type'].unique()
        pro_type_dict = dict(zip(pro_type, range(len(pro_type))))
        print('problem type: ', pro_type_dict)

        pro_feat = []
        pro_skill_adj = []
        skill_id_dict, skill_cnt = {}, 0
        for pro_id in range(len(problems)):
            tmp_df = df[df['problem_id'] == problems[pro_id]]
            tmp_df_0 = tmp_df.iloc[0]

            # pro_feature: [ms_of_response, answer_type, mean_correct_num]
            ms = tmp_df['ms_first_response'].abs().mean()
            p = tmp_df['correct'].mean()
            pro_type_id = pro_type_dict[tmp_df_0['problem_type']]
            tmp_pro_feat = [0.] * (len(pro_type_dict) + 2)
            tmp_pro_feat[0] = ms
            tmp_pro_feat[pro_type_id + 1] = 1.
            tmp_pro_feat[-1] = p
            pro_feat.append(tmp_pro_feat)

            # build problem-skill bipartite
            s = tmp_df_0['skill_id']
            skill_id_dict[s] = skill_cnt
            skill_cnt += 1
            pro_skill_adj.append([pro_id, skill_id_dict[s], 1])

        pro_skill_adj = np.array(pro_skill_adj).astype(np.int32)
        pro_feat = np.array(pro_feat).astype(np.float32)
        pro_feat[:, 0] = (pro_feat[:, 0] - np.min(pro_feat[:, 0])) / (np.max(pro_feat[:, 0]) - np.min(pro_feat[:, 0]))
        pro_num = np.max(pro_skill_adj[:, 0]) + 1
        skill_num = np.max(pro_skill_adj[:, 1]) + 1
        print('problem number %d, skill number %d' % (pro_num, skill_num))

        # save pro-skill-graph in sparse matrix form
        pro_skill_sparse = sparse.coo_matrix(
            (pro_skill_adj[:, 2].astype(np.float32), (pro_skill_adj[:, 0], pro_skill_adj[:, 1])),
            shape=(pro_num, skill_num))
        sparse.save_npz(os.path.join(self.output_folder, 'pro_skill_sparse.npz'), pro_skill_sparse)

        # save pro-id-dict, skill-id-dict
        self.save_dict(pro_id_dict, os.path.join(self.output_folder, 'pro_id_dict.txt'))
        self.save_dict(skill_id_dict, os.path.join(self.output_folder, 'skill_id_dict.txt'))

        # save pro_feat_arr
        np.savez(os.path.join(self.output_folder, 'pro_feat.npz'), pro_feat=pro_feat)

    def generate_user_sequence(self, seq_file):
        # generate user interaction sequence
        # and write to data.txt

        df = pd.read_csv(os.path.join(self.output_folder, '%s_processed.csv' % self.file_name), low_memory=False,
                         encoding="ISO-8859-1")
        ui_df = df.groupby(['user_id'], as_index=True)
        print('user number %d' % len(ui_df))

        user_inters = []
        cnt = 0
        for ui in ui_df:
            tmp_user, tmp_inter = ui[0], ui[1]
            tmp_problems = list(tmp_inter['problem_id'])
            tmp_skills = list(tmp_inter['skill_id'])
            tmp_ans = list(tmp_inter['correct'])
            tmp_end_time = list(tmp_inter['end_time'])
            user_inters.append([[len(tmp_inter)], tmp_skills, tmp_problems, tmp_ans, tmp_end_time])

        write_file = os.path.join(self.output_folder, seq_file)
        self.write_txt(write_file, user_inters)

    def save_dict(self, dict_name, file_name):
        f = open(file_name, 'w')
        f.write(str(dict_name))
        f.close()

    def write_txt(self, file, data):
        with open(file, 'w') as f:
            for dd in data:
                for d in dd:
                    f.write(str(d) + '\n')

    def read_user_sequence(self, filename, max_len=200, min_len=3, shuffle_flag=True):
        with open(os.path.join(self.output_folder, filename), 'r') as f:
            lines = f.readlines()
        with open(os.path.join(self.output_folder, 'skill_id_dict.txt'), 'r') as f:
            skill_id_dict = eval(f.read())
        with open(os.path.join(self.output_folder, 'pro_id_dict.txt'), 'r') as f:
            pro_id_dict = eval(f.read())

        y, skill, problem, real_len, timestamp = [], [], [], [], []
        skill_num, pro_num = len(skill_id_dict), len(pro_id_dict)
        print('skill num, pro num, ', skill_num, pro_num)

        index = 0
        while index < len(lines):
            num = eval(lines[index])[0]
            tmp_skills = eval(lines[index + 1])[:max_len]
            if self.dataset_name == 'assistment_2012':
                tmp_skills = [ele + 1 for ele in tmp_skills]  # for assist12
            if self.dataset_name == 'assistment_2009':
                tmp_skills = [skill_id_dict[ele]+1 for ele in tmp_skills]     # for assist09
            tmp_pro = eval(lines[index + 2])[:max_len]
            tmp_pro = [pro_id_dict[ele] + 1 for ele in tmp_pro]
            tmp_ans = eval(lines[index + 3])[:max_len]
            tmp_time = eval(lines[index + 4])[:max_len]

            if num >= min_len:
                tmp_real_len = len(tmp_skills)
                # Completion sequence
                tmp_ans += [-1] * (max_len - tmp_real_len)
                tmp_skills += [0] * (max_len - tmp_real_len)
                tmp_pro += [0] * (max_len - tmp_real_len)
                tmp_time += [-1] * (max_len - tmp_real_len)

                y.append(tmp_ans)
                skill.append(tmp_skills)
                problem.append(tmp_pro)
                real_len.append(tmp_real_len)
                timestamp.append(tmp_time)

            index += 5

        y = np.array(y).astype(np.float32)
        skill = np.array(skill).astype(np.int32)
        problem = np.array(problem).astype(np.int32)
        real_len = np.array(real_len).astype(np.int32)
        timestamp = np.array(timestamp).astype(np.datetime64)

        print(skill.shape, problem.shape, y.shape, real_len.shape)
        print(np.max(y), np.min(y))
        print(np.max(real_len), np.min(real_len))
        print(np.max(skill), np.min(skill))
        print(np.max(problem), np.min(problem))

        np.savez(os.path.join(self.output_folder, "%s.npz" % self.file_name), problem=problem, y=y, skill=skill,
                 time=timestamp, real_len=real_len, skill_num=skill_num, problem_num=pro_num)


def execute(data_folder='C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/data/assistments/2012_2013/',
            output_folder='C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/intermediate_files/'
                          'assistments/2012_2013/',
            file_name='2012-2013-data-with-predictions-4-final.csv', min_inter_num=3):
    DP = DataProcess(data_folder, output_folder, file_name, min_inter_num)
    DP.process_csv()
    DP.pro_skill_graph()
    DP.generate_user_sequence('data.txt')
    DP.read_user_sequence('data.txt')
    return DP
