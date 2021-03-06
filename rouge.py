import PythonROUGE.PythonROUGE
import os
import json

def get_file_list_in_dir(dir, startWith = None, limit = None):
    list = []
    for f in os.listdir(dir):
        filepath = os.path.join(dir, f)
        if os.path.isfile(filepath) and (startWith == None or f.lower().startswith(startWith.lower())): list.append(filepath)
        if limit != None and len(list) >= limit: break
    return list

def calculate(guess, refdir, refstart, limit):
    guess_summary_list = [guess]
    ref_summary_list = [
        get_file_list_in_dir(refdir, refstart, limit)
    ]
    recall_list,precision_list,F_measure_list = PythonROUGE.PythonROUGE.PythonROUGE(guess_summary_list,ref_summary_list)

    print guess_summary_list[0]
    print 'recall = ' + str(recall_list)
    print 'precision = ' + str(precision_list)
    print 'F = ' + str(F_measure_list)
    print

    return F_measure_list[0]

if __name__ == "__main__":
    out = {
        'name':'DBM vs MMR',
        'labels': [],
        'DBM':[],
        'MMR':[]
    }

    out['labels'].append('D0607G')
    out['DBM'].append(calculate('./D0607.summary', '/home/yohanes/Workspace/duc/cleansum/06/D0607G', 'D0607', 10))
    out['MMR'].append(calculate('/home/yohanes/Workspace/duc/clean/06/D0607G/D0607G.mmr.summary', '/home/yohanes/Workspace/duc/cleansum/06/D0607G', 'D0607', 10))

    out['labels'].append('D0608H')
    out['DBM'].append(calculate('./D0608.summary', '/home/yohanes/Workspace/duc/cleansum/06/D0608H', 'D0608', 10))
    out['MMR'].append(calculate('/home/yohanes/Workspace/duc/clean/06/D0608H/D0608H.mmr.summary', '/home/yohanes/Workspace/duc/cleansum/06/D0608H', 'D0608', 10))

    out['labels'].append('D0609I')
    out['DBM'].append(calculate('./D0609.summary', '/home/yohanes/Workspace/duc/cleansum/06/D0609I', 'D0609', 10))
    out['MMR'].append(calculate('/home/yohanes/Workspace/duc/clean/06/D0609I/D0609I.mmr.summary', '/home/yohanes/Workspace/duc/cleansum/06/D0609I', 'D0609', 10))

    with open('report-rouge.json', 'wb') as outfile:
        json.dump([out], outfile)
