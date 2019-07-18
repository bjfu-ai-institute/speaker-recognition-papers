from pyasv.speech_processing import ext_fbank_feature
from pyasv.pipeline import TFrecordClassBalanceGen
from pyasv.config import Config


train_urls = ['/home/fhq/data/aishell/url/aishell-small/train.scp']

def write_dict_to_text(path, dic, key_before_value=True, data_one_line=False, spaced=" "):
    with open(path, 'w') as f:
        for key in dic.keys():
            if type(dic[key]) == list or type(dic[key]) == set:
                if data_one_line:
                    data = "%s " % key
                    for dat in dic[key]:
                        data += dat + spaced
                    f.writelines(data + '\n')
                else:
                    for dat in dic[key]:
                        data = "%s %s\n"%(dat, key)
                        f.writelines(data)



def process(urls, enroll=None, test=None):
    ids = 0
    spk2id = {}
    id2utt = {}
    count = 0
    for url in urls:
        with open(url, 'r') as f:
            datas = f.readlines()
        for line in datas:
            p, spk = line.replace("\n", "").split(' ')
            if spk not in spk2id.keys():
                spk2id[spk] = ids
                id2utt[spk2id[spk]] = []
                ids += 1
            id2utt[spk2id[spk]].append(p)
        write_dict_to_text("tmp_%d"%count, id2utt)
        for key in spk2id.keys():
            id2utt[spk2id[key]] = []
        count += 1
    if enroll is not None:
        with open(enroll, 'r') as f:
            datas = f.readlines()
        for line in datas:
            url, spk = line.replace("\n", "").split(' ')
            if spk not in spk2id.keys():
                spk2id[spk] = ids
                id2utt[spk2id[spk]] = []
                ids += 1
            id2utt[spk2id[spk]].append(url)
        write_dict_to_text("tmp_enroll", id2utt)
        for key in spk2id.keys():
            id2utt[spk2id[key]] = []

    if test is not None:
        with open(test, 'r') as f:
            datas = f.readlines()
        for line in datas:
            url, spk = line.replace("\n", "").split(' ')
            if spk not in spk2id.keys():
                spk2id[spk] = ids
                id2utt[spk2id[spk]] = []
                ids += 1
            id2utt[spk2id[spk]].append(url)
        write_dict_to_text("tmp_test", id2utt)
        for key in spk2id.keys():
            id2utt[spk2id[key]] = []

    return len(id2utt.keys())

config = Config("x_vector.json")
writer = TFrecordClassBalanceGen(config, 'train_')
length = process(train_urls)
print(length)
config.n_speaker = length
for i in range(len(train_urls)):
    x, y = ext_fbank_feature("tmp_%d"%i, config)
    writer.write(x, y)

