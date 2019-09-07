#!/bin/bash

if [ -z "$1" ]; then
    echo "Need a destination."
    exit -1
fi

site=${1#*@}
user=${1%@*}

echo 'to '$user@$site
#rm _data _log -rf

rm *__pycache__* -rf
rm */__pycache__* -rf

# mv exp ../
# # scp -r -P 15044 ./* student@speaker.is99kdf.xyz:~/lhf/work/irm_test/extract_tfrecord
# scp -r -P 15043 ./* room@speaker.is99kdf.xyz:~/work/speech_en_test/c001_se

# mv ../exp ./

# rsync -av -e 'ssh -p 15043' --exclude-from='.vscode/exclude.lst' ./* room@speaker.is99kdf.xyz:~/work/paper_se_test/$1
rsync -av -e 'ssh ' --exclude-from='.idea/exclude.lst' ./* $user@$site:~/code/
# -a ：递归到目录，即复制所有文件和子目录。另外，打开归档模式和所有其他选项（相当于 -rlptgoD）
# -v ：详细输出
# -e ssh ：使用 ssh 作为远程 shell，这样所有的东西都被加密
# --exclude='*.out' ：排除匹配模式的文件，例如 *.out 或 *.c 等。
