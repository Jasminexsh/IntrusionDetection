# coding:utf-8
# kdd99数据集预处理
# 将kdd99符号型数据转化为数值型数据

import numpy as np
import csv

Train_File_Multi_classify = r'data\1-kddcup.data_10_percent_corrected.csv'
Unlabeled_Test_File_Multi_classify = r'data\2-kddcup.testdata.unlabeled_10_percent.csv'
Labeled_Test_File_Multi_classify = r'data\3-corrected.csv'


def pre_handel_data1():
    source_file = Train_File_Multi_classify
    handled_file = r'data\train_data_10_percent_corrected_multi_classify.csv'
    data_file = open(handled_file, 'w', newline='')  # python3.x中添加newline=''这一参数使写入的文件没有多余的空行
    with open(source_file, 'r') as data_source:
        csv_reader = csv.reader(data_source)
        csv_writer = csv.writer(data_file)
        count = 0  # 记录数据的行数，初始化为0
        for row in csv_reader:
            temp_line = np.array(row)  # 将每行数据存入temp_line数组里
            temp_line[1] = handle_protocol(row)  # 将源文件行中3种协议类型转换成数字标识
            temp_line[2] = handle_service(row)  # 将源文件行中70种网络服务类型转换成数字标识
            temp_line[3] = handle_flag(row)  # 将源文件行中11种网络连接状态转换成数字标识
            temp_line[41] = handle_label1(row)  # 将源文件行中23种攻击类型转换成数字标识
            csv_writer.writerow(temp_line)
            count += 1
        data_file.close()


def pre_handel_data2():
    source_file = Unlabeled_Test_File_Multi_classify
    handled_file = r'data\test_data_10_percent_corrected_multi_classify_unlabeled.csv'
    data_file = open(handled_file, 'w', newline='')  # python3.x中添加newline=''这一参数使写入的文件没有多余的空行
    with open(source_file, 'r') as data_source:
        csv_reader = csv.reader(data_source)
        csv_writer = csv.writer(data_file)
        count = 0  # 记录数据的行数，初始化为0
        for row in csv_reader:
            temp_line = np.array(row)  # 将每行数据存入temp_line数组里
            temp_line[1] = handle_protocol(row)  # 将源文件行中3种协议类型转换成数字标识
            temp_line[2] = handle_service(row)  # 将源文件行中70种网络服务类型转换成数字标识
            temp_line[3] = handle_flag(row)  # 将源文件行中11种网络连接状态转换成数字标识
            csv_writer.writerow(temp_line)
            count += 1
        data_file.close()


def pre_handel_data3():
    source_file = Labeled_Test_File_Multi_classify
    handled_file = r'data\test_data_10_percent_corrected_multi_classify_labeled.csv'
    data_file = open(handled_file, 'w', newline='')  # python3.x中添加newline=''这一参数使写入的文件没有多余的空行
    with open(source_file, 'r') as data_source:
        csv_reader = csv.reader(data_source)
        csv_writer = csv.writer(data_file)
        count = 0  # 记录数据的行数，初始化为0
        for row in csv_reader:
            temp_line = np.array(row)  # 将每行数据存入temp_line数组里
            temp_line[1] = handle_protocol(row)  # 将源文件行中3种协议类型转换成数字标识
            temp_line[2] = handle_service(row)  # 将源文件行中70种网络服务类型转换成数字标识
            temp_line[3] = handle_flag(row)  # 将源文件行中11种网络连接状态转换成数字标识
            temp_line[41] = handle_label2(row)  # 将源文件行中23种攻击类型转换成数字标识
            csv_writer.writerow(temp_line)
            count += 1
        data_file.close()


# 将相应的非数字类型转换为数字标识即符号型数据转化为数值型数据
def find_index(x, y):
    return [i for i in range(len(y)) if y[i] == x]


# 将训练集中的标签数字化
def find_index1(x, y):
    for i in range(len(y)):
        if y[i] == x:
            if i == 0:
                return [0]
            elif i == 5:
                return [1]
            else:
                return [2]


# 将带标签列的测试集中的标签数字化
def find_index2(x, y):
    for i in range(len(y)):
        if y[i] == x:
            if i == 0:
                return [0]
            elif i == 4:
                return [1]
            else:
                return [2]


# 定义将源文件行中3种协议类型转换成数字标识的函数
def handle_protocol(inputs):
    protocol_list = ['tcp', 'udp', 'icmp']
    if inputs[1] in protocol_list:
        return find_index(inputs[1], protocol_list)[0]


def handle_service(inputs):
    service_list = ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u',
                    'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest',
                    'hostnames',
                    'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell',
                    'ldap',
                    'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp',
                    'nntp',
                    'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje',
                    'shell',
                    'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time',
                    'urh_i', 'urp_i',
                    'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50']
    if inputs[2] in service_list:
        return find_index(inputs[2], service_list)[0]


def handle_flag(inputs):
    flag_list = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']
    if inputs[3] in flag_list:
        return find_index(inputs[3], flag_list)[0]


def handle_label1(inputs):
    label_list1 = ['normal.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.', 'smurf.',
                  'guess_passwd.', 'pod.', 'teardrop.', 'portsweep.', 'ipsweep.', 'land.', 'ftp_write.', 'back.',
                  'imap.', 'satan.', 'phf.', 'nmap.', 'multihop.', 'warezmaster.', 'warezclient.',
                  'spy.', 'rootkit.']
    # global label_list  # 在函数内部使用全局变量并修改它
    if inputs[41] in label_list1:
        return find_index1(inputs[41], label_list1)[0]
    """else:
        label_list.append(inputs[41])
        return find_index1(inputs[41], label_list)[0]"""


def handle_label2(inputs):
    label_list2 = ['normal.', 'snmpgetattack.', 'named.', 'xlock.', 'smurf.', 'ipsweep.', 'multihop.', 'xsnoop.',
                   'sendmail.', 'guess_passwd.', 'saint.', 'buffer_overflow.', 'portsweep.', 'pod.', 'apache2.', 'phf.',
                   'udpstorm.', 'warezmaster.', 'perl.', 'satan.', 'xterm.', 'mscan.', 'processtable.', 'ps.', 'nmap.',
                   'rootkit.', 'neptune.', 'loadmodule.', 'imap.', 'back.', 'httptunnel.', 'worm.', 'mailbomb.',
                   'ftp_write.', 'teardrop.', 'land.', 'sqlattack.', 'snmpguess.']
    # global label_list  # 在函数内部使用全局变量并修改它
    if inputs[41] in label_list2:
        return find_index2(inputs[41], label_list2)[0]
    """else:
        label_list.append(inputs[41])
        return find_index2(inputs[41], label_list)[0]"""


if __name__ == '__main__':
    # label_list = []
    pre_handel_data1()
    pre_handel_data2()
    pre_handel_data3()
    print('END')
