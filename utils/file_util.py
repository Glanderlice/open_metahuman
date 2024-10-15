import hashlib
import os


def make_dir(dir_path, clear=False):
    if clear:
        delete_dir(dir_path, delete_self=False)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def clear_dir(dir_path, clear_suffix=None):
    """
    清空文件夹下指定后缀的文件,如果文件夹不存在则新建空文件夹
    :param dir_path: 文件夹路径
    :param clear_suffix: 待清除文件的后缀, 请输入str或tuple(str,...), 如未指定默认删除所有文件
    :return:
    """
    if os.path.isdir(dir_path):
        for file_name in os.listdir(dir_path):
            if not clear_suffix or file_name.endswith(clear_suffix):
                file = os.path.join(dir_path, file_name)
                if os.path.isfile(file):
                    os.remove(file)
    else:
        os.makedirs(dir_path, exist_ok=True)


def delete_dir(directory, delete_self=True, console=False):
    """
    清空/删除目录及其下所有文件或子目录
    :param console:
    :param directory: 源目录
    :param delete_self: 是否删除源目录
    :return: None
    """
    if not os.path.isdir(directory):
        if console:
            print(f'directory {directory} does not exist')
        return
    for root, dirs, files in os.walk(directory, topdown=False):
        if console:
            print(root, dirs, files)
        for name in files:
            file_path = os.path.join(root, name)
            os.remove(file_path)
            if console:
                print('delete', file_path)
        for name in dirs:
            dir_path = os.path.join(root, name)
            os.rmdir(dir_path)
            if console:
                print('delete', dir_path)
    if delete_self:
        os.rmdir(directory)
    if console:
        print(f'{directory} has been cleared.')


def get_file_paths(input_dir, suffix=None):
    """获取input_dir文件夹下所有后缀名为suffix的文件路径, 当suffix=None表示返回何子文件/文件夹路径"""
    filenames = None
    if suffix is None:
        filenames = [file for file in os.listdir(input_dir)]
    elif isinstance(suffix, (tuple, list)):
        filenames = [file for file in os.listdir(input_dir) if any(file.lower().endswith(sfx) for sfx in suffix)]
    elif isinstance(suffix, str):
        filenames = [file for file in os.listdir(input_dir) if file.lower().endswith(suffix)]
    if filenames:
        filenames = sorted(filenames)
        filenames = [os.path.join(input_dir, filename) for filename in filenames]
    return filenames


def get_file_size(file_path):
    """计算文件大小：MB"""
    if file_path and os.path.exists(file_path):
        bytes_size = os.path.getsize(file_path)  # 获取文件大小，单位为字节
        return round(bytes_size / (1024 * 1024), 2)
    else:
        return 0


def get_folder_size(folder_path):
    """计算文件夹大小：MB"""
    total_size = 0
    if folder_path and os.path.isdir(folder_path):
        for dir_path, _, filenames in os.walk(folder_path):
            for filename in filenames:
                filepath = os.path.join(dir_path, filename)
                total_size += os.path.getsize(filepath)
    return round(total_size / (1024 * 1024), 2) if total_size > 0 else 0


def convert_encoding(input_file, output_file, input_encoding='GBK', output_encoding='UTF-8'):
    """文件编码转换: 默认从GBK->UTF-8"""
    with open(input_file, 'r', encoding=input_encoding) as infile:
        content = infile.read()
    converted_content = content.encode(output_encoding)
    with open(output_file, 'wb') as outfile:
        outfile.write(converted_content)


def file_hash(file_path, algorithm='sha256'):
    """为文件生成哈希值：64位16进制数构成的字符串"""
    if not file_path or not os.path.isfile(file_path):
        return None
    hash_obj = hashlib.new(algorithm)
    with open(file_path, 'rb') as file:
        while chunk := file.read(8192):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()
