import csv
from collections import defaultdict

def read_csv_group_by_id(csv_file):
    group = defaultdict(list)
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        id_index = header.index('ID')
        for row in reader:
            id_val = row[id_index]
            rest = tuple(row[:id_index] + row[id_index+1:])
            group[id_val].append(rest)
    return group

def compare_groups(group1, group2):
    match_count = 0
    for id_val in group1:
        records1 = group1[id_val]
        records2 = group2.get(id_val)

        if records2 is None:
            print(f"ID {id_val} 不在文件2中")
            continue

        if len(records1) != len(records2):
            print(f"ID {id_val} 数量不一致：文件1有{len(records1)}条，文件2有{len(records2)}条")
            continue

        if sorted(records1) != sorted(records2):
            print(f"ID {id_val} 内容不一致：")
            print(f"  文件1记录：{records1}")
            print(f"  文件2记录：{records2}")
            continue

        match_count += 1

    return match_count

def main(file1, file2):
    group1 = read_csv_group_by_id(file1)
    group2 = read_csv_group_by_id(file2)

    match_count = compare_groups(group1, group2)
    print(f"\n完全一致的 ID 数量为：{match_count}")

if __name__ == '__main__':
    main('file1.csv', 'file2.csv')
