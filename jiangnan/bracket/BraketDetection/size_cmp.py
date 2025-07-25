import csv
from collections import defaultdict



def compute_ids(csv_file,encode="utf-8",index_names=[]):
    ids=set()
    group = defaultdict(list)
    with open(csv_file, newline='', encoding=encode) as f:
        reader = csv.reader(f)
        header = next(reader)
        id_index = 2
        c_indices=[]
        for name in index_names:
            c_indices.append(header.index(name))
        for row in reader:
            id_val = row[id_index]
            val1,val2=id_val.split(',')
            val1=float(val1.strip()[1:])
            val2=float(val2.strip()[:-1])
            ids.add((val1,val2))
            row_content=""
            for c_index in c_indices:
                row_content+=row[c_index]
            rest = tuple(row_content)
            group[f"({val1},{val2})"].append(rest)
    return list(ids),group


def match_ids(ids,csv_file,encode="utf-8",index_names=[]):
    id_map={}
    group = defaultdict(list)
    with open(csv_file, newline='', encoding=encode) as f:
        reader = csv.reader(f)
        header = next(reader)
        id_index = 2
        c_indices=[]
        for name in index_names:
            c_indices.append(header.index(name))
        for row in reader:
            id_val = row[id_index]
            val1,val2=id_val.split(',')
            val1=float(val1.strip()[1:])
            val2=float(val2.strip()[:-1])
            dis=float("inf")
            id2=None
            for id in ids:
                distance=(id[0]-val1)*(id[0]-val1)+(id[1]-val2)*(id[1]-val2)
                if distance<dis:
                    dis=distance
                    id2=id
            if id2 is not None and dis<50*50:


            


                row_content=""
                for c_index in c_indices:
                    row_content+=row[c_index]
                rest = tuple(row_content)
                group[f"({id2[0]},{id2[1]})"].append(rest)
    return id_map,group
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

def main(file1, file2,index_names):
    ids,group1 = compute_ids(file1,encode="gbk",index_names=index_names)
    id_map,group2 =  match_ids(ids,file2,encode="gbk",index_names=index_names)


    match_count = compare_groups(group1, group2)
    print(f"\n完全一致的 ID 数量为：{match_count}")

if __name__ == '__main__':
    gt=r"./output/尺寸标注信息.csv"  
    test=r"./output/尺寸标注信息.csv"
    main(gt, test,["标注句柄" ,"肘板类别","标注值"]) #要测试的列名
