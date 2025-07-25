import json

# 读取type.json文件
with open('./type.json', 'r', encoding='UTF-8') as file:
    data = json.load(file)

# 遍历每个键值对
for key, value in data.items():
    # 过滤掉non_free_edges中"edges": ["0"]的元素
    value['non_free_edges'] = [item for item in value['non_free_edges'] if item['edges'] != ["0"]]

# 将修改后的数据写回type.json文件
with open('type.json', 'w', encoding='UTF-8') as file:
    json.dump(data, file, indent=4)

print("处理完成，已删除所有'edges': ['0']的元素。")