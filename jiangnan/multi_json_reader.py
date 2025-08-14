import json

def split_json_objects(json_stream):
    """
    通过括号匹配切割JSON流为单个JSON对象
    """
    json_objects = []
    bracket_count = 0
    start_pos = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(json_stream):
        # 处理字符串内的引号（避免字符串内的括号被计算）
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            continue
            
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
            
        # 只在字符串外部计算括号
        if not in_string:
            if char == '{':
                bracket_count += 1
            elif char == '}':
                bracket_count -= 1
                
                # 当括号计数回到0时，说明一个完整的JSON对象结束
                if bracket_count == 0:
                    json_str = json_stream[start_pos:i+1]
                    try:
                        json_obj = json.loads(json_str)
                        json_objects.append(json_obj)
                    except json.JSONDecodeError as e:
                        print(f"解析JSON对象时出错: {e}")
                    
                    # 寻找下一个JSON对象的开始
                    start_pos = i + 1
                    while start_pos < len(json_stream) and json_stream[start_pos].isspace():
                        start_pos += 1
                    i = start_pos - 1  # 调整索引
    
    return json_objects

def multi_json_reader(file_path):
    json_objects = []
    items_objects = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # 返回解析后的JSON对象列表
        json_objects = split_json_objects(content)

        for obj in json_objects:
            items_objects.extend(obj.get('data', {}).get('items', []))
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(items_objects, f, ensure_ascii=False, indent=4)

 
if __name__ == "__main__":
    json_objects = multi_json_reader(r"C:\Users\aa666aa666\Desktop\jiangnan-1\jiangnan\test.json")