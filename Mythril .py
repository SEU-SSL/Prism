import subprocess
import json

# 定义字节码列表，这里只是示例，你需要替换为实际的字节码
bytecodes = [
    "0x608060405234801561001057600080fd5b50600436106100365760003560e01c80632e64cec11461003b575b600080fd5b61004361005f565b6040518082815260200191505060405180910390f35b6060604051808381526020018281526020019250505060405180910390f3fea2646970667358221220123456789abcdef123456789abcdef123456789abcdef123456789abcdef64736f6c63430008070033",
    # 可以添加更多字节码
]



# pip install mythril
for index, bytecode in enumerate(bytecodes):
    try:
        # 构建 Mythril 命令
        command = f"myth analyze --code {bytecode} -o json"
        # 执行命令并捕获输出
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            # 解析 JSON 结果
            analysis_result = json.loads(result.stdout)
            # 可以在这里对 analysis_result 进行进一步处理，比如保存到文件
            with open(f"mythril_result_{index}.json", "w") as f:
                json.dump(analysis_result, f, indent=4)
        else:
            print(f"Error analyzing bytecode {index}: {result.stderr}")
    except Exception as e:
        print(f"An error occurred while analyzing bytecode {index}: {e}")