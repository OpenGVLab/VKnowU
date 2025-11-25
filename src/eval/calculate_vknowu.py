from collections import defaultdict
import json
import re
acc = defaultdict(list)
total = defaultdict(int)

file_name = "./eval_results/eval_vknowu/xxx_xxx.json"
with open(file_name,"r") as f:
    result = json.load(f)["results"]
answer_set = ["A", "B", "C", "D", "E"]
for item in result: 
    category, num = item["qid"].split("@")   
    pred = item["prediction"]
    if pred not in answer_set:
        if "<answer>" in pred:
            pred = re.search(r"<answer>(.*)</answer>", pred).group(1)
        elif pred in item["options"]:
            pred = chr(item["options"].index(pred) + ord("A"))
        else:
            print(item["qid"])
    
    gt = re.search(r"<answer>(.*)</answer>", item["solution"]).group(1)
    if pred == gt:
        acc[category].append(True)
    else:
        acc[category].append(False)
    total[category] += 1

yes = 0
for k, v in sorted(acc.items(), key=lambda item: item[1]):
    print(f"category: {k}, Accuracy: {sum(v)/total[k]*100:.2f}, num_questions: {total[k]}")
    yes += sum(v)
print(f"Overall Accuracy: {yes/sum(total.values())*100:.2f}")