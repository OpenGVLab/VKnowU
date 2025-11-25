from collections import defaultdict
import json
import re
import os

file_name = "./eval_results/eval_xxx/xxx_xxx.json"

with open(file_name,"r") as f:
    acc = defaultdict(list)
    total = defaultdict(int)
    cnt = 0
    with open(file_name,"r") as f:
        result = json.load(f)["results"]
        answer_set = ["A", "B", "C", "D", "E"]
        for item in result: 
            category = item["problem_type"]
            pred = item["prediction"]
            if pred not in answer_set:
                if "<answer>" in pred:
                    pred = re.findall(r"<answer>(.*?)</answer>", pred)[0]
                elif pred in item["options"]:
                    pred = chr(item["options"].index(pred) + ord("A"))
                # elif len(pred) > 0 and pred[0] in answer_set:
                #     pred = pred[0]
                # elif len(pred) > 0 and pred[-1] in answer_set:
                #     pred = pred[-1]
                # elif len(pred) > 2 and pred[-2] in answer_set and pred[-1]==".":
                #     pred = pred[-2]
                else:
                    # print(item["qid"], pred)
                    # print(item["problem_id"])
                    cnt += 1
            
            answer = re.search(r"<answer>(.*)</answer>", item["solution"]).group(1)
            # print(pred, answer)
            if pred == answer:
                acc[category].append(True)
            else:
                acc[category].append(False)
            total[category] += 1

    yes = 0
    for k, v in sorted(acc.items(), key=lambda item: item[1]):
        # print(f"category: {k}, Accuracy: {sum(v)/total[k]*100:.2f}, num_questions: {total[k]}")
        yes += sum(v)
    print(f"Overall Accuracy: {yes/sum(total.values())*100:.2f}")
    print(f"cnt: {cnt}, total: {sum(total.values())}")