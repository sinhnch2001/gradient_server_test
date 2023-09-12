import numpy as np
import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import evaluate
import datasets
import re
import json

# https://github.com/huggingface/evaluate/issues/428
from datasets import DownloadConfig
from sklearn.metrics import f1_score

def formatstring(label):
    print(label)
    # Split the string at '||' and create a list of slots_of_domains
    # "<TYPE> TOD <ACTION> negate|restaurants_2-none-none ~ thank|general-none-none <STATE> restaurants_2-slot10-sfo ~ restaurants_2-slot0-namu gaji ~ restaurants_2-slot2-1 pm ~ restaurants_2-slot1-march 1st ~ restaurants_2-slot8-2"

    type_current = label.split('<ACTION>')
    # ["<TYPE> TOD ", " negate|restaurants_2-none-none ~ thank|general-none-none <STATE> restaurants_2-slot10-sfo ~ restaurants_2-slot0-namu gaji ~ restaurants_2-slot2-1 pm ~ restaurants_2-slot1-march 1st ~ restaurants_2-slot8-2"]

    type = "TOD" if "TOD" in type_current[0] else "ODD" # "<TYPE> TOD "
    current_action_state = type_current[1].split('<STATE>')
    # [" negate|restaurants_2-none-none ~ thank|general-none-none ", " restaurants_2-slot10-sfo ~ restaurants_2-slot0-namu gaji ~ restaurants_2-slot2-1 pm ~ restaurants_2-slot1-march 1st ~ restaurants_2-slot8-2"]

    list_current_action = current_action_state[0].split("~")
    # [" negate|restaurants_2-none-none ", " thank|general-none-none "]

    list_current_state = current_action_state[1].split("~")
    # [" restaurants_2-slot10-sfo ", " restaurants_2-slot0-namu gaji ", " restaurants_2-slot2-1 pm ", " restaurants_2-slot1-march 1st ", " restaurants_2-slot8-2"]

    for i in range(len(list_current_action)):
        list_current_action[i] = list_current_action[i].strip().replace("|","-")
    for i in range(len(list_current_state)):
        list_current_state[i] = list_current_state[i].strip()
    if "" in list_current_action and len(list_current_action) == 1:
        list_current_action = []
    if "" in list_current_state and len(list_current_state) == 1:
        list_current_state = []

    return type, list_current_action, list_current_state

# test = json.load(open(r"C:\ALL\OJT\SERVER\gradient_server_test\data\data interim\GradSearch\FUSEDCHAT\test.json"))
# all_output = {}
# for dial in test:
#     label = dial["label"]
#     output = formatstring(label)
#     all_output.setdefault(label, output)
# with open(r"C:\ALL\OJT\SERVER\gradient_server_test\data\data interim\GradSearch\FUSEDCHAT\test_output.json", 'w') as f:
#     json.dump(all_output, f, indent=4)

class Metric:
    def __init__(self, metric_name):
        self.metric_name = metric_name
        self.predict_slot = []
        self.label_slot = []
        self.num_slot_domain_fusedchat = 65
        self.seen_ketod = ['SERVICES_1', 'CALENDAR_1', 'RIDESHARING_2', 'MUSIC_2', 'SERVICES_2', 'HOTELS_3', 'HOTELS_1', 'HOMES_1', 'BUSES_2', 'RIDESHARING_1', 'TRAVEL_1', 'MEDIA_1', 'WEATHER_1', 'EVENTS_1', 'MUSIC_1', 'MOVIES_1', 'FLIGHTS_1', 'RESTAURANTS_1', 'RENTALCARS_2', 'BUSES_1', 'SERVICES_3', 'RENTALCARS_1', 'EVENTS_2', 'FLIGHTS_2', 'HOTELS_2']
        self.unseen_ketod = ['SERVICES_4', 'HOMES_2', 'MUSIC_3', 'TRAINS_1', 'MEDIA_3', 'PAYMENT_1', 'MESSAGING_1', 'RESTAURANTS_2', 'BUSES_3', 'MOVIES_3', 'EVENTS_3', 'FLIGHTS_4', 'RENTALCARS_3', 'HOTELS_4']

        if self.metric_name == "rouge" or self.metric_name == "bleu" or self.metric_name == "bertscore":
            self.metric = evaluate.load(self.metric_name)
        elif self.metric_name == "bleurt":
            self.metric = evaluate.load(self.metric_name, "bleurt-base-128", download_config=DownloadConfig(use_etag=False))

    def add_batch(self, decoded_preds, decoded_labels):
        if self.metric_name == "rouge" or self.metric_name == "bleu" or self.metric_name == "bleurt" or self.metric_name == "bertscore" or self.metric_name == "bleurt":
            self.metric.add_batch(
                predictions=decoded_preds,
                references=decoded_labels)
        elif self.metric_name == "f1":
            for i in range(len(decoded_preds)):
                self.predict_slot.append(decoded_preds[i])
                self.label_slot.append(decoded_labels[i])
        else:
            for i in range(len(decoded_preds)):
                _,_,p_slot = formatstring(decoded_preds[i])
                _,_,l_slot = formatstring(decoded_labels[i])
                self.predict_slot.append(p_slot)
                self.label_slot.append(l_slot)

    def compute(self):
        if self.metric_name == "rouge":
            result_rouge = self.metric.compute(use_stemmer=True)
            result_rouge = {k: round(v * 100, 4) for k, v in result_rouge.items()}
            result = result_rouge

        elif self.metric_name == "bleu":
            result_bleu = self.metric.compute()
            for k, v in result_bleu.items():
                if k == 'precisions':
                    for i in range(len(v)):
                        result_bleu['precisions'][i] = round(v[i] * 100, 4)
                else:
                    result_bleu[k] = round(v * 100, 4)
            result = result_bleu

        elif self.metric_name == "bertscore":
            result_bert = self.metric.compute(model_type="distilbert-base-uncased")
            result_bert["precision"] = round(np.mean(result_bert["precision"]) * 100, 4)
            result_bert["recall"] = round(np.mean(result_bert["recall"]) * 100, 4)
            result_bert["f1"] = round(np.mean(result_bert["f1"]) * 100, 4)
            result = result_bert

        elif self.metric_name == "bleurt":
            result_bleurt = self.metric.compute()
            result_bleurt["scores"] = round(np.mean(result_bleurt["scores"])*100, 4)
            result = result_bleurt

        elif self.metric_name == "f1":
            f1_total = f1_score(self.label_slot, self.predict_slot, average="weighted")
            result = {"F1": round(f1_total * 100, 4)}

        elif self.metric_name == "jga":
            JGA_total = []
            JGA_seen = []
            JGA_unseen = []
            for index in range(0, len(self.label_slot)):
                JGA = 1 if set(self.label_slot[index]) == set(self.predict_slot[index]) else 0
                JGA_total.append(JGA)
                for slot in self.label_slot[index]:
                    if slot.split("-")[0].upper() in self.unseen_ketod:
                        JGA_unseen.append(JGA)
                        break
                    if slot == self.label_slot[index][-1]:
                        JGA_seen.append(JGA)
            if len(JGA_seen) > 0 and len(JGA_unseen) > 0:
                result = {"JGA_avg":round(sum(JGA_total)/len(JGA_total)*100, 4),
                          "JGA_seen":round(sum(JGA_seen)/len(JGA_seen)*100, 4),
                          "JGA_unseen":round(sum(JGA_unseen)/len(JGA_unseen)*100, 4),
                          "JGA_list": JGA_total}
            else:
                result = {"JGA_avg":round(sum(JGA_total)/len(JGA_total)*100, 4),
                          "JGA_list": JGA_total}
        
        elif self.metric_name == "rsa":
            RSA_total = []
            for index in range(0, len(self.label_slot)):
                T = set(self.label_slot[index]) | set(self.predict_slot[index])
                M = set(self.label_slot[index]) - set(self.predict_slot[index])
                W = set(self.predict_slot[index]) - set(self.label_slot[index])
                if len(T) > 0:
                    RSA = len(T-M-W)/len(T)
                    RSA_total.append(RSA)
            result = {"RSA":round(sum(RSA_total)/len(RSA_total)*100, 4)}

        elif self.metric_name == "sa":
            SA_total = []
            for index in range(0, len(self.label_slot)):
                T = self.num_slot_domain_fusedchat
                M = set(self.label_slot[index]) - set(self.predict_slot[index])
                W = set(self.predict_slot[index]) - set(self.label_slot[index])
                if T > 0:
                    SA = (T-len(M)-len(W))/T
                    SA_total.append(SA)
            result = {"Slot Accuracy":round(sum(SA_total)/len(SA_total)*100, 4)}

        elif self.metric_name == "aga":
            AGA_total = []
            AGA_seen = []
            AGA_unseen = []
            for index in range(0, len(self.label_slot)):
                AGA = len(set(self.label_slot[index]).intersection(set(self.predict_slot[index]))) / len(set(self.label_slot[index]))
                AGA_total.append(AGA)
                for slot in self.label_slot[index]:
                    if slot.split("-")[0] in self.unseen_ketod:
                        AGA_unseen.append(AGA)
                        break
                    if slot == self.label_slot[index][-1]:
                        AGA_seen.append(AGA)
            if len(AGA_seen) > 0 and len(AGA_unseen) > 0:
                result = {"AGA_total": round(sum(AGA_total) / len(AGA_total) * 100, 4),
                          "AGA_seen": round(sum(AGA_seen) / len(AGA_seen) * 100, 4),
                          "AGA_unseen": round(sum(AGA_unseen) / len(AGA_unseen) * 100, 4),
                          "AGA_list": AGA_total}
            else:
                result = {"AGA_total": round(sum(AGA_total) / len(AGA_total) * 100, 4),
                          "AGA_list": AGA_total}

        return result


# ketod_dst_tod = json.load(open("C:\ALL\OJT\server\gradient_server_test\data\\new dst\ketod_dst_tod.json"))
# decoded_labels = []
# decoded_preds = []
# for sample in ketod_dst_tod:
#     decoded_labels.append(sample["label"])
#     decoded_preds.append(sample["predict"])
# metric = Metric("sa")
# metric.add_batch(decoded_preds=decoded_preds, decoded_labels=decoded_labels)
# print(  metric.predict_full[61], "\n",
#         metric.label_full[61], "\n",
#         metric.predict_slot[61], "\n",
#         metric.label_slot[61])
# print(  metric.predict_full[313], "\n",
#         metric.label_full[313], "\n",
#         metric.predict_slot[313], "\n",
#         metric.label_slot[313])
# print(  metric.predict_full[63], "\n",
#         metric.label_full[63], "\n",
#         metric.predict_slot[63], "\n",
#         metric.label_slot[63])


