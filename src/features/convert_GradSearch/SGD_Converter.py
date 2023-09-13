import sys
import json
import random
import glob
import os
from typing import List, Dict, Union, Optional
from src.features.converter import DialConverter

sys.path.append('/content/drive/MyDrive/Colab Notebooks/baseline_v1/gradients.baselinev1.dialogstate')
list_sgd_domain = ['buses_1', 'buses_2', 'buses_3', 'calendar_1', 'events_1', 'events_2', 'events_3',
                   'flights_1', 'flights_2', 'flights_4', 'homes_1', 'homes_2', 'hotels_1', 'hotels_2',
                   'hotels_3', 'hotels_4', 'media_1', 'media_3', 'messaging_1', 'movies_1', 'movies_3',
                   'music_1', 'music_2', 'music_3', 'payment_1', 'rentalcars_1', 'rentalcars_2', 'rentalcars_3',
                   'restaurants_1', 'restaurants_2', 'ridesharing_1', 'ridesharing_2', 'services_1',
                   'services_2', 'services_3', 'services_4', 'trains_1', 'travel_1', 'weather_1']

list_user_action = ['INFORM', 'REQUEST','INFORM_INTENT','NEGATE_INTENT',
                    'AFFIRM_INTENT', 'AFFIRM','NEGATE','SELECT','THANK',
                    'BYE','GREET','ASKING','REQUEST_ALTS']

class SGDConverter(DialConverter):
    def __init__(self,
                 file_path: str,
                 save_path: str,
                 tag_user: str = 'USER',
                 tag_system: str = 'SYSTEM',
                 ) -> None:
        """
        Args:
            save_path: path to save the processed dataset
        """
        super().__init__(file_path, save_path, tag_user, tag_system)

    def __call__(self, instruction_path, ontolopy_path):
        print(f"Start  processing {self.__class__.__name__}")
        self.process(instruction_path=instruction_path, ontolopy_path=ontolopy_path)
        print(f"Finish processing {self.__class__.__name__} at {self.save_path}")

    def process(self, instruction_path, ontolopy_path) -> None:
        # Separate complete dialogues to sub dialogues
        list_data_path = glob.glob(os.path.join(self.file_path, '*.json'))
        list_instruction = self.load_instruction(instruction_path)
        list_ontology = self.load_ontology(ontolopy_path)

        for data_path in list_data_path:
            filename = os.path.basename(data_path)
            dataset = self.load_datapath(data_path)

            list_all_sample = []
            # Analyze all dialogues
            for dialogue in dataset:
                # get summarize dialogue
                list_gold_domain = self.get_list_gold_domain(dialogue, list_ontology)
                # process dialogue into sub-dialogues
                list_sub_dialogue = []
                for i in range(len(dialogue['turns'])):
                    if i % 2 == 0:
                        list_sub_dialogue.append(dialogue['turns'][0:i + 1])
                # process raw list_sub_dialogue to interim list_sub_sample
                list_sub_sample = self.get_list_sub_sample(list_sub_dialogue, list_gold_domain, list_ontology,
                                                           list_instruction)
                list_all_sample.extend(list_sub_sample)
            self.save_datapath(list_all_sample, filename)

    def get_list_sub_sample(self, list_sub_dialogue, list_gold_domain, list_ontology, list_instruction):
        list_sub_sample = []
        dict_state_one_dialogue = dict()
        for _,sub_dialogue in enumerate(list_sub_dialogue):
            item = dict()

            # get context, current_user, instruction, list_user_action and ontology
            list_turn = []
            for turn in sub_dialogue:
                speaker = self.tag_user if turn['speaker'] == 'USER' else self.tag_system
                list_turn.append(speaker + ": " + turn['utterance'])

            item['instruction'] = self.get_instruction(list_instruction).strip()
            item['history'] = ' '.join([list_turn[i].strip() for i in range(len(list_turn)-1)]).strip()
            item['current'] = list_turn[-1].strip()
            item['list_user_action'] = '[' + ', '.join(action.lower() for action in list_user_action) + ']'
            item['ontology'] = ' || '.join(gold_domain for gold_domain in list_gold_domain).strip()

            # get type, current_state and current_action
            list_type, list_current_state, list_current_action = set(), list(), set()
            frames = sub_dialogue[-1]['frames']

            for frame in frames:
                domain = frame["service"].lower()
                actions = frame["actions"]
                state = frame['state']
                slot_values = state["slot_values"]
                onto_mapping = self.map_ontology(domain, list_ontology)

                for action in actions:
                    act = action["act"].lower().strip()
                    if act in ["affirm", "negate", "select", "negate_intent", "request_alts", "affirm_intent"]:
                        list_current_action.add(act + ">" + domain + '-none-none')
                        list_type.add("TOD")
                    elif act in ["thank_you"]:
                        list_current_action.add("thank>general-none-none")
                        list_type.add("ODD")
                    elif act in ["goodbye"]:
                        list_current_action.add("bye>general-none-none")
                        list_type.add("ODD")
                    else:
                        slot = action["slot"].strip().lower()
                        for slotstr, description_listslots in onto_mapping.items():
                            for description, listslots in description_listslots.items():
                                if slot in listslots:
                                    slot = slotstr
                        if act == "request":
                            value = "?"
                        else:
                            values = action["values"]
                            for v in values:
                                if v.lower() in item['current']:
                                    value = v.strip().lower()
                                    break
                            else:
                                value = values[0].strip().lower()
                        list_current_action.add(act + ">" + domain + '-' + slot + '-' + value)
                        list_type.add("TOD")

                for slot, values in slot_values.items():
                    for slotstr, description_listslots in onto_mapping.items():
                        for description, listslots in description_listslots.items():
                            if slot.lower() in listslots:
                                slot = slotstr
                    for v in values:
                        if v.lower() in item['current']:
                            value = v.strip().lower()
                            break
                    else:
                        value = values[0].strip().lower()

                    if domain not in dict_state_one_dialogue.keys():
                        dict_state_one_dialogue.setdefault(domain, dict())
                    if slot not in dict_state_one_dialogue[domain].keys():
                        dict_state_one_dialogue[domain].setdefault(slot, value)
                    if value != dict_state_one_dialogue[domain][slot]:
                        dict_state_one_dialogue[domain][slot] = value

            for domain, dict_slot_value in dict_state_one_dialogue.items():
                for slot, value in dict_slot_value.items():
                    list_current_state.append(domain + '-' + slot + '-' + value)

            final_type = "TOD" if "TOD" in list_type else "ODD"
            final_current_action = ' | '.join(current_action for current_action in list_current_action).lower().strip()
            final_current_state = ' | '.join(current_state for current_state in list_current_state).lower().strip()

            item['label'] = "(type) " + final_type + " (current action) " + final_current_action + " (current state) " + final_current_state
            list_sub_sample.append(item)
        return list_sub_sample

    # LOAD and SAVE data
    def load_datapath(self, data_path) -> List[Dict]:
        with open(data_path, 'r+') as f:
            dataset = json.load(f)
        return dataset

    def save_datapath(self, data_processed: List[Dict], filename: str):
        with open(os.path.join(self.save_path, filename), 'w') as f:
            json.dump(data_processed, f, indent=4)

    # LOAD and GET instruction
    def load_instruction(self, instruct_path) -> List[str]:
        with open(instruct_path, encoding="utf8") as f:
            instructions = f.readlines()
        return instructions

    def get_instruction(self, list_instructions):
        random_instruction = list_instructions[random.randint(0, len(list_instructions) - 1)]
        return random_instruction

    # LOAD and MAP and GET ontology
    def load_ontology(self, ontolopy_path: Optional[str] = None) -> Union[Dict, List[Dict]]:
        with open(ontolopy_path, encoding="utf8") as f:
            ontologies = json.load(f)
        return ontologies

    def map_ontology(self, domain, ontologies, count=0):
        map_ontology_domain = {}
        for description, lists_slot in ontologies[domain.lower()].items():
            map_ontology_domain.setdefault("slot" + str(count), {description: lists_slot})
            count = count + 1
        return map_ontology_domain
        # {  "slot0":{"area to search for attractions": ["area"]},
        #    "slot1":{"name of the attraction": ["name"],
        #    "slot2":{"type of the attraction": ["type"]}}

    def get_ontology(self, domain_name, ontologies):
        onto_mapping = self.map_ontology(domain_name, ontologies)
        tmps = []
        for slotstr, description_listslots in onto_mapping.items():
            tmps.append(slotstr + "=" + list(description_listslots.keys())[0])

        value_onto = domain_name + ":[" + ', '.join(tmp for tmp in tmps) + "]"
        return value_onto, onto_mapping
        # value_onto = DOMAIN:(slot0=des0,slot1=des1,slot2=des2)

    def get_list_gold_domain(self, dialogue, ontology):
        gold_domain = []
        tmp_domain = set()
        goal = dialogue["services"]
        for domain in goal:
            tmp_domain.add(domain.lower())
        for domain in tmp_domain:
            if domain in list_sgd_domain:
                value_onto, _ = self.get_ontology(domain, ontology)
                gold_domain.append(value_onto)
        return gold_domain

if __name__ == '__main__':
    # TEST
    sgd_converter = SGDConverter(
        file_path=r'C:\ALL\OJT\SERVER\gradient_server_test\data\data raw\SGD',
        save_path=r'C:\ALL\OJT\SERVER\gradient_server_test\data\data interim\GradSearch\SGD').__call__(
        instruction_path=r"C:\ALL\OJT\SERVER\gradient_server_test\data\instructions\instruct_GradSearch.txt",
        ontolopy_path=r"C:\ALL\OJT\SERVER\gradient_server_test\data\schema guided\schema.json")












