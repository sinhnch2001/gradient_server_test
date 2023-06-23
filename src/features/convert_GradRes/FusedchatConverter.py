import json
import random
import glob
import os

from src.features.converter import DialConverter
from typing import List, Dict, Tuple, Union, Optional
from pathlib import Path


class FusedchatConverter(DialConverter):
    def __init__(self,
                 file_path: str,
                 save_path: str,
                 tag_speaker: str = 'USER',
                 tag_agent: str = 'AGENT',
                 style_tod: List[str] = ["politely"],
                 style_odd: List[str] = ["empathically", "safety", "friendly"],
                 window_context: int = 0,
                 ) -> None:
        """
        Args:
            save_path: path to save the processed dataset
        """
        super().__init__(file_path,
                         save_path,
                         tag_speaker,
                         tag_agent,
                         window_context,
                         )
        self.style_tod = style_tod
        self.style_odd = style_odd

    def __call__(self, instruct_path, ontolopy_path):
        print(f"Start  processing {self.__class__.__name__}")
        self.process(instruct_path=instruct_path, ontolopy_path=ontolopy_path)
        print(f"Finish processing {self.__class__.__name__} at {self.save_path}")

    def process(self, instruct_path, ontolopy_path):
        """Implement your convert logics in this function
            1. `instruction`: instruction for zero-shot learning
            2. `context`: means dialogue history
            3. `state_of_user`: a support document to response current
                user's utterance
            4. `system_action`: what system intends to do
            5. `response`: label for response generation
        A dataset after being formatted should be a List[Dict] and saved
        as a .json file
        Note: text preprocessing needs to be performed during transition
                from crawl dataset to module3-based format
        """
        # Separate complete dialogues to sub dialogues
        data_path_list = glob.glob(os.path.join(self.file_path,'*.json'))
        for data_path in data_path_list:
            filename = os.path.basename(data_path)

            dataset = self.load_datapath(data_path)
            dialogues = self.get_sub_dialogues(dataset)

            list_instructions = self.define_instruct(instruct_path)
            list_ontologies = self.define_ontology(ontolopy_path)

            # Analyze all dialogues
            list_sample_dict = []
            for dialogue in dialogues:
                instruction = self.get_instruction(list_instructions)
                context = self.get_context(dialogue)
                ontology = self.get_ontology(dialogue, list_ontologies)
                system_action = self.get_system_action(dialogue, list_ontologies)
                documents = self.get_documents()
                style = self.get_style('tod') if ontology != "" else self.get_style()
                response = self.get_response(dialogue)

                sample_dict = {
                    "instruction": instruction,
                    "context": context,
                    "ontology": ontology,
                    "system_action": system_action,
                    "documents": documents,
                    "style": style,
                    "response": response,
                }
                list_sample_dict.append(sample_dict)

            self.save_datapath(list_sample_dict, filename)

    def load_datapath(self, data_path) -> List[Dict]:
        with open(data_path, 'r+') as f:
            dataset = json.load(f)
        return dataset

    def define_instruct(self, instruct_path) -> List[str]:
        with open(instruct_path) as f:
            instructions = f.readlines()
        return instructions

    def define_ontology(self, ontolopy_path: Optional[str] = None) -> Union[Dict, List[Dict]]:
        with open(ontolopy_path) as f:
            ontologies = json.load(f)
        return ontologies

    def map_ontology(self, ontologies, list_domains):
        map_ontology_domain = {}
        count = 0
        for domain in list_domains:
            map_slot = {}
            for slot in ontologies[domain.lower().strip()].keys():
                map_slot.setdefault(slot, "slot" + str(count))
                map_ontology_domain.setdefault(domain, map_slot)
                count = count + 1
        return map_ontology_domain

    def save_datapath(self, data_processed: List[Dict], filename: str):
        with open(os.path.join(self.save_path, filename), 'w') as f:
            json.dump(data_processed, f, indent=4)

    def get_sub_dialogues(self, dataset):
        sub_dialogues = []
        for i_dialogue in range(len(dataset)):

            # Get all log in a dialogue
            log = dataset[i_dialogue]["log"]
            [log[j].update({"speaker": "USER"} if j % 2 == 0 else {"speaker": "SYSTEM"}) for j in range(len(log))]
            dialog_action = dataset[i_dialogue]["dialog_action"]
            [log[int(j)].update(dialog_action[j]) for j in dialog_action]

            # Scan all log in the dialogue
            for i_log in range(len(log)):

                # Separate one sample from first turn to each system turn
                if i_log % 2 == 1:
                    """
                        if self.num_of_utterances == 5
                            EX 0: i_turn = 3
                            i_turn - self.num_of_utterances = -2
                            sub_dialogues.append(turns[0:4]) context: 0,1,2 response: 3 

                            EX 1: i_turn = 5
                            i_turn - self.num_of_utterances = 0
                            sub_dialogues.append(turns[0:6]) context: 0,...,4 response: 5

                            EX 2: i_turn = 7
                            i_turn - self.num_of_utterances = 2
                            sub_dialogues.append(turns[2:8]) context: 2,...,6 response: 7
                    """
                    if i_log - self.window_context < 0:
                        sub_dialogues.append(log[0:i_log + 1])
                    else:
                        sub_dialogues.append(log[i_log - self.window_context:i_log + 1])
        return sub_dialogues

    def get_instruction(self, list_instructions):
        random_instruction = list_instructions[random.randint(0, len(list_instructions) - 1)]
        return random_instruction[:-1]

    def get_context(self, dialogue):
        context = ""
        """
            Concat text with length for context
            Ex: context = <tag_speaker> .... <tag_agent> .... <tag_speaker>
        """
        for i in range(len(dialogue) - 1):
            turn = dialogue[i]
            if turn["speaker"] == "USER":
                context = context + self.tag_speaker + ": " + turn["text"] + " "
            elif turn["speaker"] == "SYSTEM":
                context = context + self.tag_agent + ": " + turn["text"] + " "
        return context.strip()

    def get_ontology(self, dialogue, ontologies):
        ontology = ""
        list_domain_dialogue = []
        dialog_act = dialogue[-1]["dialog_act"]
        for domain_action, list_slots in dialog_act.items():
            if domain_action != "chitchat":
                domain, action = domain_action.split('-')
                if domain.lower().strip() != "general":
                    if domain.lower().strip() not in list_domain_dialogue:
                        list_domain_dialogue.append(domain.lower().strip())
        map_ontology_domain = self.map_ontology(ontologies, list_domain_dialogue)
        for domain in list_domain_dialogue:
            ontology = ontology + domain.upper() + ":("
            for slot, description in ontologies[domain].items():
                ontology = ontology + map_ontology_domain[domain][slot] + "=" + description + ";"
            ontology = ontology[:-1] + ") || "
        return ontology[:-4]

    def get_system_action(self, dialogue, ontologies):
        system_action = ""
        actions_dict = {}
        list_domain_dialogue = []
        """
            actions_list = {"OFFER": [slot=value, slot=value, ....], 
                            "INFORM": [slot=value, slot=value, ....],
                            .....,
                            "REQUEST": [slot=value, slot=value, ....]}
        """
        dialog_act = dialogue[-1]["dialog_act"]
        for domain_action, list_slots in dialog_act.items():
            if domain_action != "chitchat":
                domain, action = domain_action.split('-')
                if domain.lower().strip() != "general":
                    if domain.lower().strip() not in list_domain_dialogue:
                        list_domain_dialogue.append(domain.lower().strip())

        map_ontology_domain = self.map_ontology(ontologies, list_domain_dialogue)
        for domain_action, list_slots in dialog_act.items():
            if domain_action != "chitchat":
                domain, action = domain_action.split('-')
                if domain.lower().strip() != "general":
                    for slot in list_slots:
                        slot_system = slot[0].strip().lower()
                        value = slot[1].strip().lower()
                        if slot_system != 'none':
                            if action not in actions_dict.keys():
                                actions_dict.setdefault(action, [])
                            for i_act in range(len(actions_dict[action])):
                                if map_ontology_domain[domain.lower().strip()][slot_system] in actions_dict[action][
                                    i_act]:
                                    actions_dict[action][i_act] = actions_dict[action][i_act] + "|" + value
                                    break
                            else:
                                actions_dict[action].append(
                                    map_ontology_domain[domain.lower().strip()][slot_system] + "=" + value)

        for action, slot_value in actions_dict.items():
            if len(system_action) > 0:
                system_action = system_action + ", " + action.upper()
            else:
                system_action = system_action + action.upper()
            if len(slot_value) > 0:
                system_action = system_action + ":" + "(" + ", ".join([sv for sv in slot_value]) + ")"

        return system_action.strip()

    def get_documents(self):
        '''
        For FushedChat - documents ODD is empty
        '''
        return ""

    def get_style(self, type: str = 'None'):
        if type == 'tod':
            return self.style_tod[random.randint(0, len(self.style_tod) - 1)]
        return self.style_odd[random.randint(0, len(self.style_odd) - 1)]

    def get_response(self, dialogue):
        response = dialogue[-1]["text"]
        return response


if __name__ == '__main__':
    # TEST
    fusedchat_converter = FusedchatConverter(
        file_path='C:\ALL\OJT\gradients.baselinev1.dialogstate\data\\raw\Fusedchat',
        save_path='C:\ALL\OJT\gradients.baselinev1.dialogstate\data\interim\module 3\Fusedchat',
        window_context=5).__call__(
        instruct_path='C:\ALL\OJT\gradients.baselinev1.dialogstate\data\instructions_module_3.txt',
        ontolopy_path='C:\ALL\OJT\gradients.baselinev1.dialogstate\data\processed_schema\schema_final.json')